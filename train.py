"""Train Flow Matching or Mean Flow on MNIST.

Usage:
    python train.py --method flow
    python train.py --method meanflow
    python train.py --method wfmeanflow     # weak-form mean flow (no JVP)

All methods share the exact same UNet, optimizer, data order and seeds; the
only differences are the loss and (for mean flow) the extra `r` conditioning.
At every checkpoint the last-5 window is pruned and 4 EMA-generated images are
logged to wandb.

After EVERY checkpoint, evaluate.py's evaluation (FID / MSE / PSNR / SSIM /
NFE / wall-time) is run on the EMA model and logged to the same wandb run
under `eval/steps_<N>/...` with the training step, so metrics can be compared
across checkpoints (and across the flow / meanflow runs, which use identical
evaluation parameters from Config). Disable with --no-eval-at-ckpt.
"""
import argparse
import copy
import glob
import json
import os
import time

import torch
import torchvision.utils as vutils
import wandb

from config import Config, load_env, resolve_wandb_entity, resolve_wandb_mode
from data import get_mnist_loaders, get_train_tensor
from unet import SmallUNet
from flow_matching import fm_loss, fm_sample
from mean_flow import mf_loss, mf_sample
from weak_mean_flow import wf_loss
from evaluate import CheckpointEvaluator


class EMA:
    """Exponential moving average of model parameters (used for sampling)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def save_checkpoint(step, model, ema, opt, cfg, method, out_dir):
    path = os.path.join(out_dir, f"ckpt_step_{step:07d}.pt")
    torch.save(
        {
            "step": step,
            "method": method,
            "model": model.state_dict(),
            "ema": ema.shadow,
            "opt": opt.state_dict(),
            "config": cfg.to_dict(),
        },
        path,
    )
    # keep only the last `keep_ckpts` checkpoints
    ckpts = sorted(glob.glob(os.path.join(out_dir, "ckpt_step_*.pt")))
    for old in ckpts[: max(0, len(ckpts) - cfg.keep_ckpts)]:
        os.remove(old)
    return path


@torch.no_grad()
def log_samples(step, method, ema_model, cfg, device):
    """Generate 4 images with fixed noise (comparable across checkpoints)."""
    ema_model.eval()
    g = torch.Generator(device=device).manual_seed(1234)
    noise = torch.randn((4, cfg.in_channels, cfg.image_size, cfg.image_size),
                        device=device, generator=g)
    if method == "flow":
        samples = fm_sample(ema_model, noise.shape, steps=cfg.fm_sample_steps,
                            device=device, noise=noise)
        caption = f"flow, {cfg.fm_sample_steps} Euler steps, step {step}"
    else:
        samples = mf_sample(ema_model, noise.shape, steps=1,
                            device=device, noise=noise)
        caption = f"{method}, 1 step, step {step}"
    grid = vutils.make_grid((samples + 1.0) / 2.0, nrow=4)
    wandb.log({"samples/generated": wandb.Image(grid, caption=caption)}, step=step)


def eval_at_checkpoint(step, method, ema_model, cfg, device, evaluator, eval_log_path):
    """Run evaluate.py's evaluation on the EMA model and log every metric to
    the training wandb run at this step (curves comparable across checkpoints)
    plus append the full result to eval_log.jsonl. Returns the evaluator
    (lazily created on first call so real-data stats are computed only once).
    """
    if evaluator is None:
        evaluator = CheckpointEvaluator(cfg, device,
                                        num_samples=cfg.ckpt_eval_num_samples,
                                        num_real=cfg.ckpt_eval_num_real,
                                        seed=cfg.ckpt_eval_seed)
    results, _ = evaluator.evaluate(method, ema_model, list(cfg.ckpt_eval_steps))
    payload = {}
    for key, metrics in results.items():
        for name, val in metrics.items():
            payload[f"eval/{key}/{name}"] = val
    wandb.log(payload, step=step)
    with open(eval_log_path, "a") as f:
        f.write(json.dumps({
            "step": step, "method": method,
            "eval_params": {"num_samples": cfg.ckpt_eval_num_samples,
                            "num_real": cfg.ckpt_eval_num_real,
                            "steps": list(cfg.ckpt_eval_steps),
                            "seed": cfg.ckpt_eval_seed},
            "results": results,
        }) + "\n")
    summary = " | ".join(
        f"{key}: FID={m['fid']:.2f} SSIM={m['ssim']:.3f} ({m['nfe']} NFE)"
        for key, m in results.items())
    print(f"[{method}] eval @ step {step}: {summary}")
    return evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["flow", "meanflow", "wfmeanflow"], required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=0, help=">0 overrides epochs (smoke test)")
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    # ---- per-checkpoint evaluation (defaults come from Config -> identical
    # for both methods; only override all of them together to keep it fair)
    p.add_argument("--no-eval-at-ckpt", dest="eval_at_ckpt", action="store_false",
                   default=None, help="disable evaluation after every checkpoint")
    p.add_argument("--eval-num-samples", type=int, default=None,
                   help="generated samples per checkpoint eval (default: Config)")
    p.add_argument("--eval-num-real", type=int, default=None,
                   help="real images for the FID reference (default: Config)")
    p.add_argument("--eval-steps", nargs="+", type=int, default=None,
                   help="NFE sweep for checkpoint eval, applied to BOTH methods")
    p.add_argument("--eval-seed", type=int, default=None,
                   help="fixed noise / real-data seed for checkpoint eval")
    args = p.parse_args()

    load_env()
    cfg = Config()
    for name in ("epochs", "batch_size", "lr", "ckpt_every", "log_every", "seed"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    if args.eval_at_ckpt is not None:
        cfg.eval_at_ckpt = args.eval_at_ckpt
    if args.eval_num_samples is not None:
        cfg.ckpt_eval_num_samples = args.eval_num_samples
    if args.eval_num_real is not None:
        cfg.ckpt_eval_num_real = args.eval_num_real
    if args.eval_steps is not None:
        cfg.ckpt_eval_steps = tuple(args.eval_steps)
    if args.eval_seed is not None:
        cfg.ckpt_eval_seed = args.eval_seed
    out_dir = args.out_dir or os.path.join(cfg.ckpt_root, args.method)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = get_mnist_loaders(cfg)
    model = SmallUNet(cfg.in_channels, cfg.base_ch, cfg.ch_mult, cfg.time_dim,
                      use_r=(args.method != "flow")).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"method={args.method} params={n_params/1e6:.2f}M device={device}")

    ema = EMA(model, cfg.ema_decay)
    ema_model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", cfg.wandb_project),
        entity=resolve_wandb_entity(),
        name=args.run_name or args.method,
        mode=resolve_wandb_mode(args.wandb_mode),
        config={**cfg.to_dict(), "method": args.method, "n_params": n_params},
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps if args.max_steps > 0 else cfg.epochs * steps_per_epoch
    log_path = os.path.join(out_dir, "loss_log.jsonl")
    eval_log_path = os.path.join(out_dir, "eval_log.jsonl")
    evaluator = None          # lazily created at the first checkpoint
    data_pool = None          # wfmeanflow: training images for the auxiliary forcing batch
    if args.method == "wfmeanflow":
        data_pool = get_train_tensor(train_loader).to(device)

    step, done, t0 = 0, False, time.time()
    while not done:
        for x0, _ in train_loader:
            x0 = x0.to(device, non_blocking=True)
            if args.method == "flow":
                loss, metrics = fm_loss(model, x0)
            elif args.method == "meanflow":
                loss, metrics = mf_loss(model, x0, cfg)
            else:
                loss, metrics = wf_loss(model, x0, cfg, data_pool)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)
            step += 1

            if step % cfg.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                wandb.log(
                    {
                        "train/loss": metrics["loss"],
                        "train/raw_mse": metrics["raw_mse"],
                        "train/grad_norm": float(gnorm),
                        "train/steps_per_sec": step / max(elapsed, 1e-9),
                        # method-specific extras (e.g. wfmeanflow weak_loss / anchor_mse)
                        **{f"train/{k}": v for k, v in metrics.items()
                           if k not in ("loss", "raw_mse")},
                    },
                    step=step,
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": metrics["loss"],
                                        "raw_mse": metrics["raw_mse"]}) + "\n")
                print(f"[{args.method}] step {step}/{total_steps} "
                      f"loss={metrics['loss']:.4f} raw_mse={metrics['raw_mse']:.4f}")

            if step % cfg.ckpt_every == 0 or step == total_steps:
                path = save_checkpoint(step, model, ema, opt, cfg, args.method, out_dir)
                ema.copy_to(ema_model)
                log_samples(step, args.method, ema_model, cfg, device)
                print(f"[{args.method}] checkpoint -> {path}")
                if cfg.eval_at_ckpt:
                    evaluator = eval_at_checkpoint(step, args.method, ema_model,
                                                   cfg, device, evaluator, eval_log_path)

            if step >= total_steps:
                done = True
                break

    print(f"[{args.method}] done in {(time.time() - t0)/60:.1f} min")
    run.finish()


if __name__ == "__main__":
    main()