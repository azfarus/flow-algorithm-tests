"""Evaluation pipeline: Flow Matching vs Mean Flow.

For each trained checkpoint this generates N samples at several NFE budgets
(flow: 1/4/8/16/32 Euler steps; meanflow / wfmeanflow: 1 step, same
mf_sample sampler) and computes:
  - FID (InceptionV3-2048 features, torch-fidelity compatible)
  - MSE / PSNR / SSIM against a fixed real reference batch
  - NFE and wall-clock sampling time
Results are written to results/eval_results.json, sample grids to
results/grids/, comparison plots to results/plots/, and everything is logged
to a wandb "evaluation" run.

The core logic lives in `CheckpointEvaluator`, which train.py also imports to
run this exact evaluation after EVERY checkpoint (logging eval/... metrics to
the training wandb run so results can be compared across checkpoints). Both
methods are always evaluated with identical parameters (sample counts, NFE
sweep, seed) so numbers are comparable across methods and checkpoints.

Usage:
    python evaluate.py                      # eval all methods, latest ckpts
    python evaluate.py --num-samples 1000   # quick eval
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb
from pytorch_msssim import ssim as ssim_fn
from scipy import linalg
from torchmetrics.image.fid import NoTrainInceptionV3

from config import Config, load_env, resolve_wandb_entity, resolve_wandb_mode
from data import get_real_images
from unet import SmallUNet
from flow_matching import fm_sample
from mean_flow import mf_sample
from plot import make_all_plots

ROOT = os.path.dirname(os.path.abspath(__file__))


def to_uint8(x):
    """[-1,1] Nx1x28x28 -> uint8 Nx3x28x28 (InceptionV3 input)."""
    x = ((x.clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    return x.repeat(1, 3, 1, 1)


@torch.no_grad()
def extract_features(inception, imgs_uint8, device, batch_size=256):
    feats = []
    for i in range(0, len(imgs_uint8), batch_size):
        feats.append(inception(imgs_uint8[i:i + batch_size].to(device)).cpu())
    return torch.cat(feats).numpy().astype(np.float64)


def fid_from_features(f_real, f_fake):
    mu_r, mu_f = f_real.mean(0), f_fake.mean(0)
    cov_r = np.cov(f_real, rowvar=False)
    cov_f = np.cov(f_fake, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_r @ cov_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_f
    return float(diff @ diff + np.trace(cov_r + cov_f - 2.0 * covmean))


@torch.no_grad()
def generate(method, model, n, steps, device, seed, batch_size=500):
    """Returns (samples in [-1,1], wall_time_sec, nfe). Fixed seeded noise."""
    model.eval()
    sampler = fm_sample if method == "flow" else mf_sample
    g = torch.Generator(device=device).manual_seed(seed)
    chunks = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        noise = torch.randn((b, 1, 28, 28), device=device, generator=g)
        chunks.append(sampler(model, noise.shape, steps=steps, device=device, noise=noise))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(chunks), time.time() - t0, steps


def pair_metrics(fake, real):
    """MSE / PSNR / SSIM between fake and a same-sized real reference batch."""
    mse = F.mse_loss(fake, real).item()
    psnr = 10.0 * math.log10(4.0 / max(mse, 1e-12))   # pixel range is 2
    s = ssim_fn((fake + 1.0) / 2.0, (real + 1.0) / 2.0, data_range=1.0).item()
    return mse, psnr, s


def load_latest_ckpt(method, ckpt_root, device):
    ckpts = sorted(glob.glob(os.path.join(ckpt_root, method, "ckpt_step_*.pt")))
    if not ckpts:
        return None, None
    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
    cfgd = ckpt.get("config", {})
    model = SmallUNet(cfgd.get("in_channels", 1), cfgd.get("base_ch", 64),
                      tuple(cfgd.get("ch_mult", (1, 2, 2))), cfgd.get("time_dim", 256),
                      use_r=(method != "flow")).to(device)
    model.load_state_dict(ckpt["ema"])   # sample with EMA weights
    model.eval()
    return model, ckpt


class CheckpointEvaluator:
    """Reusable evaluator: real-data statistics are computed ONCE, then any
    number of checkpoints (method + model) can be evaluated cheaply.

    Used by this script (final comparison over the latest checkpoints) and by
    train.py (per-checkpoint eval after every save). The parameters
    (num_samples, num_real, seed) and the NFE steps list must be identical for
    both methods so results are comparable across methods and checkpoints.
    """

    def __init__(self, cfg, device, num_samples, num_real, seed):
        self.device = device
        self.num_samples = num_samples
        self.seed = seed
        print("extracting real InceptionV3 features ...")
        self.inception = NoTrainInceptionV3(name="inception-v3-compat",
                                            features_list=["2048"]).to(device).eval()
        real_fid = get_real_images(cfg, num_real, seed=seed)
        self.real_feats = extract_features(self.inception, to_uint8(real_fid), device)
        self.real_ref = get_real_images(cfg, num_samples, seed=seed + 1).to(device)

    @torch.no_grad()
    def evaluate(self, method, model, steps_list):
        """Returns (results, samples): results maps f"steps_{s}" -> metrics dict,
        samples maps f"steps_{s}" -> the generated images in [-1, 1]."""
        results, samples = {}, {}
        for steps in steps_list:
            imgs, wall, nfe = generate(method, model, self.num_samples, steps,
                                       self.device, seed=self.seed)
            fake_feats = extract_features(self.inception, to_uint8(imgs), self.device)
            fid = fid_from_features(self.real_feats, fake_feats)
            mse, psnr, s = pair_metrics(imgs.to(self.device), self.real_ref)
            key = f"steps_{steps}"
            results[key] = {
                "fid": fid, "mse": mse, "psnr": psnr, "ssim": s,
                "nfe": nfe, "wall_time_sec": wall,
                "sec_per_image": wall / self.num_samples,
            }
            samples[key] = imgs
        return results, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["flow", "meanflow", "wfmeanflow"],
                   choices=["flow", "meanflow", "wfmeanflow"])
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--num-real", type=int, default=10000, help="real imgs for FID")
    p.add_argument("--flow-steps", nargs="+", type=int, default=None)
    p.add_argument("--mf-steps", nargs="+", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ckpt-root", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    args = p.parse_args()

    load_env()
    cfg = Config()
    flow_steps = args.flow_steps or list(cfg.fm_eval_steps)
    mf_steps = args.mf_steps or list(cfg.mf_eval_steps)
    grids_dir = os.path.join(args.results_dir, "grids")
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(grids_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", cfg.wandb_project),
        entity=resolve_wandb_entity(),
        name="evaluation",
        mode=resolve_wandb_mode(args.wandb_mode),
        config=vars(args),
    )

    # ---- real data: FID features computed once, reference batch for paired metrics
    evaluator = CheckpointEvaluator(cfg, device, num_samples=args.num_samples,
                                    num_real=args.num_real, seed=args.seed)

    results = {}
    for method in args.methods:
        model, ckpt = load_latest_ckpt(method, args.ckpt_root, device)
        if model is None:
            print(f"[{method}] no checkpoint found under {args.ckpt_root}/{method} - skipping")
            continue
        print(f"[{method}] loaded checkpoint at step {ckpt['step']}")
        steps_list = flow_steps if method == "flow" else mf_steps
        method_results, method_samples = evaluator.evaluate(method, model, steps_list)
        results[method] = {}
        for key, metrics in method_results.items():
            metrics["ckpt_step"] = ckpt["step"]
            results[method][key] = metrics
            grid_path = os.path.join(grids_dir, f"{method}_{key}.png")
            vutils.save_image((method_samples[key][:16] + 1.0) / 2.0, grid_path, nrow=4)
            wandb.log({
                f"eval/{method}/{key}": metrics,
                f"grids/{method}_{key}": wandb.Image(grid_path,
                                                     caption=f"{method} {metrics['nfe']} NFE"),
            })
            print(f"[{method}] {key}: FID={metrics['fid']:.2f} MSE={metrics['mse']:.4f} "
                  f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f} "
                  f"({metrics['wall_time_sec']:.1f}s)")

    results_json = os.path.join(args.results_dir, "eval_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", results_json)

    # ---- comparison plots (also logged to wandb)
    plot_paths = make_all_plots(results_json, args.ckpt_root, plots_dir)
    for path in plot_paths:
        wandb.log({f"plots/{os.path.splitext(os.path.basename(path))[0]}":
                   wandb.Image(path)})
    wandb.log({"eval/summary": results})
    run.finish()
    print("plots:", *plot_paths, sep="\n  ")


if __name__ == "__main__":
    main()
