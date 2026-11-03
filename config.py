"""Shared configuration for the Flow Matching vs Mean Flow experiments."""
import os
from dataclasses import dataclass, asdict


def load_env(path=None):
    """Minimal .env loader (no external dependency). Sets vars only if unset."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def resolve_wandb_entity():
    """Pick a wandb entity: WANDB_ENTITY env var, else the logged-in username
    (needed for accounts without a default entity). Returns None if unknown."""
    ent = os.environ.get("WANDB_ENTITY", "").strip()
    if ent:
        return ent
    try:
        import wandb
        return wandb.Api().viewer.username
    except Exception as e:
        print(f"[wandb] could not resolve entity automatically ({e})")
        return None


def resolve_wandb_mode(requested=None):
    """Pick a wandb mode; fall back to offline if the API key looks unset."""
    if requested:
        return requested
    mode = os.environ.get("WANDB_MODE", "online")
    key = os.environ.get("WANDB_API_KEY", "")
    logged_in = key and not key.startswith("your_")
    netrc = os.path.exists(os.path.expanduser("~/.netrc"))
    if mode == "online" and not (logged_in or netrc):
        print("[wandb] no API key found (see .env) -> falling back to offline mode")
        mode = "offline"
    return mode


@dataclass
class Config:
    # ---- data ----
    data_root: str = "./data"
    image_size: int = 28
    in_channels: int = 1

    # ---- model (identical for both methods) ----
    base_ch: int = 64
    ch_mult: tuple = (1, 2, 2)      # 28x28 -> 14x14 -> 7x7
    time_dim: int = 256

    # ---- training (identical for both methods) ----
    batch_size: int = 256
    epochs: int = 1000
    lr: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    seed: int = 42
    num_workers: int = 4
    log_every: int = 50             # steps between loss logs
    ckpt_every: int = 200          # steps between checkpoints (+ 4 samples logged)
    keep_ckpts: int = 5             # rolling window of last-N checkpoints
    ckpt_root: str = "./checkpoints"

    # ---- sampling ----
    fm_sample_steps: int = 16       # Euler steps for flow matching at ckpt time
    fm_eval_steps: tuple = (1, 4, 8, 16, 32)   # NFE sweep for evaluation
    mf_eval_steps: tuple = (1,)                 # mean flow (and weak-form) is one-step

    # ---- per-checkpoint evaluation ----
    # evaluate.py is run after EVERY checkpoint and results are logged to wandb
    # (eval/... curves vs training step -> comparable across checkpoints).
    # These params are SHARED by both methods so flow and meanflow evals are
    # directly comparable: same sample counts, same NFE sweep, same seed.
    eval_at_ckpt: bool = True            # run evaluation after every checkpoint
    ckpt_eval_num_samples: int = 1000    # generated samples per metric computation
    ckpt_eval_num_real: int = 2000       # real images for the FID reference
    ckpt_eval_steps: tuple = (1, 4, 16)  # NFE sweep applied to BOTH methods
    ckpt_eval_seed: int = 1234           # fixed noise / real-data seed

    # ---- mean flow ----
    mf_ratio_r_eq_t: float = 0.25   # fraction of (r, t) pairs with r == t
    mf_adaptive_c: float = 1e-3     # adaptive loss-weighting constant

    # ---- weak-form mean flow (same sampler / eval steps as mean flow) ----
    wf_anchor_ratio: float = 0.25   # fraction of batch on the diagonal r == t (anchor)
    wf_anchor_weight: float = 1.0   # lambda_a: weight of the diagonal FM anchor
    wf_num_fields: int = 4096*16        # K test fields, resampled every step
    wf_freq_scales: tuple = (0.5, 1.0, 2.0, 4.0)  # frequency ladder for omega
    wf_time_basis: int = 3          # cosine terms per time axis in eta(r, t)
    wf_aux_batch: int = 4096        # N auxiliary samples for the forcing term

    # ---- wandb ----
    wandb_project: str = "meanflow-vs-flow"

    def to_dict(self):
        d = asdict(self)
        d["ch_mult"] = list(self.ch_mult)
        d["fm_eval_steps"] = list(self.fm_eval_steps)
        d["mf_eval_steps"] = list(self.mf_eval_steps)
        d["ckpt_eval_steps"] = list(self.ckpt_eval_steps)
        d["wf_freq_scales"] = list(self.wf_freq_scales)
        return d
