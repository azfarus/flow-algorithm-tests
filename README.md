# Testing the Flows: Flow Matching vs Mean Flow vs Weak-Form Mean Flow on MNIST

A minimal PyTorch (CUDA) playground that trains the **same small UNet** with three
objectives on MNIST and compares them fairly:

| | Flow Matching (`flow`) | Mean Flow (`meanflow`, Geng et al., 2025) | Weak-Form Mean Flow (`wfmeanflow`) |
|---|---|---|---|
| Learns | instantaneous velocity `v(x, t)` | average velocity `u(x, r, t)` (via JVP) | average velocity `u(x, r, t)` (**no JVP**, no stop-gradient target) |
| Loss | MSE to `e - x` | adaptive MSE to `sg[v - (t-r) du/dt]` | weak residual vs random test fields (U-statistic) + FM anchor on `r = t` |
| Sampling | multi-step Euler ODE | **one step** `x0 = z1 - u(z1, 0, 1)` | same sampler as Mean Flow |
| Conditioning | `t` | `t` and `r` | `t` and `r` |

Weak-Form Mean Flow follows `weak-form-meanflow/*.md`: the MeanFlow residual is
tested against random Fourier fields `phi = a eta(r,t) cos(omega.z + beta)` and
integrated by parts, so all derivatives fall on `phi` (analytic). Per step:
`K = wf_num_fields` fresh fields, an unbiased U-statistic over the batch, the
forcing term from an independent auxiliary batch of `wf_aux_batch` training
images (no network call), and a flow-matching anchor on the `wf_anchor_ratio`
fraction of the batch with `r = t`. One deviation from the proposal's defaults:
the output direction `a` is half aligned with `omega` (a purely random `a` is
almost orthogonal to `omega` at d = 784 and the weak loss could not tell `u*`
from `u = 0` in a Gaussian sanity check). Knobs live in `config.py` (`wf_*`).

## Setup

Dependencies (already standard): `torch`, `torchvision`, `wandb`, `torchmetrics`,
`pytorch-msssim`, `scipy`, `matplotlib`.

1. **Add your wandb API key** to `.env`:

   ```bash
   # edit .env
   WANDB_API_KEY=<your key from https://wandb.ai/authorize>
   ```

   The code loads `.env` automatically. If no key is found, runs fall back to
   wandb **offline** mode (`wandb sync` later to upload). You can also force a
   mode with `--wandb-mode {online,offline,disabled}`.

2. (Optional) smoke test everything in ~1 minute:

   ```bash
   python train.py --method flow     --max-steps 50 --ckpt-every 25 --log-every 10
   python train.py --method meanflow --max-steps 50 --ckpt-every 25 --log-every 10
   python train.py --method wfmeanflow --max-steps 50 --ckpt-every 25 --log-every 10
   python evaluate.py --num-samples 256 --num-real 1000
   ```

   Weak-form mean flow alone (no wandb, scratch output dir):

   ```bash
   python train.py --method wfmeanflow --max-steps 50 --ckpt-every 25 --log-every 10 \
       --wandb-mode disabled --out-dir /tmp/wftest/wfmeanflow
   python evaluate.py --methods wfmeanflow --num-samples 256 --num-real 1000 \
       --ckpt-root /tmp/wftest --results-dir /tmp/wftest/results --wandb-mode disabled
   ```

## Running with tmux

```bash
cd /mnt/data/azfarus/projects/testing-the-flows
```

### Each method individually (one session per method)

```bash
tmux new-session -d -s flow       'python train.py --method flow       2>&1 | tee logs_flow.txt'
tmux new-session -d -s meanflow   'python train.py --method meanflow   2>&1 | tee logs_meanflow.txt'
tmux new-session -d -s wfmeanflow 'python train.py --method wfmeanflow 2>&1 | tee logs_wfmeanflow.txt'

tmux attach -t wfmeanflow     # Ctrl+b d to detach
tmux ls                       # list running sessions

# evaluate only the method(s) you trained
tmux new-session -d -s eval 'python evaluate.py --methods wfmeanflow 2>&1 | tee logs_eval.txt'
```

### All three together (one session, one window each)

```bash
tmux new-session -d -s flows -n flow   'python train.py --method flow       2>&1 | tee logs_flow.txt'
tmux new-window  -t flows -n meanflow   'python train.py --method meanflow   2>&1 | tee logs_meanflow.txt'
tmux new-window  -t flows -n wfmeanflow 'python train.py --method wfmeanflow 2>&1 | tee logs_wfmeanflow.txt'

# watch progress
tmux attach -t flows          # Ctrl+b then 0/1/2 to switch windows, Ctrl+b d to detach

# when all three are finished, evaluate in a fourth window (all methods by default)
tmux new-window -t flows -n eval 'python evaluate.py 2>&1 | tee logs_eval.txt'

# cleanup
tmux kill-session -t flows
```

Useful training overrides (shared by all methods for fairness):
`--epochs`, `--batch-size`, `--lr`, `--ckpt-every`, `--seed`, `--run-name`.
Full runs: 20 epochs (~4.7k steps) each.

## Outputs

```
checkpoints/flow/       last 5 ckpts (rolling window) + loss_log.jsonl + eval_log.jsonl
checkpoints/meanflow/   last 5 ckpts (rolling window) + loss_log.jsonl + eval_log.jsonl
checkpoints/wfmeanflow/ last 5 ckpts (rolling window) + loss_log.jsonl + eval_log.jsonl
results/eval_results.json   FID / MSE / PSNR / SSIM / NFE / wall-time
results/grids/              16-image sample grids per method per NFE
results/plots/              loss curves, FID-vs-NFE, metric bars
```

In **wandb** (project `meanflow-vs-flow`): per-step training losses/grad-norms,
4 generated images at every checkpoint (`samples/generated`), per-checkpoint
evaluation curves (`eval/steps_<N>/fid`, `.../mse`, `.../psnr`, `.../ssim`,
`.../nfe`, `.../wall_time_sec` — one point per checkpoint, so metrics can be
compared across checkpoints and across the flow / meanflow / wfmeanflow runs), and an
`evaluation` run with all metrics, grids and comparison plots. `wfmeanflow`
additionally logs `train/weak_loss` (U-statistic weak residual; can be slightly
negative on a batch) and `train/anchor_mse`; its `train/raw_mse` is the anchor
MSE (flow matching on `r = t`).

## Per-checkpoint evaluation

After **every** checkpoint, `train.py` runs `evaluate.py`'s evaluation
(`CheckpointEvaluator`: FID / MSE / PSNR / SSIM / NFE / wall-time) on the EMA
model and logs it to the same wandb run at that training step. The evaluation
parameters are shared by all methods via `Config` (so numbers are directly
comparable): `ckpt_eval_num_samples=1000`, `ckpt_eval_num_real=2000`,
`ckpt_eval_steps=(1, 4, 16)` — the NFE sweep is applied to **all** methods —
and `ckpt_eval_seed=1234`. Results are also appended to
`checkpoints/<method>/eval_log.jsonl`. Override with `--eval-num-samples`,
`--eval-num-real`, `--eval-steps`, `--eval-seed`, or turn it off with
`--no-eval-at-ckpt`. The heavy standalone sweep (`evaluate.py`, 5k samples,
flow at 1/4/8/16/32 NFE) is unchanged and meant for the final comparison.

## Evaluation details

`evaluate.py` loads the latest EMA checkpoint of each method and sweeps the
compute budget: flow at **1/4/8/16/32** NFEs vs mean flow and weak-form mean
flow at **1** NFE
(`--flow-steps`, `--mf-steps` to change). Metrics: FID (InceptionV3-2048,
5k generated vs 10k real), MSE/PSNR/SSIM vs a fixed real reference batch, NFE,
and wall-clock sampling time. All noise is seeded for reproducibility.

## Files

| file | purpose |
|---|---|
| `config.py` | shared hyperparameters, `.env` loader |
| `data.py` | seeded MNIST loaders, real-image helper |
| `unet.py` | small time(+r)-conditioned UNet (~3M params) |
| `flow_matching.py` | FM loss + Euler sampler |
| `mean_flow.py` | Mean Flow loss (JVP) + one-step sampler |
| `weak_mean_flow.py` | Weak-Form Mean Flow loss (no JVP); sampling reuses `mf_sample` |
| `train.py` | training loop, EMA, wandb, rolling 5 ckpts |
| `evaluate.py` | FID/MSE/PSNR/SSIM/NFE/time eval pipeline |
| `plot.py` | comparison figures |
