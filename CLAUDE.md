# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Remote python: `/home/ndag/anaconda3/bin/python` (plain `python` not on PATH over non-interactive ssh).
- This folder is a mount of the remote machine `ndagpc` (see `~/.ssh/config`). Training/GPU work runs on that machine (CUDA); run heavy commands there via `ssh ndagpc`, not on the local box. README refers to the remote-side path `/mnt/data/azfarus/projects/testing-the-flows`.
- No git repo, no test suite, no linter. Flat scripts, no package install step. Deps: `torch`, `torchvision`, `wandb`, `torchmetrics`, `pytorch-msssim`, `scipy`, `matplotlib`.
- `.env` holds `WANDB_API_KEY` (and optional `WANDB_PROJECT/ENTITY/MODE`); loaded by `config.load_env`. Never print or commit it. Without a key, wandb falls back to offline mode.

## Purpose

Fair comparison of **Flow Matching** (`flow`, multi-step Euler ODE) vs **Mean Flow** (`meanflow`, one-step, JVP-based; Geng et al. 2025) vs **Weak-Form Mean Flow** (`wfmeanflow`, one-step, JVP-free; spec in `weak-form-meanflow/*.md`) on MNIST, using the same small UNet (~3M params), optimizer, seeds and data order. Original constraints (`cline-prompts/init-prompt.md`): keep code simple, <=10 files, <=500 lines per file (currently 9 .py files).

## Commands

```bash
# smoke test (~1 min)
python train.py --method flow     --max-steps 50 --ckpt-every 25 --log-every 10
python train.py --method meanflow --max-steps 50 --ckpt-every 25 --log-every 10
python train.py --method wfmeanflow --max-steps 50 --ckpt-every 25 --log-every 10
python evaluate.py --num-samples 256 --num-real 1000

# full runs (one tmux window each, see README), then final comparison
python train.py --method flow     2>&1 | tee logs_flow.txt
python train.py --method meanflow 2>&1 | tee logs_meanflow.txt
python train.py --method wfmeanflow 2>&1 | tee logs_wfmeanflow.txt
python evaluate.py 2>&1 | tee logs_eval.txt      # writes results/ (json, grids, plots via plot.py)
```

Useful `train.py` flags: `--epochs --batch-size --lr --max-steps --ckpt-every --log-every --seed --run-name --out-dir --wandb-mode {online,offline,disabled} --no-eval-at-ckpt --eval-num-samples --eval-num-real --eval-steps --eval-seed`. Use `--wandb-mode disabled` for local experiments.

## Architecture

- `config.py`: single `Config` dataclass = source of truth for all hyperparameters shared by all methods (changing a default affects every run; CLI flags in `train.py` override it). Also wandb entity/mode resolution. Note defaults (`epochs=1000`, `lr=2e-5`, `ckpt_every=200`) differ from README's "20 epochs" text; trust the code.
- `unet.py`: `SmallUNet`. `use_r=False` -> `model(x, t)` for flow; `use_r=True` (any method != flow) -> `model(x, t, r)` adds an `r` sinusoidal embedding to the time embedding (mean flow learns `u(x, r, t)`).
- `flow_matching.py`: `fm_loss` / `fm_sample`. Path `x_t=(1-t)x0 + t*eps`, target `eps - x0`; sampling integrates t=1 -> 0 with Euler.
- `mean_flow.py`: `mf_loss` uses `torch.func.jvp` with tangents `(v, 0, 1)` for `du/dt`; target is stop-gradient, adaptive weighting `1/(err²+c)`; `mf_sample` one-step `x0 = z1 - u(z1, 0, 1)` (multi-step supported). `(r, t)` sampled with `r<=t`, fraction `mf_ratio_r_eq_t` has `r==t`.
- `weak_mean_flow.py`: `wf_loss(model, x0, cfg, data_pool)`. Random Fourier test fields (fresh each step), U-statistic of the weak residual, forcing term from an independent aux batch drawn from `data_pool` (whole train set on GPU, `data.get_train_tensor`), FM anchor on the `wf_anchor_ratio` slice with `r=t`. Sampling = `mf_sample`. `raw_mse` = anchor MSE. Sanity check: with an analytic `u*` on Gaussian data the weak term should sit at the Var(b̂) floor (~1e-5) and be clearly larger for `u=0`.
- `train.py`: shared loop for all methods (dispatches on `--method`), EMA (sampling/eval always use EMA weights), rolling window of last 5 checkpoints in `checkpoints/<method>/ckpt_step_*.pt` + `loss_log.jsonl` / `eval_log.jsonl`. At each checkpoint it logs 4 samples to wandb and runs `evaluate.CheckpointEvaluator` (FID/MSE/PSNR/SSIM/NFE/time at `ckpt_eval_steps`, same params for all methods) under `eval/steps_<N>/...`.
- `evaluate.py`: also the library for `CheckpointEvaluator`; standalone `main()` loads the latest EMA checkpoint per method and sweeps NFE (flow 1/4/8/16/32 vs meanflow/wfmeanflow 1) -> `results/`.
- `data.py`: seeded MNIST loaders (`./data`, normalized to [-1,1]); identical batch order across methods relies on the seed in `Config`.
- `plot.py`: loss curves / FID-vs-NFE / metric bars from `results/eval_results.json` and `checkpoints/*/loss_log.jsonl`.

`logs_*.txt`, `wandb/`, `checkpoints/`, `__pycache__/` are run artifacts (large); don't read them wholesale.
