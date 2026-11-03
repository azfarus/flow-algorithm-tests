"""Comparison plots for flow / meanflow / wfmeanflow: loss curves, FID vs NFE, metric bars.

Reads training loss logs (checkpoints/<method>/loss_log.jsonl) and the eval
results JSON written by evaluate.py, then writes PNGs to results/plots/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

METHOD_LABELS = {"flow": "Flow Matching", "meanflow": "Mean Flow",
                 "wfmeanflow": "Weak-Form Mean Flow"}
METHOD_COLORS = {"flow": "tab:blue", "meanflow": "tab:orange", "wfmeanflow": "tab:green"}


def plot_loss_curves(ckpt_root, out_path):
    series = {}
    for method, label in METHOD_LABELS.items():
        path = os.path.join(ckpt_root, method, "loss_log.jsonl")
        if not os.path.exists(path):
            continue
        steps, vals = [], []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                steps.append(d["step"])
                vals.append(d.get("raw_mse", d["loss"]))
        series[method] = (steps, vals)
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, (s, v) in series.items():
        ax.plot(s, v, label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.85)
    ax.set_xlabel("training step")
    ax.set_ylabel("velocity MSE (raw)")
    ax.set_yscale("log")
    ax.set_title("Training loss (raw MSE; wfmeanflow: diagonal anchor)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_fid_vs_nfe(results, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, entries in results.items():
        xs = sorted(int(k.split("_")[1]) for k in entries)
        ys = [entries[f"steps_{x}"]["fid"] for x in xs]
        ax.plot(xs, ys, marker="o", label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method))
    ax.set_xlabel("NFE (number of function evaluations)")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("Sample quality vs compute budget")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({x for e in results.values()
                          for x in (int(k.split("_")[1]) for k in e)}))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metric_bars(results, out_path):
    labels, colors, groups = [], [], {"mse": [], "psnr": [], "ssim": []}
    for method, entries in results.items():
        for key in sorted(entries, key=lambda k: int(k.split("_")[1])):
            n = int(key.split("_")[1])
            labels.append(f"{METHOD_LABELS.get(method, method)}\n{n} step" + ("s" if n > 1 else ""))
            colors.append(METHOD_COLORS.get(method))
            for m in groups:
                groups[m].append(entries[key][m])
    if not labels:
        return None
    titles = {"mse": "MSE (lower better)", "psnr": "PSNR dB (higher better)",
              "ssim": "SSIM (higher better)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (m, vals) in zip(axes, groups.items()):
        ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(titles[m])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Flow Matching vs Mean Flow vs Weak-Form Mean Flow: distribution-paired metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_all_plots(results_json, ckpt_root, out_dir):
    """Builds every comparison plot; returns list of PNG paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    p = plot_loss_curves(ckpt_root, os.path.join(out_dir, "loss_curves.png"))
    if p:
        paths.append(p)
    if os.path.exists(results_json):
        with open(results_json) as f:
            results = json.load(f)
        p = plot_fid_vs_nfe(results, os.path.join(out_dir, "fid_vs_nfe.png"))
        paths.append(p)
        p = plot_metric_bars(results, os.path.join(out_dir, "metrics.png"))
        if p:
            paths.append(p)
    return paths


if __name__ == "__main__":
    cfg_root = os.path.dirname(os.path.abspath(__file__))
    for p in make_all_plots(os.path.join(cfg_root, "results", "eval_results.json"),
                            os.path.join(cfg_root, "checkpoints"),
                            os.path.join(cfg_root, "results", "plots")):
        print("wrote", p)
