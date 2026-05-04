#!/usr/bin/env python
"""Plot and summarize Transformer-LM Phi-recurrence diagnostics.

Reads integrated histories under ``results/transformer_lm/`` and produces the
paper figure ``fig_transformer_lm.png`` plus a compact JSON summary. This script
does not run training.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{os.environ.get('USER', 'user')}"),
)
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "supplementary_material" / "results"
DEFAULT_FIG_OUT = REPO / "paper" / "figures" / "fig_transformer_lm.png"
DEFAULT_SUMMARY_OUT = (
    REPO / "supplementary_material" / "results" / "transformer_lm" / "summary.json"
)

SOURCES = {
    "small_gpt2 / WikiText": "wikitext_small_gpt2",
    "gpt2 / OpenWebText": "openweb_gpt2",
}

OPT_FILES = {
    "SGD": ("multiseed/conv_histories.json", None),
    "SGDM": ("sgdm/sgdm_histories.json", "SGDM ($\\mu$=0.9)"),
    "Adam": ("adam/adam_histories.json", "Adam (coupled WD)"),
}

SUMMARY_OPTS = {
    "SGD": ("multiseed/conv_histories.json", None),
    "SGDM": ("sgdm/sgdm_histories.json", "SGDM ($\\mu$=0.9)"),
    "Adam": ("adam/adam_histories.json", "Adam (coupled WD)"),
    "AdamW": ("adam/adam_histories.json", "AdamW (decoupled WD)"),
}

SCHEDS = ["Constant", "Step", "Cosine"]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_key(data: dict, key: str | None):
    if key is None:
        return data
    if key in data:
        return data[key]
    # Accept cleaned labels produced by run_transformer_lm.py if users edit keys.
    if key.startswith("SGDM"):
        for candidate in data:
            if candidate.startswith("SGDM"):
                return data[candidate]
    raise KeyError(key)


def load_runs(root: Path, file_path: str, key: str | None):
    return resolve_key(load_json(root / file_path), key)


def block_keys(history: dict, kind: str) -> list[str]:
    return sorted(
        k for k in history
        if k.startswith("block") and k.endswith(f"_{kind}")
    )


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_runs(runs: dict, warmup: int = 1000) -> dict:
    summary = {}
    for sched in SCHEDS:
        seeds = runs[sched]
        ratio_vals = []
        exp_vals = []
        threshold_vals = []
        best_acc = []
        final_acc = []

        for hist in seeds:
            start = min(warmup, len(hist.get("B_t", [])))
            for key in block_keys(hist, "ratio_residual"):
                vals = np.asarray(hist[key][start:], dtype=float)
                if vals.size and np.isfinite(vals).any():
                    ratio_vals.append(float(np.nanmedian(vals)))
            for key in block_keys(hist, "expansion"):
                vals = np.asarray(hist[key][start:], dtype=float)
                if vals.size:
                    exp_vals.append(float(np.nanmean(vals)))
            for key in block_keys(hist, "threshold_acc"):
                vals = np.asarray(hist[key][start:], dtype=float)
                if vals.size:
                    threshold_vals.append(float(np.nanmean(vals)))
            if hist.get("eval_test_acc"):
                best_acc.append(float(np.nanmax(hist["eval_test_acc"])))
                final_acc.append(float(hist["eval_test_acc"][-1]))

        ratio_arr = np.asarray(ratio_vals, dtype=float)
        ratio_finite = ratio_arr[np.isfinite(ratio_arr)]
        first = seeds[0] if seeds else {}
        summary[sched] = {
            "n_seeds": len(seeds),
            "n_steps": len(first.get("B_t", [])),
            "n_tracked_blocks": len(block_keys(first, "ratio_residual")),
            "ratio_residual_median": finite_or_none(np.median(ratio_finite))
            if ratio_finite.size else None,
            "ratio_residual_min": finite_or_none(np.min(ratio_finite))
            if ratio_finite.size else None,
            "ratio_residual_max": finite_or_none(np.max(ratio_finite))
            if ratio_finite.size else None,
            "expansion_fraction_mean": finite_or_none(np.nanmean(exp_vals))
            if exp_vals else None,
            "expansion_fraction_min": finite_or_none(np.nanmin(exp_vals))
            if exp_vals else None,
            "expansion_fraction_max": finite_or_none(np.nanmax(exp_vals))
            if exp_vals else None,
            "threshold_accuracy_mean": finite_or_none(np.nanmean(threshold_vals))
            if threshold_vals else None,
            "best_token_accuracy_mean": finite_or_none(np.nanmean(best_acc))
            if best_acc else None,
            "final_token_accuracy_mean": finite_or_none(np.nanmean(final_acc))
            if final_acc else None,
        }
    return summary


def write_summary(results_dir: Path, summary_path: Path) -> dict:
    out = {}
    for dataset_label, dataset_dir in SOURCES.items():
        root = results_dir / "transformer_lm" / dataset_dir
        out[dataset_dir] = {"label": dataset_label, "optimizers": {}}
        for opt_name, (path, key) in SUMMARY_OPTS.items():
            runs = load_runs(root, path, key)
            out[dataset_dir]["optimizers"][opt_name] = summarize_runs(runs)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def panel_residuals(ax, root: Path, title: str):
    labels, values = [], []
    for opt_name, (path, key) in OPT_FILES.items():
        runs = load_runs(root, path, key)
        for sched in SCHEDS:
            meds = []
            for hist in runs[sched]:
                layer_meds = []
                for k in block_keys(hist, "ratio_residual"):
                    vals = np.asarray(hist[k][1000:], dtype=float)
                    if vals.size:
                        layer_meds.append(np.nanmedian(vals))
                if layer_meds:
                    meds.append(np.nanmedian(layer_meds))
            meds = [m for m in meds if np.isfinite(m)]
            if meds:
                labels.append(f"{opt_name}\n{sched}")
                values.append(meds)

    positions = np.arange(len(labels))
    ax.boxplot(values, positions=positions, widths=0.6, showfliers=False)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(
        1.2e-7,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="float32 floor",
    )
    ax.set_ylabel("median ratio residual")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, which="both", alpha=0.3)


def panel_adam_expansion(ax, root: Path, title: str):
    colors = {"Constant": "C3", "Step": "C1", "Cosine": "C0"}
    runs = load_runs(root, "adam/adam_histories.json", "Adam (coupled WD)")
    win = 200
    for sched in SCHEDS:
        traces = []
        for hist in runs[sched]:
            keys = block_keys(hist, "expansion")
            if not keys:
                continue
            arr = np.mean([hist[k] for k in keys], axis=0)
            kernel = np.ones(win) / win
            traces.append(np.convolve(arr, kernel, mode="valid"))
        if not traces:
            continue
        mean = np.mean(traces, axis=0)
        std = np.std(traces, axis=0)
        x = np.arange(len(mean)) + win // 2
        ax.plot(x, mean, color=colors[sched], label=f"Adam / {sched}")
        ax.fill_between(x, mean - std, mean + std, color=colors[sched], alpha=0.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("expansion fraction")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="center right", fontsize=7)
    ax.grid(True, alpha=0.3)


def make_figure(results_dir: Path, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for col, (name, dataset_dir) in enumerate(SOURCES.items()):
        root = results_dir / "transformer_lm" / dataset_dir
        panel_residuals(axes[0, col], root, f"Median ratio residual ({name})")
        panel_adam_expansion(
            axes[1, col],
            root,
            f"Adam coupled-WD expansion fraction ({name})",
        )

    fig.suptitle(
        "Transformer-LM validation: LayerNorm attention blocks obey the same "
        "$B_t$ recurrence diagnostics",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out_path", type=Path, default=DEFAULT_FIG_OUT)
    parser.add_argument("--summary_path", type=Path, default=DEFAULT_SUMMARY_OUT)
    args = parser.parse_args()

    write_summary(args.results_dir, args.summary_path)
    make_figure(args.results_dir, args.out_path)
    print(f"wrote {args.out_path}")
    print(f"wrote {args.summary_path}")


if __name__ == "__main__":
    main()
