#!/usr/bin/env python
"""Generate PDF versions of the figures referenced by the paper.

This script only uses saved summaries/histories or deterministic theory
simulations already present in the supplement. It does not run training.

Most outputs are true Matplotlib/vector PDFs. Figures whose plotted time-series
data are not available in the release tree are emitted as raster PDF fallbacks
from the existing PNG and listed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{os.environ.get('USER', 'user')}"),
)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "path.simplify": True,
        "path.simplify_threshold": 0.5,
    }
)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

SUPP = Path(__file__).resolve().parents[1]
REPO = SUPP.parent
sys.path.insert(0, str(SUPP))

from scripts import generate_paper_figures as paper_figs  # noqa: E402
from scripts.run_spiral_source import plot_longrun, plot_spiral_source  # noqa: E402
from scripts.replot_transformer_lm import make_figure as make_transformer_figure  # noqa: E402
from scripts.replot_transformer_lm import write_summary as write_transformer_summary  # noqa: E402


SCHEDULES = ["Constant", "Step", "Cosine"]
COLORS = {"Constant": "#1f77b4", "Step": "#ff7f0e", "Cosine": "#2ca02c"}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def smooth(arr, window=200):
    arr = np.asarray(arr, dtype=float)
    if window <= 1 or arr.size <= window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: arr.size]


def mean_std(arr_list):
    stacked = np.asarray(arr_list, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def subsample(x, *ys, max_points=700):
    x = np.asarray(x)
    if x.size <= max_points:
        return (x, *ys)
    idx = np.linspace(0, x.size - 1, max_points).astype(int)
    return (x[idx], *(np.asarray(y)[idx] for y in ys))


def redirect_png_savefigs_to_pdf():
    """Make existing plotting functions write .pdf when they ask for .png."""

    original_plt_savefig = plt.savefig
    original_fig_savefig = Figure.savefig

    def pdf_path(path):
        if isinstance(path, (str, os.PathLike)):
            p = Path(path)
            if p.suffix.lower() == ".png":
                return p.with_suffix(".pdf")
        return path

    def plt_savefig(path, *args, **kwargs):
        return original_plt_savefig(pdf_path(path), *args, **kwargs)

    def fig_savefig(self, path, *args, **kwargs):
        return original_fig_savefig(self, pdf_path(path), *args, **kwargs)

    plt.savefig = plt_savefig
    Figure.savefig = fig_savefig


def gen_schedule_control_compact(results_dir: Path, out_dir: Path) -> None:
    histories = load_json(results_dir / "exact_map" / "histories.json")
    schedule_keys = ["Constant", "Step Decay", "Cosine Decay"]

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 8.2), sharex="col")
    for col, name in enumerate(schedule_keys):
        data = histories[name]
        B = np.asarray(data["B"], dtype=float)
        R = np.asarray(data["R"], dtype=float)
        Phi = np.asarray(data["Phi"], dtype=float)
        x_b = np.arange(B.size)
        x_full = np.arange(Phi.size)
        threshold = np.sqrt(np.maximum(B - 1.0, 0.0))

        ax = axes[0, col]
        ax.plot(x_b, B, color="#1f77b4", linewidth=1.6)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(name, fontsize=16)
        if col == 0:
            ax.set_ylabel(r"Control factor $B_t$", fontsize=13)

        ax = axes[1, col]
        ax.plot(x_full, R, color="#e24a33", linewidth=1.5, label=r"$R_t$")
        ax.plot(x_b, threshold, color="#2ca25f", linestyle="--", linewidth=1.5,
                label=r"$\sqrt{(B_t-1)_+}$")
        if col == 0:
            ax.set_ylabel(r"$R_t$ vs threshold", fontsize=13)
            ax.legend(fontsize=10, loc="upper right")

        ax = axes[2, col]
        ax.plot(x_full, Phi, color="#9467bd", linewidth=1.7)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration $t$", fontsize=12)
        if col == 0:
            ax.set_ylabel(r"Effective stepsize $\Phi_t$", fontsize=13)

    fig.suptitle("Schedule control in the exact isotropic map", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "fig_schedule_control_compact.pdf", bbox_inches="tight")
    plt.close(fig)


def gen_sgdm_radial_pump_compact(results_dir: Path, out_dir: Path) -> None:
    all_results = load_json(results_dir / "sgdm" / "sgdm_histories.json")
    sgdm_key = next((k for k in all_results if k.startswith("SGDM")), list(all_results)[-1])
    layer = "middle conv2"

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.2), sharex="col")
    for col, sched in enumerate(SCHEDULES):
        hists = all_results[sgdm_key][sched]
        color = COLORS[sched]

        denom_m, denom_s = mean_std([h[f"{layer}_denom_median"] for h in hists])
        overall_m, _ = mean_std([h[f"{layer}_expansion"] for h in hists])
        pos_m, pos_s = mean_std([h[f"{layer}_expansion_if_c_positive"] for h in hists])
        neg_m, neg_s = mean_std([h[f"{layer}_expansion_if_c_nonpositive"] for h in hists])
        B_m, _ = mean_std([h["B_t"] for h in hists])

        steps = np.arange(len(denom_m))
        denom_m = smooth(denom_m)
        denom_s = smooth(denom_s)
        B_m = smooth(B_m)
        overall_m = smooth(overall_m)
        pos_m = smooth(pos_m)
        pos_s = smooth(pos_s)
        neg_m = smooth(neg_m)
        neg_s = smooth(neg_s)

        xs, denom_m, denom_s, B_m, overall_m, pos_m, pos_s, neg_m, neg_s = subsample(
            steps, denom_m, denom_s, B_m, overall_m, pos_m, pos_s, neg_m, neg_s
        )

        ax = axes[0, col]
        ax.plot(xs, denom_m, color=color, linewidth=1.4,
                label=r"median $\|u_t-\Phi_t z_t\|^2$")
        ax.fill_between(xs, denom_m - denom_s, denom_m + denom_s, color=color, alpha=0.18)
        ax.plot(xs, B_m, color="gray", linestyle="--", linewidth=1.1, label=r"$B_t$")
        ax.set_title(sched, fontsize=12)
        if col == 0:
            ax.set_ylabel("denominator / forcing")
        if col == 2:
            ax.legend(fontsize=8, loc="upper right")

        ax = axes[1, col]
        ax.plot(xs, pos_m, color="#1f77b4", linewidth=1.2, label=r"expand | $c_t>0$")
        ax.fill_between(xs, pos_m - pos_s, pos_m + pos_s, color="#1f77b4", alpha=0.12)
        ax.plot(xs, neg_m, color="#ff7f0e", linewidth=1.2, label=r"expand | $c_t\leq0$")
        ax.fill_between(xs, neg_m - neg_s, neg_m + neg_s, color="#ff7f0e", alpha=0.12)
        ax.plot(xs, overall_m, color="black", linestyle="--", linewidth=1.0,
                label="overall expansion")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step")
        if col == 0:
            ax.set_ylabel("expansion rate")
        if col == 2:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("SGDM radial-pump validation (conv2, 5-seed mean, 200-step smooth)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "fig_sgdm_radial_pump_compact.pdf", bbox_inches="tight")
    plt.close(fig)


def gen_minibatch(results_dir: Path, out_dir: Path) -> None:
    data = load_json(results_dir / "minibatch" / "minibatch_histories.json")
    batch_labels = ["BS=512", "BS=256", "BS=128", "BS=32"]
    batch_colors = ["#333333", "#1f77b4", "#ff7f0e", "#d62728"]
    sched_names = ["Constant", "Step Decay", "Cosine Decay"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5), sharex=True)
    for col, sched in enumerate(sched_names):
        ax_exp = axes[0, col]
        ax_rr = axes[1, col]

        for label, color in zip(batch_labels, batch_colors):
            hists = data[sched][label]
            exp_m, exp_s = mean_std([h["expansion_fraction"] for h in hists])
            rr_m, _ = mean_std([h["ratio_residual"] for h in hists])
            steps = np.arange(len(exp_m))

            ax_exp.plot(steps, smooth(exp_m, 10), color=color, linewidth=1.2, label=label)
            ax_exp.fill_between(
                steps,
                smooth(exp_m - exp_s, 10),
                smooth(exp_m + exp_s, 10),
                color=color,
                alpha=0.12,
            )
            ax_rr.plot(steps, smooth(rr_m, 10), color=color, linewidth=1.2)

        ax_exp.set_ylim(-0.05, 1.05)
        ax_exp.set_title(sched, fontsize=11, fontweight="bold")
        if col == 0:
            ax_exp.set_ylabel("Expansion fraction", fontsize=9)
            ax_exp.legend(fontsize=7, loc="upper right")

        ax_rr.set_yscale("log")
        ax_rr.set_ylim(1e-8, 2)
        ax_rr.set_xlabel("Step", fontsize=9)
        ax_rr.grid(True, alpha=0.2)
        if col == 0:
            ax_rr.set_ylabel("Ratio residual", fontsize=9)

    fig.suptitle(
        r"Minibatch robustness: $B_t$ regime transitions survive all batch sizes "
        r"(BN-MLP on MNIST, 5-seed mean)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "fig_minibatch.pdf", bbox_inches="tight")
    plt.close(fig)


def gen_adam_epsilon_residual(results_dir: Path, out_dir: Path) -> None:
    summary = load_json(results_dir / "adam_epsilon" / "adam_epsilon_summary.json")
    eps_values = np.asarray(summary["eps_values"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for col, sched in enumerate(SCHEDULES):
        sched_data = summary["schedules"][sched]
        theorem_residual = []
        scale_gap = []
        for eps in eps_values:
            item = sched_data[f"{eps:.0e}"]
            theorem_residual.append(item["eps0_residual_median_mean"])
            scale_gap.append(item["scale_break_gap_mean"])
        theorem_residual = np.asarray(theorem_residual, dtype=float)
        scale_gap = np.asarray(scale_gap, dtype=float)

        ax = axes[col]
        pos = scale_gap > 0
        ax.loglog(
            eps_values[pos],
            scale_gap[pos],
            color="#d62728",
            marker="s",
            linewidth=1.2,
            label="scale-breaking gap",
        )
        ax.loglog(
            eps_values,
            theorem_residual,
            color="#1f77b4",
            marker="o",
            linewidth=1.2,
            label=r"$\epsilon=0$ theorem residual",
        )
        ax.set_title(sched, fontsize=11)
        ax.set_xlabel(r"$\epsilon$")
        ax.grid(True, which="both", alpha=0.25)
        if col == 0:
            ax.set_ylabel("Median residual / gap")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(r"Adam $\epsilon$-continuity relative to the $\epsilon=0$ theorem (conv2)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "fig_adam_epsilon_residual.pdf", bbox_inches="tight")
    plt.close(fig)


def png_to_raster_pdf(src_png: Path, dst_pdf: Path) -> None:
    img = mpimg.imread(src_png)
    height, width = img.shape[:2]
    dpi = 220
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    fig.savefig(dst_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=SUPP / "results")
    parser.add_argument("--out_dir", type=Path, default=REPO / "paper" / "figures")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    redirect_png_savefigs_to_pdf()

    print(f"Generating paper figure PDFs in {args.out_dir}")

    gen_schedule_control_compact(args.results_dir, args.out_dir)
    plot_spiral_source(0.90, str(args.out_dir / "fig_spiral.png"))
    plot_longrun(0.90, str(args.out_dir / "fig_iso_longrun.png"))
    results_dir_s = str(args.results_dir)
    out_dir_s = str(args.out_dir)

    paper_figs.gen_fig_target_b_multiseed(results_dir_s, out_dir_s)
    gen_sgdm_radial_pump_compact(args.results_dir, args.out_dir)
    paper_figs.gen_fig_optimizer_exponent_scan(results_dir_s, out_dir_s)
    gen_minibatch(args.results_dir, args.out_dir)
    paper_figs.gen_fig_extended_criticality(results_dir_s, out_dir_s)
    gen_adam_epsilon_residual(args.results_dir, args.out_dir)
    paper_figs.gen_beyond_sgd_figures(results_dir_s, out_dir_s)
    write_transformer_summary(
        args.results_dir,
        args.results_dir / "transformer_lm" / "summary.json",
    )
    make_transformer_figure(args.results_dir, args.out_dir / "fig_transformer_lm.pdf")

    raster_fallbacks = {
        "fig_neural_transitions.pdf": "missing per-step MLP/Conv histories",
        "fig_anisotropic_phi.pdf": "only anisotropic summary JSON is saved",
    }
    for pdf_name in raster_fallbacks:
        png = args.out_dir / pdf_name.replace(".pdf", ".png")
        png_to_raster_pdf(png, args.out_dir / pdf_name)

    used = [
        "fig_schedule_control_compact.pdf",
        "fig_neural_transitions.pdf",
        "fig_target_b_multiseed.pdf",
        "fig_sgdm_radial_pump_compact.pdf",
        "fig_optimizer_exponent_scan.pdf",
        "fig_spiral.pdf",
        "fig_iso_longrun.pdf",
        "fig_extended_criticality_controls.pdf",
        "fig_minibatch.pdf",
        "fig_anisotropic_phi.pdf",
        "fig_adam_epsilon_residual.pdf",
        "fig_beyond_sgd.pdf",
        "fig_ratio_residual_spectrum.pdf",
        "fig_layerwise.pdf",
        "fig_transformer_lm.pdf",
    ]
    missing = [name for name in used if not (args.out_dir / name).exists()]
    print("\nRaster fallback PDFs:")
    for name, reason in raster_fallbacks.items():
        print(f"  {name}: {reason}")
    if missing:
        raise FileNotFoundError(f"Missing expected PDFs: {missing}")
    print("\nDone. All paper-referenced figure PDFs are present.")


if __name__ == "__main__":
    main()
