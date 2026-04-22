"""
replot_figures.py -- Regenerate paper figures with improved readability.

For 10k-step figures (Conv multiseed, SGDM comparison), applies moving-average
smoothing to make trends visible without the noise from individual step plotting.

Usage:
    python scripts/replot_figures.py --results_dir results --out_dir results
"""

import argparse
import json
import os
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{os.environ.get('USER', 'user')}"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.theory import schedule_constant, schedule_step, schedule_cosine, schedule_controls


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def mean_std(arr_list):
    stacked = np.array(arr_list)
    return stacked.mean(axis=0), stacked.std(axis=0)


def smooth(arr, window=200):
    """Moving average smoothing."""
    if window <= 1 or len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def subsample(x, y_mean, y_std=None, max_points=500):
    """Subsample long arrays to max_points for cleaner plotting."""
    if len(x) <= max_points:
        if y_std is not None:
            return x, y_mean, y_std
        return x, y_mean
    step = max(1, len(x) // max_points)
    idx = np.arange(0, len(x), step)
    if y_std is not None:
        return x[idx], y_mean[idx], y_std[idx]
    return x[idx], y_mean[idx]


COLORS = {'Constant': '#1f77b4', 'Step': '#ff7f0e', 'Cosine': '#2ca02c'}


# -----------------------------------------------------------------------
# Replot fig_conv_multiseed.png (Figure 4)
# -----------------------------------------------------------------------

def replot_conv_multiseed(results_dir, out_dir):
    """Replot ConvNet multiseed figure with smoothed expansion fractions."""
    hist_path = os.path.join(results_dir, 'multiseed', 'conv_histories.json')
    if not os.path.exists(hist_path):
        print(f"  Skipping conv_multiseed: {hist_path} not found")
        # Try to load individual histories from the results
        return False

    with open(hist_path) as f:
        all_results = json.load(f)

    T = 10000
    lr_hi, lr_lo, wd = 0.5, 0.05, 0.05
    schedules = {
        'Constant': schedule_constant(lr_hi, wd, T),
        'Step': schedule_step(lr_hi, lr_lo, wd, T, T // 2),
        'Cosine': schedule_cosine(lr_hi, lr_lo, wd, T),
    }

    layer_names = ['early conv1', 'middle conv2', 'late conv3']
    win = 200  # Smoothing window

    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), sharex='col')
    for j, sname in enumerate(['Constant', 'Step', 'Cosine']):
        etas, lams = schedules[sname]
        _, B_arr = schedule_controls(etas, lams)
        steps = np.arange(len(B_arr))
        hists = all_results[sname]

        for i, lname in enumerate(layer_names):
            ax = axes[i, j]
            # B_t is already clean (deterministic)
            ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.8)
            ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)
            if i == 0:
                ax.set_title(sname, fontsize=11)
            if j == 0:
                ax.set_ylabel(lname, fontsize=9)

            ax2 = ax.twinx()
            key = f'{lname}_expansion'
            exp_arrs = [h[key] for h in hists]
            exp_m, exp_s = mean_std(exp_arrs)

            # Smooth the mean and std
            exp_m_s = smooth(exp_m, win)
            exp_s_s = smooth(exp_s, win)

            # Subsample for clean rendering
            xs, ym, ys = subsample(steps, exp_m_s, exp_s_s, max_points=500)
            ax2.plot(xs, ym, color=COLORS[sname], linewidth=1.2)
            ax2.fill_between(xs, ym - ys, ym + ys,
                             color=COLORS[sname], alpha=0.25)
            ax2.set_ylim(-0.05, 1.05)

            if i == 2:
                ax.set_xlabel('Step')

    fig.suptitle('BN ConvNet on CIFAR-10: filterwise transitions (5-seed mean $\\pm$ std)',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'multiseed', 'fig_conv_multiseed.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")
    return True


# -----------------------------------------------------------------------
# Replot fig_sgdm_comparison.png (Figure 6)
# -----------------------------------------------------------------------

def replot_sgdm_comparison(results_dir, out_dir):
    """Replot SGDM comparison figure with smoothed expansion fractions."""
    hist_path = os.path.join(results_dir, 'sgdm', 'sgdm_histories.json')
    if not os.path.exists(hist_path):
        print(f"  Skipping sgdm_comparison: {hist_path} not found")
        return False

    with open(hist_path) as f:
        all_results = json.load(f)

    T = 10000
    lr_hi, lr_lo, wd = 0.5, 0.05, 0.05
    schedules = {
        'Constant': schedule_constant(lr_hi, wd, T),
        'Step': schedule_step(lr_hi, lr_lo, wd, T, T // 2),
        'Cosine': schedule_cosine(lr_hi, lr_lo, wd, T),
    }

    win = 200
    method_names = list(all_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex='col', sharey='row')
    for row, method_name in enumerate(method_names):
        for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
            ax = axes[row, col]
            etas, lams = schedules[sname]
            _, B_arr = schedule_controls(etas, lams)
            steps = np.arange(len(B_arr))
            hists = all_results[method_name][sname]

            ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.8, label=r'$B_t$')
            ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)

            ax2 = ax.twinx()
            key = 'middle conv2_expansion'
            exp_m, exp_s = mean_std([h[key] for h in hists])
            exp_m_s = smooth(exp_m, win)
            exp_s_s = smooth(exp_s, win)
            xs, ym, ys = subsample(steps, exp_m_s, exp_s_s, max_points=500)
            ax2.plot(xs, ym, color=COLORS[sname], linewidth=1.2)
            ax2.fill_between(xs, ym - ys, ym + ys,
                             color=COLORS[sname], alpha=0.2)
            ax2.set_ylim(-0.05, 1.05)

            if row == 0:
                ax.set_title(sname, fontsize=11)
            if col == 0:
                ax.set_ylabel(method_name + '\n' + r'$B_t$', fontsize=9)
            if col == 2:
                ax2.set_ylabel('Expansion frac.', fontsize=9)
            if row == 1:
                ax.set_xlabel('Step')

    fig.suptitle('SGD vs SGDM: $B_t$-driven regime transitions (conv2, 5-seed)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'sgdm', 'fig_sgdm_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")
    return True


# -----------------------------------------------------------------------
# Replot fig_adam_comparison.png (Figure 7)
# -----------------------------------------------------------------------

def replot_adam_comparison(results_dir, out_dir):
    """Replot Adam comparison figure with smoothed expansion fractions."""
    hist_path = os.path.join(results_dir, 'adam', 'adam_histories.json')
    if not os.path.exists(hist_path):
        print(f"  Skipping adam_comparison: {hist_path} not found")
        return False

    with open(hist_path) as f:
        all_results = json.load(f)

    T = 10000
    lr_hi, lr_lo, wd = 0.5, 0.05, 0.05
    schedules = {
        'Constant': schedule_constant(lr_hi, wd, T),
        'Step': schedule_step(lr_hi, lr_lo, wd, T, T // 2),
        'Cosine': schedule_cosine(lr_hi, lr_lo, wd, T),
    }

    win = 200
    method_labels = list(all_results.keys())

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex='col')
    for row, method_label in enumerate(method_labels):
        for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
            ax = axes[row, col]
            etas, lams = schedules[sname]
            _, B_arr = schedule_controls(etas, lams)
            steps = np.arange(len(B_arr))
            hists = all_results[method_label][sname]

            ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.8)
            ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)

            ax2 = ax.twinx()
            key = 'middle conv2_expansion'
            exp_m, exp_s = mean_std([h[key] for h in hists])
            exp_m_s = smooth(exp_m, win)
            exp_s_s = smooth(exp_s, win)
            xs, ym, ys = subsample(steps, exp_m_s, exp_s_s, max_points=500)
            ax2.plot(xs, ym, color=COLORS.get(sname, '#555555'), linewidth=1.2)
            ax2.fill_between(xs, ym - ys, ym + ys,
                             color=COLORS.get(sname, '#555555'), alpha=0.2)
            ax2.set_ylim(-0.05, 1.05)

            if row == 0:
                ax.set_title(sname, fontsize=11)
            if col == 0:
                ax.set_ylabel(method_label + '\n' + r'$B_t$', fontsize=8)
            if col == 2:
                ax2.set_ylabel('Expansion frac.', fontsize=9)
            if row == 2:
                ax.set_xlabel('Step')

    fig.suptitle(r'$B_t$-driven regime transitions: SGD vs Adam vs AdamW (conv2, 5-seed)',
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'adam', 'fig_adam_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")
    return True


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()

    print("Replotting figures with improved readability...")
    replot_conv_multiseed(args.results_dir, args.out_dir)
    replot_sgdm_comparison(args.results_dir, args.out_dir)
    replot_adam_comparison(args.results_dir, args.out_dir)
    print("Done.")


if __name__ == '__main__':
    main()
