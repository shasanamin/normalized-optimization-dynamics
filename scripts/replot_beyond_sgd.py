"""
replot_beyond_sgd.py -- Regenerate improved "Beyond SGD" figures from saved histories.

Key fixes:
1. Uses stored B_t per seed (correct for AdamW decoupled WD), not recomputed
2. Single y-axis expansion fraction (no confusing dual-axis)
3. Dedicated ratio residual spectrum figure (degradation spectrum)
4. Layerwise comparison across early/middle/late blocks
5. Accuracy comparison across all optimizers

Generates:
- fig_beyond_sgd.png: 4-row x 3-col expansion fraction grid (main paper)
- fig_ratio_residual_spectrum.png: Ratio residual degradation spectrum
- fig_beyond_sgd_accuracy.png: Test accuracy comparison
- fig_layerwise.png: Layerwise expansion fraction comparison

Usage:
    python scripts/replot_beyond_sgd.py --results_dir results --out_dir results/beyond_sgd
"""
import argparse
import json
import os
import sys
import tempfile
import numpy as np
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{os.environ.get('USER', 'user')}"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def smooth(arr, window=200):
    if window <= 1 or len(arr) <= window:
        return np.array(arr)
    arr = np.array(arr, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]

def mean_std(arr_list):
    stacked = np.array(arr_list)
    return stacked.mean(0), stacked.std(0)

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DEFAULT_OUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, 'beyond_sgd')
WIN = 200

METHOD_COLORS = {
    'SGD': '#555555',
    'SGDM': '#1f77b4',
    'Adam': '#d62728',
    'AdamW': '#9467bd',
}
METHOD_LS = {
    'SGD': '--',
    'SGDM': '-',
    'Adam': '-',
    'AdamW': '-',
}
SCHED_NAMES = ['Constant', 'Step', 'Cosine']

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_all_methods(results_dir):
    """Load data and return ordered list of (short_label, data, key_in_data)."""
    sgdm_path = os.path.join(results_dir, 'sgdm', 'sgdm_histories.json')
    adam_path = os.path.join(results_dir, 'adam', 'adam_histories.json')

    if not os.path.exists(sgdm_path) or not os.path.exists(adam_path):
        print(f"Missing data: {sgdm_path} or {adam_path}")
        return None, None, None

    sgdm_data = load_json(sgdm_path)
    adam_data = load_json(adam_path)

    methods = [
        ('SGD',   sgdm_data, 'SGD (no momentum)'),
        ('SGDM',  sgdm_data, r'SGDM ($\mu$=0.9)'),
        ('Adam',  adam_data,  'Adam (coupled WD)'),
        ('AdamW', adam_data,  'AdamW (decoupled WD)'),
    ]
    return methods, sgdm_data, adam_data


# -----------------------------------------------------------------------
# Figure 1: "Beyond SGD" -- Expansion fraction grid
#   4 rows (SGD/SGDM/Adam/AdamW) x 3 cols (Constant/Step/Cosine)
#   Each panel: expansion fraction (solid color) + B_t (gray dashed)
#   Single shared y-axis for expansion fraction, B_t range clipped
# -----------------------------------------------------------------------
def plot_beyond_sgd(results_dir, out_dir):
    methods, _, _ = get_all_methods(results_dir)
    if methods is None:
        return

    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 3, figsize=(12, 8), sharex='col')

    for row, (short_label, data, key) in enumerate(methods):
        for col, sname in enumerate(SCHED_NAMES):
            ax = axes[row, col]
            hists = data[key][sname]

            # Use stored B_t from seed 0 (correct for each optimizer)
            B_t = np.array(hists[0]['B_t'])
            steps = np.arange(len(B_t))

            exp_key = 'middle conv2_expansion'
            exp_m, exp_s = mean_std([h[exp_key] for h in hists])
            exp_ms = smooth(exp_m, WIN)
            exp_ss = smooth(exp_s, WIN)

            # Expansion fraction
            color = METHOD_COLORS[short_label]
            ax.plot(steps, exp_ms, color=color, linewidth=1.2, zorder=2)
            ax.fill_between(steps, exp_ms - exp_ss, exp_ms + exp_ss,
                            color=color, alpha=0.15)
            ax.set_ylim(-0.05, 1.05)

            # B_t on twin axis (gray, clipped to reasonable range)
            ax2 = ax.twinx()
            ax2.plot(steps, B_t, color='gray', linestyle='--', linewidth=0.6,
                     alpha=0.5, zorder=1)
            ax2.axhline(1.0, color='black', linestyle=':', linewidth=0.4, alpha=0.4)
            # Clip B_t axis to show structure without dominating
            bt_max = min(np.max(B_t) * 1.1, 3.0)
            bt_min = max(np.min(B_t) * 0.9, 0.0)
            ax2.set_ylim(bt_min, bt_max)
            ax2.tick_params(axis='y', labelsize=6, colors='gray')

            if row == 0:
                ax.set_title(sname, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(short_label + '\nExp. frac.', fontsize=8)
            if col == 2:
                ax2.set_ylabel('$B_t$', fontsize=7, color='gray')
            else:
                ax2.set_yticklabels([])
            if row == n_methods - 1:
                ax.set_xlabel('Step', fontsize=9)
            ax.tick_params(axis='both', labelsize=6)

    fig.suptitle(
        'Beyond SGD: expansion fraction (conv2, 5-seed mean, 200-step smooth)\n'
        'Colored = expansion fraction, gray dashed = $B_t$',
        fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(out_dir, 'fig_beyond_sgd.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# Figure 2: Ratio residual spectrum (degradation spectrum)
#   1 row x 3 cols: one per schedule, all 4 methods overlaid
# -----------------------------------------------------------------------
def plot_ratio_residual_spectrum(results_dir, out_dir):
    methods, _, _ = get_all_methods(results_dir)
    if methods is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for col, sname in enumerate(SCHED_NAMES):
        ax = axes[col]
        for short_label, data, key in methods:
            hists = data[key][sname]
            res_key = 'middle conv2_ratio_residual'
            res_m, res_s = mean_std([h[res_key] for h in hists])
            res_ms = smooth(res_m, 100)
            steps = np.arange(len(res_ms))
            ax.plot(steps, res_ms,
                    METHOD_LS[short_label], color=METHOD_COLORS[short_label],
                    linewidth=1.2, label=short_label)

        ax.set_yscale('log')
        ax.set_title(sname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Step')
        if col == 0:
            ax.set_ylabel('Ratio residual')
            ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(1e-7, 2)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        r'Degradation spectrum: ratio residual $|\Phi_{t+1}/\Phi_t - B_t/(1+R_t^2)|$',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(out_dir, 'fig_ratio_residual_spectrum.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# Figure 3: Accuracy comparison (all optimizers combined)
# -----------------------------------------------------------------------
def plot_accuracy_combined(results_dir, out_dir):
    methods, _, _ = get_all_methods(results_dir)
    if methods is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for col, sname in enumerate(SCHED_NAMES):
        ax = axes[col]
        for short_label, data, key in methods:
            hists = data[key][sname]
            eval_epochs = hists[0]['eval_epoch']
            acc_m, acc_s = mean_std([h['eval_test_acc'] for h in hists])
            ax.plot(eval_epochs, acc_m,
                    METHOD_LS[short_label], color=METHOD_COLORS[short_label],
                    linewidth=1.2, label=short_label)
            ax.fill_between(eval_epochs, acc_m - acc_s, acc_m + acc_s,
                            color=METHOD_COLORS[short_label], alpha=0.12)

        ax.set_title(sname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        if col == 0:
            ax.set_ylabel('Test accuracy')
            ax.legend(fontsize=8)

    fig.suptitle('Test accuracy: all optimizers (5-seed mean $\\pm$ std)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(out_dir, 'fig_beyond_sgd_accuracy.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# Figure 4: Layerwise comparison
#   3 rows (early conv1 / middle conv2 / late conv3) x 3 cols (schedules)
#   All 4 optimizers overlaid in each panel
# -----------------------------------------------------------------------
def plot_layerwise(results_dir, out_dir):
    methods, _, _ = get_all_methods(results_dir)
    if methods is None:
        return

    layers = [
        ('Early (conv1)', 'early conv1_expansion'),
        ('Middle (conv2)', 'middle conv2_expansion'),
        ('Late (conv3)', 'late conv3_expansion'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 7), sharex='col', sharey=True)

    for row, (layer_label, exp_key) in enumerate(layers):
        for col, sname in enumerate(SCHED_NAMES):
            ax = axes[row, col]
            for short_label, data, key in methods:
                hists = data[key][sname]
                exp_m, _ = mean_std([h[exp_key] for h in hists])
                exp_ms = smooth(exp_m, WIN)
                steps = np.arange(len(exp_ms))
                ax.plot(steps, exp_ms,
                        METHOD_LS[short_label], color=METHOD_COLORS[short_label],
                        linewidth=1.0, label=short_label)

            ax.set_ylim(-0.05, 1.05)
            if row == 0:
                ax.set_title(sname, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(layer_label, fontsize=9)
            if row == 2:
                ax.set_xlabel('Step', fontsize=9)
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Layerwise expansion fraction: all optimizers (5-seed mean, 200-step smooth)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(out_dir, 'fig_layerwise.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# Figure 5: Layerwise ratio residual
#   3 rows (early/middle/late) x 3 cols (schedules), all optimizers overlaid
# -----------------------------------------------------------------------
def plot_layerwise_ratio_residual(results_dir, out_dir):
    methods, _, _ = get_all_methods(results_dir)
    if methods is None:
        return

    layers = [
        ('Early (conv1)', 'early conv1_ratio_residual'),
        ('Middle (conv2)', 'middle conv2_ratio_residual'),
        ('Late (conv3)', 'late conv3_ratio_residual'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 7), sharex='col', sharey=True)

    for row, (layer_label, res_key) in enumerate(layers):
        for col, sname in enumerate(SCHED_NAMES):
            ax = axes[row, col]
            for short_label, data, key in methods:
                hists = data[key][sname]
                res_m, _ = mean_std([h[res_key] for h in hists])
                res_ms = smooth(res_m, 100)
                steps = np.arange(len(res_ms))
                ax.plot(steps, res_ms,
                        METHOD_LS[short_label], color=METHOD_COLORS[short_label],
                        linewidth=1.0, label=short_label)

            ax.set_yscale('log')
            ax.set_ylim(1e-7, 2)
            ax.grid(True, alpha=0.15)
            if row == 0:
                ax.set_title(sname, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(layer_label, fontsize=9)
            if row == 2:
                ax.set_xlabel('Step', fontsize=9)
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(
        r'Layerwise ratio residual: $|\Phi_{t+1}/\Phi_t - B_t/(1+R_t^2)|$ (5-seed mean, 100-step smooth)',
        fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(out_dir, 'fig_layerwise_ratio_residual.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate beyond-SGD figures from saved SGDM/Adam histories.'
    )
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_beyond_sgd(args.results_dir, args.out_dir)
    plot_ratio_residual_spectrum(args.results_dir, args.out_dir)
    plot_accuracy_combined(args.results_dir, args.out_dir)
    plot_layerwise(args.results_dir, args.out_dir)
    plot_layerwise_ratio_residual(args.results_dir, args.out_dir)
    print("\n=== All figures regenerated ===")


if __name__ == '__main__':
    main()
