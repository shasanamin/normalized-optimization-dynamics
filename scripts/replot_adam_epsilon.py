"""
replot_adam_epsilon.py -- regenerate plots and summary from saved Adam-epsilon histories.

Supports merging multiple shard history files produced with different seed ranges.
"""

import argparse
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{os.environ.get('USER', 'user')}"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPONENT_GRID = np.linspace(0.5, 3.0, 26, dtype=float)
EPS_COLORS = {
    '1e-12': '#1f77b4',
    '1e-10': '#2ca02c',
    '1e-08': '#ff7f0e',
    '1e-06': '#d62728',
}
SCHEDULES = ['Constant', 'Step', 'Cosine']


def mean_std_arr(arr_list):
    stacked = np.array(arr_list)
    return stacked.mean(axis=0), stacked.std(axis=0)


def smooth_arr(arr, window=200):
    if window <= 1 or len(arr) <= window:
        return np.array(arr)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def load_histories(path):
    with open(path) as f:
        return json.load(f)


def merge_histories(history_dicts):
    merged = {}
    for histories in history_dicts:
        for eps_key, sched_map in histories.items():
            if eps_key not in merged:
                merged[eps_key] = {}
            for schedule, runs in sched_map.items():
                merged[eps_key].setdefault(schedule, [])
                merged[eps_key][schedule].extend(runs)
    return merged


def replot(histories, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eps_keys = sorted(histories.keys(), key=lambda key: float(key))
    eps_values = [float(key) for key in eps_keys]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for col, sname in enumerate(SCHEDULES):
        ax = axes[col]
        residual_means = []
        residual_stds = []
        gap_means = []
        for eps_key in eps_keys:
            hists = histories[eps_key][sname]
            residuals = [np.median(h['middle conv2_eps0_residual']) for h in hists]
            gaps = [np.median(h['middle conv2_scale_break_gap']) for h in hists]
            residual_means.append(np.mean(residuals))
            residual_stds.append(np.std(residuals))
            gap_means.append(np.mean(gaps))
        eps_numeric = np.array(eps_values, dtype=float)
        ax.errorbar(eps_numeric, residual_means, yerr=residual_stds,
                    color='#1f77b4', marker='o', linewidth=1.2, capsize=2,
                    label='epsilon=0 theorem residual')
        ax.plot(eps_numeric, gap_means, color='#d62728', marker='s', linewidth=1.1,
                label='scale-breaking gap')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(sname)
        ax.set_xlabel(r'$\epsilon$')
        if col == 0:
            ax.set_ylabel('Median residual / gap')
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(r'Adam $\epsilon$-continuity relative to the $\epsilon=0$ theorem (conv2)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_adam_epsilon_residual.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for col, sname in enumerate(SCHEDULES):
        ax = axes[col]
        for eps_key in eps_keys:
            hists = histories[eps_key][sname]
            exp_m, exp_s = mean_std_arr([h['middle conv2_expansion'] for h in hists])
            exp_ms = smooth_arr(exp_m, 200)
            exp_ss = smooth_arr(exp_s, 200)
            steps = np.arange(len(exp_ms))
            color = EPS_COLORS.get(eps_key)
            ax.plot(steps, exp_ms, color=color, linewidth=1.2, label=eps_key)
            ax.fill_between(steps, exp_ms - exp_ss, exp_ms + exp_ss, color=color, alpha=0.12)
        ax.set_title(sname)
        ax.set_xlabel('Step')
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel('Expansion fraction')
            ax.legend(fontsize=8, title=r'$\epsilon$')
    fig.suptitle(r'Schedule transitions remain visible across Adam $\epsilon$ values (conv2)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_adam_epsilon_expansion.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

    summary = {
        'eps_values': eps_values,
        'schedules': {},
        'aggregated_exponent_scan': {},
    }
    for sname in SCHEDULES:
        summary['schedules'][sname] = {}
        for eps_key in eps_keys:
            hists = histories[eps_key][sname]
            exponent_curves = np.array([h['middle conv2_exponent_scan_mean'] for h in hists])
            exponent_mean = exponent_curves.mean(axis=0)
            best_idx = int(np.argmin(exponent_mean))
            summary['schedules'][sname][eps_key] = {
                'best_acc_mean': float(np.mean([max(h['eval_test_acc']) for h in hists])),
                'best_acc_std': float(np.std([max(h['eval_test_acc']) for h in hists])),
                'eps0_residual_median_mean': float(np.mean([np.median(h['middle conv2_eps0_residual']) for h in hists])),
                'actual_direction_residual_mean': float(np.mean([np.median(h['middle conv2_actual_direction_residual']) for h in hists])),
                'scale_break_gap_mean': float(np.mean([np.median(h['middle conv2_scale_break_gap']) for h in hists])),
                'threshold_accuracy_mean': float(np.mean([np.mean(h['middle conv2_threshold_accuracy']) for h in hists])),
                'expansion_fraction_mean': float(np.mean([np.mean(h['middle conv2_expansion']) for h in hists])),
                'exponent_grid': EXPONENT_GRID.tolist(),
                'exponent_scan_mean': exponent_mean.tolist(),
                'best_exponent': float(EXPONENT_GRID[best_idx]),
            }

    for eps_key in eps_keys:
        curves = []
        for sname in SCHEDULES:
            curves.extend([h['middle conv2_exponent_scan_mean'] for h in histories[eps_key][sname]])
        exponent_mean = np.array(curves).mean(axis=0)
        best_idx = int(np.argmin(exponent_mean))
        summary['aggregated_exponent_scan'][eps_key] = {
            'exponent_grid': EXPONENT_GRID.tolist(),
            'exponent_scan_mean': exponent_mean.tolist(),
            'best_exponent': float(EXPONENT_GRID[best_idx]),
        }

    summary_path = os.path.join(out_dir, 'adam_epsilon_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {summary_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--histories_path', type=str, nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    histories = merge_histories([load_histories(path) for path in args.histories_path])
    replot(histories, args.out_dir)


if __name__ == '__main__':
    main()