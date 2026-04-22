"""
replot_optimizer_mechanisms.py -- Combine SGDM and Adam-theorem summaries.

Generates a compact denominator-exponent comparison panel from:
- supplementary_material/results/sgdm/sgdm_summary.json
- supplementary_material/results/adam_epsilon/adam_epsilon_summary.json
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


DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DEFAULT_OUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, 'optimizer_mechanisms')
SCHEDULES = ['Constant', 'Step', 'Cosine']
COLORS = {
    'SGD': '#555555',
    'SGDM': '#1f77b4',
    'Adam $\\epsilon=10^{-12}$': '#d62728',
    'Adam $\\epsilon=10^{-8}$': '#ff7f0e',
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def mean_curve_from_sgdm(summary, method_name):
    curves = []
    for schedule in SCHEDULES:
        curves.append(summary[method_name][schedule]['exponent_scan_mean'])
    return np.array(curves).mean(axis=0), np.array(summary[method_name][SCHEDULES[0]]['exponent_grid'])


def find_method_key(summary, prefix):
    for key in summary:
        if key.startswith(prefix):
            return key
    raise KeyError(f'Could not find method key starting with {prefix!r}')


def curve_from_adam(summary, eps_key):
    item = summary['aggregated_exponent_scan'][eps_key]
    return np.array(item['exponent_scan_mean']), np.array(item['exponent_grid'])


def find_eps_key(summary, target=None, smallest=False):
    eps_values = np.array(summary['eps_values'], dtype=float)
    if smallest:
        chosen = eps_values[np.argmin(eps_values)]
    else:
        chosen = eps_values[np.argmin(np.abs(eps_values - target))]
    return f'{chosen:.0e}'


def plot_exponent_panel(results_dir, out_dir):
    sgdm_path = os.path.join(results_dir, 'sgdm', 'sgdm_summary.json')
    adam_path = os.path.join(results_dir, 'adam_epsilon', 'adam_epsilon_summary.json')
    if not os.path.exists(sgdm_path) or not os.path.exists(adam_path):
        raise FileNotFoundError('Missing sgdm or adam_epsilon summary file.')

    sgdm_summary = load_json(sgdm_path)
    adam_summary = load_json(adam_path)

    os.makedirs(out_dir, exist_ok=True)

    sgd_curve, exponent_grid = mean_curve_from_sgdm(sgdm_summary, find_method_key(sgdm_summary, 'SGD'))
    sgdm_curve, _ = mean_curve_from_sgdm(sgdm_summary, find_method_key(sgdm_summary, 'SGDM'))
    adam_small_key = find_eps_key(adam_summary, smallest=True)
    adam_standard_key = find_eps_key(adam_summary, target=1e-8)
    adam_small_curve, _ = curve_from_adam(adam_summary, adam_small_key)
    adam_standard_curve, _ = curve_from_adam(adam_summary, adam_standard_key)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    curves = [
        ('SGD', sgd_curve),
        ('SGDM', sgdm_curve),
        ('Adam $\\epsilon=10^{-12}$', adam_small_curve),
        ('Adam $\\epsilon=10^{-8}$', adam_standard_curve),
    ]
    for label, curve in curves:
        ax.plot(exponent_grid, curve, linewidth=1.8, color=COLORS[label], label=label)
    ax.axvline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.axvline(2.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Denominator exponent')
    ax.set_ylabel('Median recurrence residual')
    ax.set_title('Best-fit denominator exponent')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    eps_values = np.array(adam_summary['eps_values'], dtype=float)
    best_exponents = []
    residuals = []
    for eps in eps_values:
        eps_key = f'{eps:.0e}'
        item = adam_summary['aggregated_exponent_scan'][eps_key]
        best_exponents.append(item['best_exponent'])
        residuals.append(np.min(item['exponent_scan_mean']))
    ax2 = ax.twinx()
    ax.plot(eps_values, best_exponents, color='#d62728', marker='o', linewidth=1.8)
    ax2.plot(eps_values, residuals, color='#1f77b4', marker='s', linewidth=1.5)
    ax.set_xscale('log')
    ax2.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Best-fit exponent', color='#d62728')
    ax2.set_ylabel('Residual at best fit', color='#1f77b4')
    ax.set_ylim(0.8, 1.2)
    ax.set_title(r'Adam continuity toward the $\epsilon=0$ law')
    ax.grid(True, alpha=0.2)

    fig.suptitle('Optimizer-specific denominator structure', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_optimizer_exponent_scan.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    plot_exponent_panel(args.results_dir, args.out_dir)


if __name__ == '__main__':
    main()