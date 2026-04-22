"""
run_exact_map.py -- Reproduce Figure 1: Schedule control in the exact 2D isotropic map.

Usage:
    python scripts/run_exact_map.py --out_dir results/exact_map
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
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.theory import (
    schedule_constant, schedule_step, schedule_cosine,
    run_2d_map_scheduled, schedule_controls,
)


def plot_schedule_control(results: dict, out_path: str) -> None:
    """4-row, 3-column panel showing B_t, R_t, Phi_t, F_t for each schedule."""
    schedules = list(results.keys())
    fig, axes = plt.subplots(4, len(schedules), figsize=(5 * len(schedules), 12),
                              sharex='col')

    row_labels = [r'$B_t$', r'$R_t$ vs threshold', r'$\Phi_t$', r'$F_t$']

    for j, name in enumerate(schedules):
        data = results[name]
        T = len(data['B'])
        ts = np.arange(T)
        ts_full = np.arange(len(data['Phi']))

        # Row 0: B_t
        ax = axes[0, j]
        ax.plot(ts, data['B'], 'b-', linewidth=1)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylabel(row_labels[0])
        ax.set_title(name, fontsize=12, fontweight='bold')

        # Row 1: R_t and threshold sqrt((B_t-1)_+)
        ax = axes[1, j]
        ax.plot(ts_full, data['R'], 'b-', linewidth=1, label=r'$R_t$')
        thresh = np.sqrt(np.maximum(data['B'] - 1, 0))
        ax.plot(ts, thresh, 'r--', linewidth=1, label=r'$\sqrt{(B_t-1)_+}$')
        ax.set_ylabel(row_labels[1])
        ax.legend(fontsize=8)

        # Row 2: Phi_t
        ax = axes[2, j]
        ax.plot(ts_full, data['Phi'], 'b-', linewidth=1)
        ax.set_ylabel(row_labels[2])
        ax.set_yscale('log')

        # Row 3: F_t
        ax = axes[3, j]
        ax.plot(ts_full, data['F'], 'b-', linewidth=1)
        ax.set_ylabel(row_labels[3])
        ax.set_yscale('log')
        ax.set_xlabel('Iteration $t$')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_log_identity(results: dict, out_path: str) -> None:
    """Verify additive log decomposition identity to machine precision."""
    schedules = list(results.keys())
    fig, axes = plt.subplots(1, len(schedules), figsize=(5 * len(schedules), 3))
    if len(schedules) == 1:
        axes = [axes]

    for j, name in enumerate(schedules):
        data = results[name]
        lhs = data['delta_log_phi_lhs']
        rhs = data['delta_log_phi_rhs']
        error = np.abs(lhs - rhs)

        axes[j].semilogy(error, 'b-', linewidth=1)
        axes[j].set_title(f'{name}\nmax error = {error.max():.2e}', fontsize=10)
        axes[j].set_xlabel('Step $t$')
        axes[j].set_ylabel(r'$|\Delta\log\Phi_t^{LHS} - \Delta\log\Phi_t^{RHS}|$')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Exact 2D map schedule experiments')
    parser.add_argument('--out_dir', type=str, default='results/exact_map')
    parser.add_argument('--T', type=int, default=180)
    parser.add_argument('--q0', type=float, default=0.0)
    parser.add_argument('--Phi0', type=float, default=100.0)
    parser.add_argument('--eta', type=float, default=0.70)
    parser.add_argument('--eta_lo', type=float, default=0.02)
    parser.add_argument('--eta_cos_min', type=float, default=0.01)
    parser.add_argument('--lam', type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Three schedules
    schedules = {
        'Constant': schedule_constant(args.eta, args.lam, args.T),
        'Step Decay': schedule_step(args.eta, args.eta_lo, args.lam, args.T, args.T // 2 - 1),
        'Cosine Decay': schedule_cosine(args.eta, args.eta_cos_min, args.lam, args.T),
    }

    results = {}
    for name, (etas, lams) in schedules.items():
        print(f"Running {name}...")
        data = run_2d_map_scheduled(args.q0, args.Phi0, etas, lams)
        results[name] = data

        # Quick summary
        log_err = np.abs(data['delta_log_phi_lhs'] - data['delta_log_phi_rhs']).max()
        print(f"  B_t range: [{data['B'].min():.4f}, {data['B'].max():.4f}]")
        print(f"  Log identity max error: {log_err:.2e}")

    # Plot
    plot_schedule_control(results, os.path.join(args.out_dir, 'fig_schedule_control.png'))
    plot_log_identity(results, os.path.join(args.out_dir, 'fig_schedule_log_identity.png'))

    # Summary JSON
    summary = {}
    for name, data in results.items():
        summary[name] = {
            'B_min': float(data['B'].min()),
            'B_max': float(data['B'].max()),
            'log_identity_max_error': float(
                np.abs(data['delta_log_phi_lhs'] - data['delta_log_phi_rhs']).max()
            ),
        }

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    serializable_results = {}
    for name, data in results.items():
        serializable_results[name] = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_results[name][key] = value.tolist()
            else:
                serializable_results[name][key] = value

    with open(os.path.join(args.out_dir, 'histories.json'), 'w') as f:
        json.dump(serializable_results, f)

    print("Done.")


if __name__ == '__main__':
    main()
