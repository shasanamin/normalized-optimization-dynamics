"""
run_anisotropic.py -- Anisotropic kappa sweep for the normalized-linear model.

Shows that the exact Phi_t recurrence governs contraction/expansion across
varying condition numbers, and how the isotropic 2D picture deforms.

Usage:
    python scripts/run_anisotropic.py --out_dir results/anisotropic
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
from src.theory import (schedule_constant, schedule_step, schedule_cosine,
                         schedule_controls, run_full_simulation,
                         make_isotropic_problem, fixed_point_values,
                         eigenvalue_modulus)


def make_controlled_aniso_problem(d=30, kappa=1.0, seed=42):
    """Create normalized-linear problem with controlled condition number.

    Constructs Sigma with eigenvalues linearly spaced between 1/sqrt(kappa) and sqrt(kappa)
    so that cond(Sigma) = kappa.
    """
    rng = np.random.default_rng(seed)

    # Random orthogonal basis
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))

    # Eigenvalues with desired condition number
    if kappa <= 1.0 + 1e-10:
        eigs = np.ones(d)
    else:
        eigs = np.linspace(1.0 / np.sqrt(kappa), np.sqrt(kappa), d)

    Sigma = Q @ np.diag(eigs) @ Q.T
    n = d + 1  # need n > d
    # Construct A such that A^T A / n = Sigma via Cholesky
    L = np.linalg.cholesky(Sigma)
    raw = rng.standard_normal((n, d))
    # Use QR to get orthonormal rows, then scale
    Q_raw, _ = np.linalg.qr(raw, mode='reduced')
    A = Q_raw * np.sqrt(n) @ L.T  # A^T A / n ≈ Sigma (exact if n=d)
    # Force exact: use SVD-based construction
    A = np.sqrt(n) * Q_raw @ L.T

    # Verify
    Sigma_check = A.T @ A / n

    # Realizable target
    beta = rng.standard_normal(d)
    beta /= np.sqrt(beta @ Sigma @ beta)  # normalize so beta^T Sigma beta = 1
    y = A @ beta

    actual_kappa = eigs[-1] / eigs[0]
    return {'A': A, 'Sigma': Sigma, 'y': y, 'beta': beta, 'kappa': actual_kappa}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--d', type=int, default=30)
    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--eta', type=float, default=0.7)
    parser.add_argument('--lam', type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    kappas = [1.0, 2.0, 5.0, 10.0, 20.0]
    eta, lam = args.eta, args.lam
    T = args.T
    a = 1.0 - eta * lam
    B_const = 1.0 / (a ** 2)

    etas, lams = schedule_constant(eta, lam, T)
    _, B_arr = schedule_controls(etas, lams)

    print(f"Constant schedule: eta={eta}, lam={lam}, a={a:.4f}, B={B_const:.4f}")
    print(f"Running d={args.d}, T={T}\n")

    results = {}
    for kappa in kappas:
        print(f"  kappa = {kappa:.1f}")
        prob = make_controlled_aniso_problem(d=args.d, kappa=kappa, seed=42)

        # Initial w: random, not aligned
        rng = np.random.default_rng(123)
        w0 = rng.standard_normal(args.d)
        w0 /= np.linalg.norm(w0)
        w0 *= 1.0  # unit norm

        sim = run_full_simulation(prob['A'], prob['Sigma'], prob['y'],
                                  etas, lams, w0)

        # Compute Phi ratio residuals
        Phi = sim['Phi']
        g = sim['g']
        ratio_actual = Phi[1:] / Phi[:-1]
        ratio_predicted = B_arr / (1.0 + Phi[:-1]**2 * g[:-1]**2)
        ratio_residual = np.abs(ratio_actual - ratio_predicted)

        # Expansion fraction over time
        expanding = (Phi[1:] > Phi[:-1]).astype(float)

        results[kappa] = {
            'F': sim['F'].tolist(),
            'Phi': sim['Phi'].tolist(),
            'q': sim['q'].tolist(),
            'R': sim['R'].tolist(),
            'ratio_residual': ratio_residual.tolist(),
            'ratio_residual_median': float(np.median(ratio_residual)),
            'ratio_residual_max': float(np.max(ratio_residual)),
            'expansion_fraction': float(expanding.mean()),
            'expansion_second_half': float(expanding[T//2:].mean()),
            'actual_kappa': float(prob['kappa']),
        }

        print(f"    ratio residual: median={np.median(ratio_residual):.2e}, max={np.max(ratio_residual):.2e}")
        print(f"    expansion frac: {expanding.mean():.3f}")

    # ---------------------------------------------------------------
    # Plot 1: Phi_t trajectory across kappas (shows recurrence persists)
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))

    # Top row: Phi_t for each kappa
    for i, kappa in enumerate(kappas[:3]):
        ax = axes[0, i]
        Phi = np.array(results[kappa]['Phi'])
        ax.plot(Phi, linewidth=0.7, color='#1f77b4')
        ax.set_title(f'$\\kappa = {kappa:.0f}$', fontsize=11)
        ax.set_ylabel(r'$\Phi_t$' if i == 0 else '')
        ax.set_xlabel('Step')
        ax.set_yscale('log')

    for i, kappa in enumerate(kappas[3:]):
        ax = axes[1, i]
        Phi = np.array(results[kappa]['Phi'])
        ax.plot(Phi, linewidth=0.7, color='#1f77b4')
        ax.set_title(f'$\\kappa = {kappa:.0f}$', fontsize=11)
        ax.set_ylabel(r'$\Phi_t$' if i == 0 else '')
        ax.set_xlabel('Step')
        ax.set_yscale('log')

    # Bottom right: ratio residual vs kappa
    ax = axes[1, 2]
    med_res = [results[k]['ratio_residual_median'] for k in kappas]
    max_res = [results[k]['ratio_residual_max'] for k in kappas]
    ax.semilogy(kappas, med_res, 'bo-', label='Median')
    ax.semilogy(kappas, max_res, 'r^--', label='Max')
    ax.set_xlabel(r'Condition number $\kappa$')
    ax.set_ylabel('Ratio residual')
    ax.set_title(r'Recurrence exactness vs $\kappa$')
    ax.legend(fontsize=8)

    fig.suptitle(r'Anisotropic normalized-linear model: $\Phi_t$ recurrence across $\kappa$ (constant LR)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_anisotropic_phi.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {path}")

    # ---------------------------------------------------------------
    # Plot 2: Phase portraits (F_t vs Phi_t) showing deformation
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, len(kappas), figsize=(3.2 * len(kappas), 3))
    for i, kappa in enumerate(kappas):
        ax = axes[i]
        Phi = np.array(results[kappa]['Phi'])
        F = np.array(results[kappa]['F'])
        # Color by time
        ax.scatter(Phi[50:], F[50:], c=np.arange(50, len(Phi)), cmap='viridis',
                  s=2, alpha=0.7)
        ax.set_xlabel(r'$\Phi_t$')
        ax.set_ylabel('$F_t$' if i == 0 else '')
        ax.set_title(f'$\\kappa = {kappa:.0f}$', fontsize=10)
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.suptitle('Phase portraits: recurrent structure under anisotropy', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_anisotropic_phase.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # ---------------------------------------------------------------
    # Plot 3: Schedule comparison under anisotropy
    # ---------------------------------------------------------------
    kappa_test = 10.0
    prob = make_controlled_aniso_problem(d=args.d, kappa=kappa_test, seed=42)
    rng = np.random.default_rng(123)
    w0 = rng.standard_normal(args.d)
    w0 /= np.linalg.norm(w0)

    sched_configs = {
        'Constant': schedule_constant(eta, lam, T),
        'Step': schedule_step(eta, 0.02, lam, T, T // 2),
        'Cosine': schedule_cosine(eta, 0.02, lam, T),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    COLORS = {'Constant': '#1f77b4', 'Step': '#ff7f0e', 'Cosine': '#2ca02c'}

    for j, (sname, (etas_s, lams_s)) in enumerate(sched_configs.items()):
        sim = run_full_simulation(prob['A'], prob['Sigma'], prob['y'],
                                  etas_s, lams_s, w0.copy())
        _, B_s = schedule_controls(etas_s, lams_s)
        steps = np.arange(len(B_s))

        # Top: B_t + expansion
        ax = axes[0, j]
        ax.plot(steps, B_s, color='gray', linestyle='--', linewidth=0.8, label=r'$B_t$')
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)
        ax.set_title(f'{sname} ($\\kappa={kappa_test:.0f}$)', fontsize=10)
        if j == 0:
            ax.set_ylabel(r'$B_t$')

        # Compute per-step expansion
        Phi = sim['Phi']
        expanding = (Phi[1:] > Phi[:-1]).astype(float)
        # Smooth
        w = 10
        kernel = np.ones(w) / w
        exp_smooth = np.convolve(expanding, kernel, mode='same')
        ax2 = ax.twinx()
        ax2.plot(steps, exp_smooth, color=COLORS[sname], linewidth=1)
        ax2.set_ylim(-0.05, 1.05)
        if j == 2:
            ax2.set_ylabel('Expansion (smoothed)')

        # Bottom: loss
        ax = axes[1, j]
        F = sim['F']
        ax.semilogy(F, color=COLORS[sname], linewidth=0.7)
        ax.set_xlabel('Step')
        if j == 0:
            ax.set_ylabel('$F_t$')

    fig.suptitle(f'Schedule transitions in anisotropic normalized-linear model ($\\kappa={kappa_test:.0f}$)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_anisotropic_schedules.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # Save summary
    summary = {}
    for k in kappas:
        summary[str(k)] = {
            'ratio_residual_median': results[k]['ratio_residual_median'],
            'ratio_residual_max': results[k]['ratio_residual_max'],
            'expansion_fraction': results[k]['expansion_fraction'],
            'expansion_second_half': results[k]['expansion_second_half'],
        }
    with open(os.path.join(args.out_dir, 'anisotropic_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Anisotropic experiment complete ===")


if __name__ == '__main__':
    main()
