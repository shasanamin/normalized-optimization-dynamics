"""
run_spiral_source.py -- Reproduce Figure 2: Spiral-source verification.

Launches trajectories from a small neighborhood of the fixed point
and plots the spiraling outward behavior.

Usage:
    python scripts/run_spiral_source.py --out_dir results/spiral
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
    fixed_point_values, fixed_point_jacobian, trace_det_disc,
    isotropic_map_step, run_2d_map_scheduled, schedule_constant,
)


def plot_spiral_source(a: float, out_path: str, n_traj: int = 12,
                       perturbation: float = 0.02, T: int = 80) -> None:
    """Plot local trajectories near the fixed point in centered coordinates."""
    fp = fixed_point_values(a)
    q_star, Phi_star = fp['q_star'], fp['Phi_star']

    fig, ax = plt.subplots(figsize=(6, 6))

    angles = np.linspace(0, 2 * np.pi, n_traj, endpoint=False)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_traj))

    for i, theta in enumerate(angles):
        q0 = q_star + perturbation * np.cos(theta)
        Phi0 = Phi_star + perturbation * 10 * np.sin(theta)

        qs = [q0]
        Phis = [Phi0]
        for _ in range(T):
            q_next, Phi_next = isotropic_map_step(qs[-1], Phis[-1], a)
            qs.append(q_next)
            Phis.append(Phi_next)

        dqs = np.array(qs) - q_star
        dPhis = np.array(Phis) - Phi_star

        ax.plot(dqs, dPhis, '-', color=cmap[i], linewidth=0.8, alpha=0.7)
        ax.plot(dqs[0], dPhis[0], 'o', color=cmap[i], markersize=4)

    ax.plot(0, 0, 'kx', markersize=10, markeredgewidth=2, label='Fixed point')
    ax.set_xlabel(r'$q_t - q_\star$')
    ax.set_ylabel(r'$\Phi_t - \Phi_\star$')
    ax.set_title(f'Spiral source instability ($a = {a}$)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_longrun(a: float, out_path: str, T: int = 500) -> None:
    """Long-run recurrence: loss oscillations and phase portrait."""
    etas, lams = schedule_constant(0.5, 1.0 - a, T)  # eta and lam such that a = 1-eta*lam
    # Actually reconstruct: a = 1 - eta*lam, so lam = (1-a)/eta
    eta = 0.5
    lam_val = (1.0 - a) / eta
    etas, lams = schedule_constant(eta, lam_val, T)

    data = run_2d_map_scheduled(0.0, 100.0, etas, lams)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(data['F'], 'b-', linewidth=0.5)
    ax1.set_xlabel('Iteration $t$')
    ax1.set_ylabel(r'Loss $F_t$')
    ax1.set_title('Recurrent loss oscillations')

    ax2.plot(data['Phi'], data['F'], 'b-', linewidth=0.3, alpha=0.5)
    ax2.set_xlabel(r'$\Phi_t$')
    ax2.set_ylabel(r'$F_t$')
    ax2.set_title(r'Phase portrait $(\Phi_t, F_t)$')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Spiral-source verification')
    parser.add_argument('--out_dir', type=str, default='results/spiral')
    parser.add_argument('--a', type=float, default=0.90)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    a = args.a

    # Verify Jacobian analytics
    tr, det, disc = trace_det_disc(a)
    modulus = np.sqrt(det)
    print(f"Fixed-point analysis for a = {a}:")
    print(f"  Trace = {tr:.4f}")
    print(f"  Determinant = {det:.4f}")
    print(f"  Discriminant = {disc:.4f} (< 0 => complex eigenvalues)")
    print(f"  Eigenvalue modulus = {modulus:.4f} (> 1 => unstable)")

    # Verify switching surface identity
    fp = fixed_point_values(a)
    print(f"  q* = {fp['q_star']:.6f}, Phi* = {fp['Phi_star']:.6f}")
    print(f"  g*Phi* = {fp['gPhi_star']:.6f}, sqrt(B-1) = {fp['switching_surface']:.6f}")
    print(f"  Match error: {abs(fp['gPhi_star'] - fp['switching_surface']):.2e}")

    # Finite-difference Jacobian check
    J_exact = fixed_point_jacobian(a)
    eps_fd = 1e-6
    q_star, Phi_star = fp['q_star'], fp['Phi_star']

    def map_fn(x):
        q_next, Phi_next = isotropic_map_step(x[0], x[1], a)
        return np.array([q_next, Phi_next])

    J_fd = np.zeros((2, 2))
    x0 = np.array([q_star, Phi_star])
    for i in range(2):
        e = np.zeros(2)
        e[i] = eps_fd
        J_fd[:, i] = (map_fn(x0 + e) - map_fn(x0 - e)) / (2 * eps_fd)

    jac_err = np.max(np.abs(J_exact - J_fd))
    print(f"  Jacobian FD error: {jac_err:.2e}")

    # Plot
    plot_spiral_source(a, os.path.join(args.out_dir, 'fig_spiral.png'))
    plot_longrun(a, os.path.join(args.out_dir, 'fig_iso_longrun.png'))

    summary = {
        'a': a,
        'trace': tr,
        'determinant': det,
        'discriminant': disc,
        'eigenvalue_modulus': modulus,
        'q_star': fp['q_star'],
        'Phi_star': fp['Phi_star'],
        'gPhi_star': fp['gPhi_star'],
        'switching_surface': fp['switching_surface'],
        'switching_surface_error': abs(fp['gPhi_star'] - fp['switching_surface']),
        'jacobian_fd_error': jac_err,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main()
