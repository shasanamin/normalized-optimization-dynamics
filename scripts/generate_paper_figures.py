#!/usr/bin/env python
"""
generate_paper_figures.py -- Regenerate all figures used in the paper.

This script reads pre-computed results from results/ and generates
publication-quality figures. It does NOT re-run experiments.

Figures that require no saved data (pure simulations):
  - fig_schedule_control.png    (exact 2D map)
  - fig_spiral.png              (spiral-source verification)
  - fig_iso_longrun.png         (long-run recurrence)
  - fig_anisotropic_phi.png     (anisotropic robustness, if summary available)

Figures regenerated from saved histories/summaries:
  - fig_mlp_multiseed.png       (from multiseed/ or re-run)
  - fig_conv_multiseed.png      (from multiseed/ or re-run)
  - fig_target_b_multiseed.png  (from multiseed/target_b_multiseed_summary.json)
  - fig_sgdm_radial_pump.png    (from sgdm/sgdm_histories.json)
  - fig_adam_epsilon_residual.png (from adam_epsilon/adam_epsilon_histories.json)
  - fig_optimizer_exponent_scan.png (from sgdm + adam_epsilon summaries)
  - fig_beyond_sgd.png          (from sgdm + adam histories)
  - fig_ratio_residual_spectrum.png (from sgdm + adam histories)
  - fig_layerwise.png           (from sgdm + adam histories)
  - fig_minibatch.png           (from minibatch/minibatch_histories.json)
  - fig_extended_criticality_controls.png (from mnist_criticality/)

Usage:
    # Generate all figures (from results/ data)
    python scripts/generate_paper_figures.py \\
        --results_dir results --out_dir results/paper_figures

    # Copy to paper directory
    python scripts/generate_paper_figures.py \\
        --results_dir results --out_dir ../paper/figures
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
from src.theory import (
    schedule_constant, schedule_step, schedule_cosine,
    run_2d_map_scheduled, schedule_controls,
    fixed_point_values, fixed_point_jacobian, trace_det_disc,
    isotropic_map_step,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
COLORS = {'Constant': '#1f77b4', 'Step': '#ff7f0e', 'Cosine': '#2ca02c'}


def smooth(arr, window=200):
    if window <= 1 or len(arr) <= window:
        return np.array(arr, dtype=float)
    arr = np.array(arr, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def mean_std(arr_list):
    stacked = np.array(arr_list)
    return stacked.mean(0), stacked.std(0)


def subsample(x, y_mean, y_std=None, max_points=500):
    if len(x) <= max_points:
        return (x, y_mean, y_std) if y_std is not None else (x, y_mean)
    step = max(1, len(x) // max_points)
    idx = np.arange(0, len(x), step)
    return (x[idx], y_mean[idx], y_std[idx]) if y_std is not None else (x[idx], y_mean[idx])


def load_json(path):
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Figure 1: Schedule control in the exact isotropic map
# -----------------------------------------------------------------------
def gen_fig_schedule_control(results_dir, out_dir):
    """Regenerate from histories.json or re-simulate."""
    hist_path = os.path.join(results_dir, 'exact_map', 'histories.json')
    if os.path.exists(hist_path):
        results = load_json(hist_path)
    else:
        print("  Re-simulating exact map...")
        eta, lam, T = 0.70, 0.15, 180
        q0, phi0 = 0.0, 100.0
        results = {}
        for name, sched_fn in [
            ('Constant', lambda: schedule_constant(eta, lam, T)),
            ('Step', lambda: schedule_step(eta, 0.02, lam, T, 79)),
            ('Cosine', lambda: schedule_cosine(eta, 0.02, lam, T)),
        ]:
            etas, lams = sched_fn()
            traj = run_2d_map_scheduled(q0, phi0, etas, lams)
            _, B = schedule_controls(etas, lams)
            results[name] = {
                'q': traj['q'].tolist(), 'Phi': traj['Phi'].tolist(),
                'F': traj['F'].tolist(), 'R': traj['R'].tolist(),
                'B': B.tolist(),
            }

    schedules = list(results.keys())
    fig, axes = plt.subplots(4, len(schedules), figsize=(5 * len(schedules), 12), sharex='col')
    row_labels = [r'$B_t$', r'$R_t$ vs threshold', r'$\Phi_t$', r'$F_t$']

    for j, name in enumerate(schedules):
        data = results[name]
        T = len(data['B'])
        ts = np.arange(T)
        ts_full = np.arange(len(data['Phi']))

        ax = axes[0, j]
        ax.plot(ts, data['B'], 'b-', linewidth=1)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_ylabel(row_labels[0])
        ax.set_title(name, fontsize=12, fontweight='bold')

        ax = axes[1, j]
        R = np.array(data['R'])
        B = np.array(data['B'])
        thresh = np.sqrt(np.maximum(B - 1, 0))
        ax.plot(ts, R[:T], 'b-', linewidth=1, label=r'$R_t$')
        ax.plot(ts, thresh, 'r--', linewidth=1, label=r'$\sqrt{(B_t-1)_+}$')
        ax.set_ylabel(row_labels[1])
        ax.legend(fontsize=8)

        ax = axes[2, j]
        ax.plot(ts_full, data['Phi'], 'b-', linewidth=1)
        ax.set_ylabel(row_labels[2])
        ax.set_yscale('log')

        ax = axes[3, j]
        ax.plot(ts_full, data['F'], 'b-', linewidth=1)
        ax.set_ylabel(row_labels[3])
        ax.set_xlabel('Step $t$')

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_schedule_control.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figures: Spiral source and long-run recurrence
# -----------------------------------------------------------------------
def gen_fig_spiral_and_longrun(results_dir, out_dir):
    """Pure simulation -- no saved data needed."""
    a = 0.90
    q_star, phi_star = fixed_point_values(a)

    # Spiral source: trajectories from perturbed initial conditions
    fig_sp, ax_sp = plt.subplots(1, 1, figsize=(5, 5))
    n_traj = 8
    eps = 0.02
    np.random.seed(0)
    for k in range(n_traj):
        angle = 2 * np.pi * k / n_traj
        q0 = q_star + eps * np.cos(angle)
        p0 = phi_star + eps * np.sin(angle)
        qs, ps = [q0], [p0]
        for _ in range(80):
            q_new, p_new = isotropic_map_step(qs[-1], ps[-1], a)
            qs.append(q_new)
            ps.append(p_new)
        ax_sp.plot(qs, ps, '-', linewidth=0.7, alpha=0.8)
    ax_sp.plot(q_star, phi_star, 'k*', markersize=10)
    ax_sp.set_xlabel(r'$q_t$')
    ax_sp.set_ylabel(r'$\Phi_t$')
    ax_sp.set_title('Spiral-source trajectories')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_spiral.png')
    fig_sp.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig_sp)
    print(f"  Saved {path}")

    # Long-run recurrence
    T_long = 500
    eta, lam = 0.70, 0.15
    etas_c, lams_c = schedule_constant(eta, lam, T_long)
    traj = run_2d_map_scheduled(0.0, 100.0, etas_c, lams_c)

    fig_lr, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(T_long + 1), traj['F'], 'b-', linewidth=0.5)
    ax1.set_xlabel('Step $t$')
    ax1.set_ylabel(r'$F_t$')
    ax1.set_title('Loss over 500 steps')

    ax2.plot(traj['Phi'], traj['F'], 'b-', linewidth=0.3, alpha=0.6)
    ax2.set_xlabel(r'$\Phi_t$')
    ax2.set_ylabel(r'$F_t$')
    ax2.set_title(r'$(\Phi_t, F_t)$ phase portrait')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_iso_longrun.png')
    fig_lr.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig_lr)
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: MLP multiseed
# -----------------------------------------------------------------------
def gen_fig_mlp_multiseed(results_dir, out_dir):
    hist_path = os.path.join(results_dir, 'multiseed', 'mlp_histories.json')
    if not os.path.exists(hist_path):
        print("  Skipping fig_mlp_multiseed: mlp_histories.json not found (run run_multiseed.py)")
        return

    all_results = load_json(hist_path)
    T = 120
    lr_hi, lr_lo, wd = 0.5, 0.05, 0.05
    schedules = {
        'Constant': schedule_constant(lr_hi, wd, T),
        'Step': schedule_step(lr_hi, lr_lo, wd, T, T // 2),
        'Cosine': schedule_cosine(lr_hi, lr_lo, wd, T),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex='col')
    for j, sname in enumerate(['Constant', 'Step', 'Cosine']):
        etas, lams = schedules[sname]
        _, B_arr = schedule_controls(etas, lams)
        steps = np.arange(len(B_arr))
        hists = all_results[sname]

        # Top: B_t + expansion
        ax = axes[0, j]
        ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)
        ax.set_title(sname, fontsize=11)
        if j == 0:
            ax.set_ylabel(r'$B_t$')

        ax2 = ax.twinx()
        exp_m, exp_s = mean_std([h['expansion_fraction'] for h in hists])
        ax2.plot(steps, exp_m, color=COLORS[sname], linewidth=1.2)
        ax2.fill_between(steps, exp_m - exp_s, exp_m + exp_s,
                         color=COLORS[sname], alpha=0.25)
        ax2.set_ylim(-0.05, 1.05)
        if j == 2:
            ax2.set_ylabel('Expansion fraction')

        # Bottom: loss + accuracy
        ax = axes[1, j]
        loss_m, loss_s = mean_std([h['train_loss'] for h in hists])
        ax.plot(range(len(loss_m)), loss_m, color='tab:red', linewidth=1)
        ax.fill_between(range(len(loss_m)), loss_m - loss_s, loss_m + loss_s,
                        color='tab:red', alpha=0.15)
        if j == 0:
            ax.set_ylabel('Train loss')
        ax.set_xlabel('Step')

        ax2 = ax.twinx()
        if 'test_acc' in hists[0]:
            acc_m, acc_s = mean_std([h['test_acc'] for h in hists])
            eval_steps = np.linspace(0, T, len(acc_m))
            ax2.plot(eval_steps, acc_m, color='tab:green', linewidth=1)
            ax2.fill_between(eval_steps, acc_m - acc_s, acc_m + acc_s,
                             color='tab:green', alpha=0.15)
        if j == 2:
            ax2.set_ylabel('Test accuracy')

    fig.suptitle('BN MLP on MNIST: schedule transitions (5-seed mean ± std)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_mlp_multiseed.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: Conv multiseed
# -----------------------------------------------------------------------
def gen_fig_conv_multiseed(results_dir, out_dir):
    hist_path = os.path.join(results_dir, 'multiseed', 'conv_histories.json')
    if not os.path.exists(hist_path):
        print("  Skipping fig_conv_multiseed: conv_histories.json not found (run run_multiseed.py)")
        return

    all_results = load_json(hist_path)
    T = 10000
    lr_hi, lr_lo, wd = 0.5, 0.05, 0.05
    schedules = {
        'Constant': schedule_constant(lr_hi, wd, T),
        'Step': schedule_step(lr_hi, lr_lo, wd, T, T // 2),
        'Cosine': schedule_cosine(lr_hi, lr_lo, wd, T),
    }
    layer_names = ['early conv1', 'middle conv2', 'late conv3']
    win = 200

    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), sharex='col')
    for j, sname in enumerate(['Constant', 'Step', 'Cosine']):
        etas, lams = schedules[sname]
        _, B_arr = schedule_controls(etas, lams)
        steps = np.arange(len(B_arr))
        hists = all_results[sname]

        for i, lname in enumerate(layer_names):
            ax = axes[i, j]
            ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.8)
            ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)
            if i == 0:
                ax.set_title(sname, fontsize=11)
            if j == 0:
                ax.set_ylabel(lname, fontsize=9)

            ax2 = ax.twinx()
            key = f'{lname}_expansion'
            exp_m, exp_s = mean_std([h[key] for h in hists])
            exp_m_s, exp_s_s = smooth(exp_m, win), smooth(exp_s, win)
            xs, ym, ys = subsample(steps, exp_m_s, exp_s_s, max_points=500)
            ax2.plot(xs, ym, color=COLORS[sname], linewidth=1.2)
            ax2.fill_between(xs, ym - ys, ym + ys, color=COLORS[sname], alpha=0.25)
            ax2.set_ylim(-0.05, 1.05)
            if i == 2:
                ax.set_xlabel('Step')

    fig.suptitle('BN ConvNet on CIFAR-10: filterwise transitions (5-seed mean ± std)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_conv_multiseed.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: Target-B multiseed
# -----------------------------------------------------------------------
def gen_fig_target_b_multiseed(results_dir, out_dir):
    sum_path = os.path.join(results_dir, 'multiseed', 'target_b_multiseed_summary.json')
    if not os.path.exists(sum_path):
        print("  Skipping fig_target_b_multiseed: summary not found")
        return

    summary = load_json(sum_path)
    B_vals, means, stds = [], [], []
    for key, val in sorted(summary.items()):
        try:
            b = float(key.replace('B=', ''))
        except ValueError:
            continue
        B_vals.append(b)
        means.append(val['final_acc_mean'])
        stds.append(val['final_acc_std'])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(B_vals, means, yerr=stds, fmt='o-', capsize=3, linewidth=1.2,
                markersize=5, color='#1f77b4')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel(r'Target $B$')
    ax.set_ylabel('Final test accuracy (%)')
    ax.set_title(r'Target-constant-$B_t$ sweep on CIFAR-10 (5-seed mean $\pm$ std)')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_target_b_multiseed.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: SGDM radial pump
# -----------------------------------------------------------------------
def gen_fig_sgdm_radial_pump(results_dir, out_dir):
    hist_path = os.path.join(results_dir, 'sgdm', 'sgdm_histories.json')
    if not os.path.exists(hist_path):
        print("  Skipping fig_sgdm_radial_pump: histories not found")
        return

    print("  Loading sgdm histories (large file)...")
    all_results = load_json(hist_path)
    win = 200

    sgdm_key = None
    for k in all_results:
        if 'SGDM' in k or 'sgdm' in k.lower():
            sgdm_key = k
            break
    if sgdm_key is None:
        sgdm_key = list(all_results.keys())[-1]

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex='col')
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        hists = all_results[sgdm_key][sname]
        layer = 'middle conv2'

        # Row 0: median c_t
        ct_key = f'{layer}_ct_median'
        if ct_key in hists[0]:
            ct_m, ct_s = mean_std([h[ct_key] for h in hists])
            ct_m_s, ct_s_s = smooth(ct_m, win), smooth(ct_s, win)
            steps = np.arange(len(ct_m))
            xs, ym, ys = subsample(steps, ct_m_s, ct_s_s)
            axes[0, col].plot(xs, ym, color='#1f77b4', linewidth=1)
            axes[0, col].fill_between(xs, ym - ys, ym + ys, alpha=0.2, color='#1f77b4')
            axes[0, col].axhline(0, color='gray', linestyle=':', linewidth=0.5)
        axes[0, col].set_title(sname, fontsize=11)
        if col == 0:
            axes[0, col].set_ylabel(r'Median $c_t$')

        # Row 1: denominator vs B_t
        den_key = f'{layer}_denom_median'
        bt_key = f'{layer}_Bt'
        if den_key in hists[0] and bt_key in hists[0]:
            den_m, _ = mean_std([h[den_key] for h in hists])
            bt_m, _ = mean_std([h[bt_key] for h in hists])
            den_m_s = smooth(den_m, win)
            bt_m_s = smooth(bt_m, win)
            steps = np.arange(len(den_m))
            xs, d_ym = subsample(steps, den_m_s)
            _, b_ym = subsample(steps, bt_m_s)
            axes[1, col].plot(xs, d_ym, color='#1f77b4', linewidth=1, label='Denominator')
            axes[1, col].plot(xs, b_ym, color='#ff7f0e', linewidth=1, linestyle='--', label=r'$B_t$')
            axes[1, col].legend(fontsize=7)
        if col == 0:
            axes[1, col].set_ylabel('Denom / $B_t$')

        # Row 2: conditional expansion
        for cond, color, label in [
            (f'{layer}_exp_ct_pos', '#d62728', r'$c_t > 0$'),
            (f'{layer}_exp_ct_neg', '#1f77b4', r'$c_t \leq 0$'),
        ]:
            if cond in hists[0]:
                cond_m, cond_s = mean_std([h[cond] for h in hists])
                cond_m_s = smooth(cond_m, win)
                steps = np.arange(len(cond_m))
                xs, ym = subsample(steps, cond_m_s)
                axes[2, col].plot(xs, ym, color=color, linewidth=1, label=label)
        axes[2, col].legend(fontsize=7)
        axes[2, col].set_xlabel('Step')
        if col == 0:
            axes[2, col].set_ylabel('Expansion freq.')

    fig.suptitle('SGDM radial-pump validation (conv2, 5-seed)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_sgdm_radial_pump.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: Adam epsilon residual
# -----------------------------------------------------------------------
def gen_fig_adam_epsilon_residual(results_dir, out_dir):
    sum_path = os.path.join(results_dir, 'adam_epsilon', 'adam_epsilon_summary.json')
    if not os.path.exists(sum_path):
        print("  Skipping fig_adam_epsilon_residual: summary not found")
        return

    summary = load_json(sum_path)
    eps_vals = sorted(summary.keys(), key=lambda x: float(x))
    EPS_COLORS = {'1e-12': '#1f77b4', '1e-10': '#2ca02c', '1e-08': '#ff7f0e', '1e-06': '#d62728'}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        ax = axes[col]
        for eps_str in eps_vals:
            if sname not in summary[eps_str]:
                continue
            data = summary[eps_str][sname]
            color = EPS_COLORS.get(eps_str, '#888888')
            label = rf'$\epsilon=10^{{{int(np.log10(float(eps_str)))}}}$'
            if 'residual_median' in data:
                ax.axhline(data['residual_median'], color=color, linewidth=1.5, label=label)
        ax.set_yscale('log')
        ax.set_title(sname)
        ax.legend(fontsize=7)
        ax.set_xlabel(r'$\epsilon$')
    axes[0].set_ylabel('Median residual')
    fig.suptitle(r'Adam augmented-state residual vs $\epsilon$', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_adam_epsilon_residual.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: Optimizer exponent scan
# -----------------------------------------------------------------------
def gen_fig_optimizer_exponent_scan(results_dir, out_dir):
    sgdm_sum = os.path.join(results_dir, 'sgdm', 'sgdm_summary.json')
    adam_sum = os.path.join(results_dir, 'adam_epsilon', 'adam_epsilon_summary.json')
    if not os.path.exists(sgdm_sum) or not os.path.exists(adam_sum):
        print("  Skipping fig_optimizer_exponent_scan: summaries not found")
        return

    # Delegate to existing script logic
    from scripts.replot_optimizer_mechanisms import main as replot_main
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ['replot_optimizer_mechanisms.py',
                 '--results_dir', results_dir,
                 '--out_dir', out_dir]
    try:
        replot_main()
    finally:
        _sys.argv = old_argv


# -----------------------------------------------------------------------
# Figures: Beyond SGD (combined optimizer comparison)
# -----------------------------------------------------------------------
def gen_beyond_sgd_figures(results_dir, out_dir):
    sgdm_path = os.path.join(results_dir, 'sgdm', 'sgdm_histories.json')
    adam_path = os.path.join(results_dir, 'adam', 'adam_histories.json')
    if not os.path.exists(sgdm_path) or not os.path.exists(adam_path):
        print("  Skipping beyond_sgd figures: sgdm/adam histories not found")
        return

    from scripts.replot_beyond_sgd import main as replot_main
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ['replot_beyond_sgd.py',
                 '--results_dir', results_dir,
                 '--out_dir', out_dir]
    try:
        replot_main()
    finally:
        _sys.argv = old_argv


# -----------------------------------------------------------------------
# Figure: Minibatch robustness
# -----------------------------------------------------------------------
def gen_fig_minibatch(results_dir, out_dir):
    hist_path = os.path.join(results_dir, 'minibatch', 'minibatch_histories.json')
    if not os.path.exists(hist_path):
        print("  Skipping fig_minibatch: histories not found")
        return

    data = load_json(hist_path)
    batch_sizes = sorted(data.keys(), key=lambda x: int(x))
    schedules = ['Constant', 'Step', 'Cosine']

    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex='col')
    for col, sname in enumerate(schedules):
        for bs in batch_sizes:
            if sname not in data[bs]:
                continue
            hists = data[bs][sname]
            color_idx = batch_sizes.index(bs) / max(1, len(batch_sizes) - 1)
            color = plt.cm.viridis(color_idx)

            # Top: expansion fraction
            exp_m, exp_s = mean_std([h['expansion_fraction'] for h in hists])
            steps = np.arange(len(exp_m))
            axes[0, col].plot(steps, exp_m, color=color, linewidth=0.8, label=f'BS={bs}')

            # Bottom: ratio residual
            rr_key = 'ratio_residual'
            if rr_key in hists[0]:
                rr_m, _ = mean_std([h[rr_key] for h in hists])
                axes[1, col].plot(steps, rr_m, color=color, linewidth=0.8)

        axes[0, col].set_title(sname)
        axes[0, col].set_ylim(-0.05, 1.05)
        axes[1, col].set_yscale('log')
        axes[1, col].set_xlabel('Step')

    axes[0, 0].set_ylabel('Expansion frac.')
    axes[1, 0].set_ylabel('Ratio residual')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Minibatch robustness (BN MLP, 5-seed mean)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_minibatch.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: Extended criticality controls
# -----------------------------------------------------------------------
def gen_fig_extended_criticality(results_dir, out_dir):
    hist_path = os.path.join(results_dir, 'mnist_criticality',
                             'mnist_criticality_multiseed_histories.json')
    if not os.path.exists(hist_path):
        print("  Skipping fig_extended_criticality_controls: histories not found")
        return

    data = load_json(hist_path)
    labels = list(data.keys())
    n_rows = 5  # LR, B_t, expansion, loss, accuracy
    fig, axes = plt.subplots(n_rows, len(labels), figsize=(3 * len(labels), 2.5 * n_rows),
                             sharex='col')
    row_labels = ['Learning rate', r'$B_t$', 'Expansion frac.', 'Train loss', 'Test accuracy']

    for j, label in enumerate(labels):
        hists = data[label]
        for i, (key, ylabel) in enumerate(zip(
            ['learning_rates', 'B_t', 'expansion_fraction', 'train_loss', 'test_acc'],
            row_labels
        )):
            ax = axes[i, j] if len(labels) > 1 else axes[i]
            if key not in hists[0]:
                continue
            vals = [h[key] for h in hists]
            m, s = mean_std(vals)
            steps = np.arange(len(m))
            ax.plot(steps, m, linewidth=1)
            ax.fill_between(steps, m - s, m + s, alpha=0.2)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == 0:
                ax.set_title(label, fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel('Step')

    fig.suptitle('Near-critical controls on MNIST (5-seed)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_extended_criticality_controls.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Figure: Anisotropic robustness
# -----------------------------------------------------------------------
def gen_fig_anisotropic(results_dir, out_dir):
    """Re-simulate anisotropic sweep (fast, no saved data needed)."""
    from src.theory import make_anisotropic_problem, run_full_simulation, schedule_constant

    d, n = 30, 60
    eta, lam, T = 0.7, 0.15, 300
    kappas = [1, 2, 5, 10, 20]

    fig, axes = plt.subplots(1, len(kappas), figsize=(3 * len(kappas), 3), sharey=True)
    for i, kappa in enumerate(kappas):
        ax = axes[i]
        try:
            A, Sigma, y = make_anisotropic_problem(n=n, d=d, kappa=kappa, seed=42)
            etas_c, lams_c = schedule_constant(eta, lam, T)
            traj = run_full_simulation(A, Sigma, y, etas_c, lams_c)
            ax.plot(range(len(traj['Phi'])), traj['Phi'], 'b-', linewidth=0.5)
            ax.set_title(rf'$\kappa={kappa}$')
            ax.set_xlabel('Step')
            if i == 0:
                ax.set_ylabel(r'$\Phi_t$')
        except Exception as e:
            ax.text(0.5, 0.5, f'κ={kappa}\n(error)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_anisotropic_phi.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate all paper figures from results data.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Path to results/ directory')
    parser.add_argument('--out_dir', type=str, default='results/paper_figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Generating paper figures from {args.results_dir} → {args.out_dir}\n")

    # Pure simulation figures (no data needed)
    print("--- Pure simulation figures ---")
    gen_fig_schedule_control(args.results_dir, args.out_dir)
    gen_fig_spiral_and_longrun(args.results_dir, args.out_dir)

    # Figures from saved results
    print("\n--- Figures from saved results ---")
    gen_fig_mlp_multiseed(args.results_dir, args.out_dir)
    gen_fig_conv_multiseed(args.results_dir, args.out_dir)
    gen_fig_target_b_multiseed(args.results_dir, args.out_dir)
    gen_fig_minibatch(args.results_dir, args.out_dir)
    gen_fig_extended_criticality(args.results_dir, args.out_dir)

    # Figures from large history files
    print("\n--- Figures from optimizer histories (may be slow to load) ---")
    gen_fig_sgdm_radial_pump(args.results_dir, args.out_dir)
    gen_fig_adam_epsilon_residual(args.results_dir, args.out_dir)

    # Delegated to existing scripts
    print("\n--- Derived figures (delegated to replot scripts) ---")
    try:
        gen_fig_optimizer_exponent_scan(args.results_dir, args.out_dir)
    except Exception as e:
        print(f"  Error generating optimizer_exponent_scan: {e}")
    try:
        gen_beyond_sgd_figures(args.results_dir, args.out_dir)
    except Exception as e:
        print(f"  Error generating beyond_sgd figures: {e}")

    # Anisotropic
    print("\n--- Anisotropic figure ---")
    try:
        gen_fig_anisotropic(args.results_dir, args.out_dir)
    except Exception as e:
        print(f"  Error: {e}. Using existing figure if available.")

    print(f"\nDone. Figures saved to {args.out_dir}/")
    print("\nTo copy to paper directory:")
    print(f"  cp {args.out_dir}/fig_*.png ../paper/figures/")


if __name__ == '__main__':
    main()
