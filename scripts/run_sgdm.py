"""
run_sgdm.py -- Exact SGDM augmented-state study on the CIFAR-10 BN ConvNet.

This script validates the SGDM theorem directly by tracking the augmented state
z_t = r_t m_{t+1}. It produces:
- exact SGDM recurrence residuals,
- radial-pump diagnostics based on c_t = <u_t, z_t>,
- a compact exponent scan showing the SGDM law fits a quadratic denominator.

Usage:
    python scripts/run_sgdm.py --out_dir results/sgdm --data_root /path/to/data --device cuda:1
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
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.theory import schedule_constant, schedule_step, schedule_cosine, schedule_controls
from src.models import SmallConvBNNet
from src.diagnostics import compute_sgdm_block_metrics, flatten_filter_blocks

DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(__file__), '..', 'data'),
)
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EXPONENT_GRID = np.linspace(0.5, 3.0, 26, dtype=float)


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_cifar10(data_root, train_limit, test_limit, device):
    import torchvision
    import torchvision.transforms as T
    train_ds = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=T.ToTensor())
    test_ds = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=T.ToTensor())
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    x_train = torch.stack([train_ds[i][0] for i in range(min(train_limit, len(train_ds)))])
    y_train = torch.tensor([train_ds[i][1] for i in range(min(train_limit, len(train_ds)))])
    x_test = torch.stack([test_ds[i][0] for i in range(min(test_limit, len(test_ds)))])
    y_test = torch.tensor([test_ds[i][1] for i in range(min(test_limit, len(test_ds)))])
    x_train = ((x_train - mean) / std).to(device)
    y_train = y_train.to(device)
    x_test = ((x_test - mean) / std).to(device)
    y_test = y_test.to(device)
    return x_train, y_train, x_test, y_test


# -----------------------------------------------------------------------
# SGDM training loop with filterwise diagnostics
# -----------------------------------------------------------------------

def manual_sgdm_step(model, lr, wd, momentum, velocity_dict):
    """Manual SGD with momentum + coupled weight decay.

    p_new = (1 - lr*wd)*p - lr*(momentum*v + grad)
    v_new = momentum*v + grad
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            if name not in velocity_dict:
                velocity_dict[name] = torch.zeros_like(p)
            v = velocity_dict[name]
            v.mul_(momentum).add_(g)
            p.mul_(1.0 - lr * wd).add_(v, alpha=-lr)


def run_sgdm_conv(model, x_train, y_train, x_test, y_test,
                   etas, lams, momentum, batch_size=128, seed=7,
                   log_every=500, eval_every=79, schedule_name=""):
    """Train ConvNet with SGDM and track exact augmented-state diagnostics."""
    T = len(etas) - 1
    gen = torch.Generator(device='cpu').manual_seed(seed)
    n_train = x_train.shape[0]
    steps_per_epoch = int(np.ceil(n_train / batch_size))

    layers = model.conv_layers()
    layer_names = list(layers.keys())
    tracked_param_names = {}
    for layer_name, layer in layers.items():
        tracked_param_names[layer_name] = next(
            name for name, param in model.named_parameters() if param is layer.weight
        )
    exponent_tensor = torch.tensor(EXPONENT_GRID, device=x_train.device, dtype=x_train.dtype)

    history = {
        'train_loss': [], 'B_t': [],
        'eval_t': [], 'eval_epoch': [],
        'eval_train_loss': [], 'eval_train_acc': [],
        'eval_test_loss': [], 'eval_test_acc': [],
        'exponents': EXPONENT_GRID.tolist(),
    }
    exponent_sums = {}
    for name in layer_names:
        history[f'{name}_expansion'] = []
        history[f'{name}_ratio_residual'] = []
        history[f'{name}_threshold_accuracy'] = []
        history[f'{name}_c_median'] = []
        history[f'{name}_abs_c_median'] = []
        history[f'{name}_denom_median'] = []
        history[f'{name}_pump_term_median'] = []
        history[f'{name}_tangent_term_median'] = []
        history[f'{name}_c_positive_fraction'] = []
        history[f'{name}_expansion_if_c_positive'] = []
        history[f'{name}_expansion_if_c_nonpositive'] = []
        history[f'{name}_split_residual'] = []
        history[f'{name}_radius_residual'] = []
        exponent_sums[name] = np.zeros_like(EXPONENT_GRID)

    velocity = {}
    model.train()

    for t in range(T):
        # Sample batch
        n = x_train.shape[0]
        if batch_size >= n:
            xb, yb = x_train, y_train
        else:
            idx = torch.randint(n, (batch_size,), generator=gen, device='cpu')
            xb, yb = x_train[idx], y_train[idx]

        # Snapshot before
        snaps_before = {}
        for name, layer in layers.items():
            snaps_before[name] = flatten_filter_blocks(layer.weight.data.clone())

        # Forward + backward
        model.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        loss.backward()

        grads = {}
        for name, layer in layers.items():
            grads[name] = flatten_filter_blocks(layer.weight.grad.data.clone())

        # SGDM step
        manual_sgdm_step(model, etas[t], lams[t], momentum, velocity)

        # Snapshot after + optimizer state
        snaps_after = {}
        momenta = {}
        for name, layer in layers.items():
            snaps_after[name] = flatten_filter_blocks(layer.weight.data.clone())
            param_name = tracked_param_names[name]
            momenta[name] = flatten_filter_blocks(velocity[param_name].clone())

        # Diagnostics
        B_t = etas[t + 1] / (etas[t] * (1 - etas[t] * lams[t]) * (1 - etas[t + 1] * lams[t + 1]))
        history['train_loss'].append(loss.item())
        history['B_t'].append(B_t)

        for name in layer_names:
            m = compute_sgdm_block_metrics(
                snaps_before[name], momenta[name], snaps_after[name],
                etas[t], lams[t], etas[t + 1], lams[t + 1]
                , exponents=exponent_tensor
            )
            history[f'{name}_expansion'].append(m['expansion_fraction'])
            history[f'{name}_ratio_residual'].append(m['ratio_residual_median'])
            history[f'{name}_threshold_accuracy'].append(m['threshold_accuracy'])
            history[f'{name}_c_median'].append(m['c_median'])
            history[f'{name}_abs_c_median'].append(m['abs_c_median'])
            history[f'{name}_denom_median'].append(m['denom_median'])
            history[f'{name}_pump_term_median'].append(m['pump_term_median'])
            history[f'{name}_tangent_term_median'].append(m['tangent_term_median'])
            history[f'{name}_c_positive_fraction'].append(m['c_positive_fraction'])
            history[f'{name}_expansion_if_c_positive'].append(m['expansion_if_c_positive'])
            history[f'{name}_expansion_if_c_nonpositive'].append(m['expansion_if_c_nonpositive'])
            history[f'{name}_split_residual'].append(m['split_residual_median'])
            history[f'{name}_radius_residual'].append(m['radius_residual_median'])
            exponent_sums[name] += np.array(m['exponent_scan'], dtype=float)

        # Eval
        if eval_every and (t % eval_every == 0 or t == T - 1):
            model.eval()
            with torch.no_grad():
                # Simple eval
                te_logits = model(x_test)
                te_loss = nn.functional.cross_entropy(te_logits, y_test).item()
                te_acc = (te_logits.argmax(1) == y_test).float().mean().item()
                tr_logits = model(x_train[:5000])
                tr_loss = nn.functional.cross_entropy(tr_logits, y_train[:5000]).item()
                tr_acc = (tr_logits.argmax(1) == y_train[:5000]).float().mean().item()
            model.train()
            history['eval_t'].append(t)
            history['eval_epoch'].append(t / steps_per_epoch)
            history['eval_train_loss'].append(tr_loss)
            history['eval_train_acc'].append(tr_acc)
            history['eval_test_loss'].append(te_loss)
            history['eval_test_acc'].append(te_acc)

        if log_every and (t + 1) % log_every == 0:
            te_acc_str = f"{history['eval_test_acc'][-1]:.3f}" if history['eval_test_acc'] else "N/A"
            print(f"  [{schedule_name}] step {t+1}/{T}  loss={loss.item():.4f}  test_acc={te_acc_str}")

    for name in layer_names:
        history[f'{name}_exponent_scan_mean'] = (exponent_sums[name] / max(T, 1)).tolist()

    return history


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=42)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--lr_high', type=float, default=0.5)
    parser.add_argument('--lr_low', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--train_limit', type=int, default=10000)
    parser.add_argument('--test_limit', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--include_sgd_baseline', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = load_cifar10(
        args.data_root, args.train_limit, args.test_limit, args.device
    )

    T = args.total_steps
    schedules = {
        'Constant': schedule_constant(args.lr_high, args.weight_decay, T),
        'Step': schedule_step(args.lr_high, args.lr_low, args.weight_decay, T, T // 2),
        'Cosine': schedule_cosine(args.lr_high, args.lr_low, args.weight_decay, T),
    }

    eval_every = max(1, int(np.ceil(min(args.train_limit, 50000) / args.batch_size)))

    # Also run plain SGD for comparison
    methods = {}
    if args.include_sgd_baseline:
        methods['SGD (no momentum)'] = 0.0
    methods[f'SGDM ($\\mu$=0.9)'] = args.momentum

    all_results = {}
    for method_name, mom in methods.items():
        all_results[method_name] = {}
        for sname, (etas, lams) in schedules.items():
            all_results[method_name][sname] = []
            for seed_idx in range(args.n_seeds):
                actual_seed = args.seed_base + seed_idx
                print(f"\n  {method_name} | {sname} | seed={actual_seed}")
                torch.manual_seed(actual_seed)
                model = SmallConvBNNet(num_classes=10).to(args.device)
                hist = run_sgdm_conv(
                    model, x_train, y_train, x_test, y_test,
                    etas, lams, momentum=mom,
                    batch_size=args.batch_size, seed=actual_seed,
                    log_every=2000, eval_every=eval_every,
                    schedule_name=f'{method_name}_{sname}_s{actual_seed}'
                )
                all_results[method_name][sname].append(hist)

    # ---------------------------------------------------------------
    # Plot: comparison figure
    # ---------------------------------------------------------------
    COLORS_SCHED = {'Constant': '#1f77b4', 'Step': '#ff7f0e', 'Cosine': '#2ca02c'}

    def mean_std_arr(arr_list):
        stacked = np.array(arr_list)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def smooth_arr(arr, window=200):
        if window <= 1 or len(arr) <= window:
            return arr
        kernel = np.ones(window) / window
        padded = np.pad(arr, (window // 2, window // 2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')[:len(arr)]

    win = 200  # Smoothing window for 10k-step series

    # Save full histories for replotting
    hist_path = os.path.join(args.out_dir, 'sgdm_histories.json')
    serializable = {}
    for mn in all_results:
        serializable[mn] = {}
        for sn in all_results[mn]:
            serializable[mn][sn] = []
            for h in all_results[mn][sn]:
                sh = {}
                for k, v in h.items():
                    if isinstance(v, list):
                        sh[k] = [float(x) if isinstance(x, (float, int, np.floating)) else x for x in v]
                    else:
                        sh[k] = v
                serializable[mn][sn].append(sh)
    with open(hist_path, 'w') as f:
        json.dump(serializable, f)
    print(f"Saved histories to {hist_path}")

    # Figure: 2 rows (SGD vs SGDM) x 3 cols (schedules)
    # Each panel: B_t + expansion fraction for conv2 (representative middle layer)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex='col', sharey='row')
    method_names = list(methods.keys())
    for row, method_name in enumerate(method_names):
        for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
            ax = axes[row, col]
            etas, lams = schedules[sname]
            _, B_arr = schedule_controls(etas, lams)
            steps = np.arange(len(B_arr))
            hists = all_results[method_name][sname]

            # B_t (theoretical, same for both)
            ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.8, label=r'$B_t$')
            ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)

            # Expansion fraction (middle conv layer), smoothed
            ax2 = ax.twinx()
            key = 'middle conv2_expansion'
            exp_m, exp_s = mean_std_arr([h[key] for h in hists])
            exp_m_s = smooth_arr(exp_m, win)
            exp_s_s = smooth_arr(exp_s, win)
            ax2.plot(steps, exp_m_s, color=COLORS_SCHED[sname], linewidth=1.2)
            ax2.fill_between(steps, exp_m_s - exp_s_s, exp_m_s + exp_s_s,
                           color=COLORS_SCHED[sname], alpha=0.2)
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
    path = os.path.join(args.out_dir, 'fig_sgdm_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {path}")

    # Figure: accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        ax = axes[col]
        for method_name in method_names:
            hists = all_results[method_name][sname]
            eval_epochs = hists[0]['eval_epoch']
            acc_m, acc_s = mean_std_arr([h['eval_test_acc'] for h in hists])
            ls = '-' if 'SGDM' in method_name else '--'
            ax.plot(eval_epochs, acc_m, ls, linewidth=1.2,
                    label=method_name.split(' ')[0])
            ax.fill_between(eval_epochs, acc_m - acc_s, acc_m + acc_s, alpha=0.15)
        ax.set_title(sname)
        ax.set_xlabel('Epoch')
        if col == 0:
            ax.set_ylabel('Test accuracy')
            ax.legend(fontsize=8)
    fig.suptitle('Test accuracy: SGD vs SGDM (5-seed)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_sgdm_accuracy.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # Figure: exact augmented-state residual comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        ax = axes[col]
        for method_name in method_names:
            hists = all_results[method_name][sname]
            key = 'middle conv2_ratio_residual'
            res_m, res_s = mean_std_arr([h[key] for h in hists])
            steps = np.arange(len(res_m))
            # Smooth for readability
            window = min(100, len(res_m) // 10)
            if window > 1:
                kernel = np.ones(window) / window
                res_m_smooth = np.convolve(res_m, kernel, mode='same')
            else:
                res_m_smooth = res_m
            ls = '-' if 'SGDM' in method_name else '--'
            ax.plot(steps, res_m_smooth, ls, linewidth=1, label=method_name.split(' ')[0])
        ax.set_title(sname)
        ax.set_xlabel('Step')
        if col == 0:
            ax.set_ylabel('Ratio residual (smoothed)')
            ax.legend(fontsize=8)
        ax.set_yscale('log')
    fig.suptitle('Exact augmented-state recurrence residual: SGD vs SGDM', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_sgdm_ratio_residual.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # Figure: radial-pump mechanism under exact SGDM law
    fig, axes = plt.subplots(3, 3, figsize=(12, 7), sharex='col')
    sgdm_label = r'SGDM ($\mu$=0.9)'
    cond_pos_color = '#1f77b4'
    cond_neg_color = '#ff7f0e'
    overall_color = '#222222'
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        hists = all_results[sgdm_label][sname]
        etas, lams = schedules[sname]
        _, B_arr = schedule_controls(etas, lams)
        steps = np.arange(len(B_arr))

        c_m, c_s = mean_std_arr([h['middle conv2_c_median'] for h in hists])
        denom_m, denom_s = mean_std_arr([h['middle conv2_denom_median'] for h in hists])
        exp_all_m, exp_all_s = mean_std_arr([h['middle conv2_expansion'] for h in hists])
        exp_pos_m, exp_pos_s = mean_std_arr([h['middle conv2_expansion_if_c_positive'] for h in hists])
        exp_neg_m, exp_neg_s = mean_std_arr([h['middle conv2_expansion_if_c_nonpositive'] for h in hists])

        c_m_s = smooth_arr(c_m, win)
        c_s_s = smooth_arr(c_s, win)
        denom_m_s = smooth_arr(denom_m, win)
        denom_s_s = smooth_arr(denom_s, win)
        exp_all_m_s = smooth_arr(exp_all_m, win)
        exp_all_s_s = smooth_arr(exp_all_s, win)
        exp_pos_m_s = smooth_arr(exp_pos_m, win)
        exp_pos_s_s = smooth_arr(exp_pos_s, win)
        exp_neg_m_s = smooth_arr(exp_neg_m, win)
        exp_neg_s_s = smooth_arr(exp_neg_s, win)

        ax = axes[0, col]
        ax.plot(steps, c_m_s, color=COLORS_SCHED[sname], linewidth=1.3)
        ax.fill_between(steps, c_m_s - c_s_s, c_m_s + c_s_s,
                        color=COLORS_SCHED[sname], alpha=0.18)
        ax.axhline(0.0, color='black', linestyle=':', linewidth=0.6)
        if col == 0:
            ax.set_ylabel(r'median $c_t$', fontsize=9)
        ax.set_title(sname, fontsize=11)

        ax = axes[1, col]
        ax.plot(steps, denom_m_s, color=COLORS_SCHED[sname], linewidth=1.3, label=r'median $\|u_t-\Phi_t z_t\|^2$')
        ax.fill_between(steps, denom_m_s - denom_s_s, denom_m_s + denom_s_s,
                        color=COLORS_SCHED[sname], alpha=0.18)
        ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.9, label=r'$B_t$')
        if col == 0:
            ax.set_ylabel('denominator / forcing', fontsize=9)
        if col == 2:
            ax.legend(fontsize=7, loc='upper right')

        ax = axes[2, col]
        ax.plot(steps, exp_pos_m_s, color=cond_pos_color, linewidth=1.2, label=r'expand | $c_t>0$')
        ax.fill_between(steps, exp_pos_m_s - exp_pos_s_s, exp_pos_m_s + exp_pos_s_s,
                        color=cond_pos_color, alpha=0.14)
        ax.plot(steps, exp_neg_m_s, color=cond_neg_color, linewidth=1.2, label=r'expand | $c_t\leq 0$')
        ax.fill_between(steps, exp_neg_m_s - exp_neg_s_s, exp_neg_m_s + exp_neg_s_s,
                        color=cond_neg_color, alpha=0.14)
        ax.plot(steps, exp_all_m_s, color=overall_color, linestyle='--', linewidth=0.9, label='overall expansion')
        ax.fill_between(steps, exp_all_m_s - exp_all_s_s, exp_all_m_s + exp_all_s_s,
                        color=overall_color, alpha=0.08)
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel('expansion rate', fontsize=9)
        if col == 2:
            ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('Step')

    fig.suptitle('SGDM radial-pump validation (conv2, 5-seed mean, 200-step smooth)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_sgdm_radial_pump.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # Summary
    summary = {}
    for method_name in method_names:
        summary[method_name] = {}
        for sname in schedules:
            hists = all_results[method_name][sname]
            accs = [max(h['eval_test_acc']) for h in hists]
            ratio_res = [np.median(h['middle conv2_ratio_residual']) for h in hists]
            threshold_acc = [np.mean(h['middle conv2_threshold_accuracy']) for h in hists]
            exponent_curves = np.array([h['middle conv2_exponent_scan_mean'] for h in hists])
            exponent_mean = exponent_curves.mean(axis=0)
            best_idx = int(np.argmin(exponent_mean))
            summary[method_name][sname] = {
                'best_acc_mean': float(np.mean(accs)),
                'best_acc_std': float(np.std(accs)),
                'ratio_residual_median_mean': float(np.mean(ratio_res)),
                'threshold_accuracy_mean': float(np.mean(threshold_acc)),
                'c_median_mean': float(np.mean([np.nanmedian(h['middle conv2_c_median']) for h in hists])),
                'expansion_if_c_positive_mean': float(np.nanmean([np.nanmean(h['middle conv2_expansion_if_c_positive']) for h in hists])),
                'expansion_if_c_nonpositive_mean': float(np.nanmean([np.nanmean(h['middle conv2_expansion_if_c_nonpositive']) for h in hists])),
                'exponent_grid': EXPONENT_GRID.tolist(),
                'exponent_scan_mean': exponent_mean.tolist(),
                'best_exponent': float(EXPONENT_GRID[best_idx]),
            }

    with open(os.path.join(args.out_dir, 'sgdm_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {args.out_dir}/sgdm_summary.json")
    print("\n=== SGDM experiment complete ===")


if __name__ == '__main__':
    main()
