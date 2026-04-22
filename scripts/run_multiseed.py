"""
run_multiseed.py -- Multi-seed experiments for BN MLP, CIFAR ConvNet, and target-B.

Runs each experiment with 5 seeds and produces plots with mean +/- std bands.

Usage:
    python scripts/run_multiseed.py --out_dir results/multiseed --data_root /path/to/data --device cuda:1
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.theory import (schedule_constant, schedule_step, schedule_cosine,
                         schedule_target_constant_b, schedule_controls)
from src.models import SimpleNormMLP, SmallConvBNNet
from src.training import run_mlp_schedule, run_conv_schedule

DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(__file__), '..', 'data'),
)
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_mnist(data_root, train_limit, test_limit, device):
    import torchvision
    import torchvision.transforms as T
    train_ds = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=T.ToTensor())
    test_ds = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=T.ToTensor())
    x_train = torch.stack([train_ds[i][0] for i in range(min(train_limit, len(train_ds)))]).view(-1, 784).to(device)
    y_train = torch.tensor([train_ds[i][1] for i in range(min(train_limit, len(train_ds)))]).to(device)
    x_test = torch.stack([test_ds[i][0] for i in range(min(test_limit, len(test_ds)))]).view(-1, 784).to(device)
    y_test = torch.tensor([test_ds[i][1] for i in range(min(test_limit, len(test_ds)))]).to(device)
    return x_train, y_train, x_test, y_test


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
# Plotting helpers
# -----------------------------------------------------------------------

def mean_std(arr_list):
    """Compute mean and std over a list of equal-length arrays."""
    stacked = np.array(arr_list)
    return stacked.mean(axis=0), stacked.std(axis=0)


def format_b_target(value):
    """Format target-B values consistently for plots and JSON keys."""
    return f"{value:.2f}".rstrip('0').rstrip('.')


COLORS = {'Constant': '#1f77b4', 'Step': '#ff7f0e', 'Cosine': '#2ca02c'}


def smooth(arr, window=200):
    """Moving average smoothing for long time series."""
    if window <= 1 or len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def plot_band(ax, x, mean, std, color, label, alpha=0.25, smooth_window=0):
    if smooth_window > 1 and len(mean) > smooth_window:
        mean = smooth(mean, smooth_window)
        std = smooth(std, smooth_window)
    ax.plot(x, mean, color=color, label=label, linewidth=1.2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


# -----------------------------------------------------------------------
# Experiment 1: Multi-seed BN MLP on MNIST
# -----------------------------------------------------------------------

def run_mlp_multiseed(args):
    print("\n=== Multi-seed BN MLP on MNIST ===")
    x_train, y_train, x_test, y_test = load_mnist(
        args.data_root, args.mlp_train_limit, args.mlp_test_limit, args.device
    )

    T_mlp = args.mlp_steps
    lr_hi, lr_lo = args.mlp_lr_high, args.mlp_lr_low
    wd = args.mlp_wd
    schedules = {
        'Constant': schedule_constant(lr_hi, wd, T_mlp),
        'Step': schedule_step(lr_hi, lr_lo, wd, T_mlp, T_mlp // 2),
        'Cosine': schedule_cosine(lr_hi, lr_lo, wd, T_mlp),
    }

    all_results = {sname: [] for sname in schedules}

    for seed in range(args.n_seeds):
        for sname, (etas, lams) in schedules.items():
            actual_seed = args.seed_base + seed
            print(f"  MLP {sname} seed={actual_seed}")
            torch.manual_seed(actual_seed)
            model = SimpleNormMLP(norm_kind="bn", hidden_dim=128).to(args.device)
            hist = run_mlp_schedule(
                model, x_train, y_train, x_test, y_test,
                etas, lams, batch_size=128, seed=actual_seed,
                log_every=0, schedule_name=f'{sname}_s{actual_seed}'
            )
            all_results[sname].append(hist)

    # Plot: 2 rows x 3 cols (schedule). Top: B_t + expansion. Bottom: loss + acc.
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex='col')
    for j, sname in enumerate(['Constant', 'Step', 'Cosine']):
        etas, lams = schedules[sname]
        _, B_arr = schedule_controls(etas, lams)
        steps = np.arange(len(B_arr))
        hists = all_results[sname]

        # Top: B_t + expansion fraction
        ax = axes[0, j]
        ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=1, label=r'$B_t$')
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)
        ax.set_ylabel(r'$B_t$' if j == 0 else '')
        ax.set_title(sname, fontsize=11)

        ax2 = ax.twinx()
        exp_m, exp_s = mean_std([h['expansion_fraction'] for h in hists])
        plot_band(ax2, steps, exp_m, exp_s, COLORS[sname], 'Expansion frac.')
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel('Expansion frac.' if j == 2 else '')

        # Bottom: loss + accuracy
        ax = axes[1, j]
        loss_m, loss_s = mean_std([h['train_loss'] for h in hists])
        plot_band(ax, steps, loss_m, loss_s, COLORS[sname], 'Train loss')
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('Train loss' if j == 0 else '')

        ax2 = ax.twinx()
        acc_m, acc_s = mean_std([h['test_acc'] for h in hists])
        plot_band(ax2, steps, acc_m, acc_s, 'darkred', 'Test acc.')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel('Test accuracy' if j == 2 else '')

    fig.suptitle('BN MLP on MNIST: schedule transitions (5-seed mean $\\pm$ std)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_mlp_multiseed.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Save summary stats
    summary = {}
    for sname, hists in all_results.items():
        accs = [max(h['test_acc']) for h in hists]
        exp_fracs = [np.mean(h['expansion_fraction'][len(h['expansion_fraction'])//2:]) for h in hists]
        summary[sname] = {
            'best_acc_mean': float(np.mean(accs)),
            'best_acc_std': float(np.std(accs)),
            'expansion_second_half_mean': float(np.mean(exp_fracs)),
            'expansion_second_half_std': float(np.std(exp_fracs)),
        }
    with open(os.path.join(args.out_dir, 'mlp_multiseed_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, 'mlp_histories.json'), 'w') as f:
        json.dump(all_results, f)

    return all_results


# -----------------------------------------------------------------------
# Experiment 2: Multi-seed CIFAR ConvNet
# -----------------------------------------------------------------------

def run_conv_multiseed(args):
    print("\n=== Multi-seed BN ConvNet on CIFAR-10 ===")
    x_train, y_train, x_test, y_test = load_cifar10(
        args.data_root, args.conv_train_limit, args.conv_test_limit, args.device
    )

    T_conv = args.conv_steps
    lr_hi, lr_lo = args.conv_lr_high, args.conv_lr_low
    wd = args.conv_wd
    schedules = {
        'Constant': schedule_constant(lr_hi, wd, T_conv),
        'Step': schedule_step(lr_hi, lr_lo, wd, T_conv, T_conv // 2),
        'Cosine': schedule_cosine(lr_hi, lr_lo, wd, T_conv),
    }

    all_results = {sname: [] for sname in schedules}
    eval_every = max(1, int(np.ceil(min(args.conv_train_limit, 50000) / 128)))  # ~1 epoch

    for seed in range(args.n_seeds):
        for sname, (etas, lams) in schedules.items():
            actual_seed = args.seed_base + seed
            print(f"  Conv {sname} seed={actual_seed}")
            torch.manual_seed(actual_seed)
            model = SmallConvBNNet(num_classes=10).to(args.device)
            hist = run_conv_schedule(
                model, x_train, y_train, x_test, y_test,
                etas, lams, batch_size=128, seed=actual_seed,
                log_every=0, eval_every=eval_every,
                schedule_name=f'{sname}_s{actual_seed}'
            )
            all_results[sname].append(hist)

    # Save full histories for replotting
    hist_path = os.path.join(args.out_dir, 'conv_histories.json')
    serializable = {}
    for sn in all_results:
        serializable[sn] = []
        for h in all_results[sn]:
            sh = {}
            for k, v in h.items():
                if isinstance(v, list):
                    sh[k] = [float(x) if isinstance(x, (float, int, np.floating)) else x for x in v]
                else:
                    sh[k] = v
            serializable[sn].append(sh)
    with open(hist_path, 'w') as f:
        json.dump(serializable, f)
    print(f"Saved conv histories to {hist_path}")

    # Plot: 3 rows (layers) x 3 cols (schedules). Each: B_t + expansion.
    layer_names = ['early conv1', 'middle conv2', 'late conv3']
    win = 200 if T_conv >= 5000 else 0  # Smooth long runs for readability
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
            plot_band(ax2, steps, exp_m, exp_s, COLORS[sname], 'Expansion',
                      smooth_window=win)
            ax2.set_ylim(-0.05, 1.05)

            if i == 2:
                ax.set_xlabel('Step')

    fig.suptitle('BN ConvNet on CIFAR-10: filterwise transitions (5-seed mean $\\pm$ std)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_conv_multiseed.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for sname, hists in all_results.items():
        eval_ts = hists[0]['eval_epoch']
        acc_m, acc_s = mean_std([h['eval_test_acc'] for h in hists])
        plot_band(axes[0], eval_ts, acc_m, acc_s, COLORS[sname], sname)
        loss_m, loss_s = mean_std([h['eval_test_loss'] for h in hists])
        plot_band(axes[1], eval_ts, loss_m, loss_s, COLORS[sname], sname)
    axes[0].set_ylabel('Test accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Test loss'); axes[1].set_xlabel('Epoch'); axes[1].set_yscale('log')
    fig.suptitle('CIFAR-10 ConvNet training curves (5-seed)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_conv_multiseed_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Summary
    summary = {}
    for sname, hists in all_results.items():
        accs = [max(h['eval_test_acc']) for h in hists]
        summary[sname] = {
            'best_acc_mean': float(np.mean(accs)),
            'best_acc_std': float(np.std(accs)),
        }
    with open(os.path.join(args.out_dir, 'conv_multiseed_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return all_results


# -----------------------------------------------------------------------
# Experiment 3: Multi-seed target-constant-B sweep
# -----------------------------------------------------------------------

def run_target_b_multiseed(args):
    print("\n=== Multi-seed target-constant-B sweep on CIFAR-10 ===")
    x_train, y_train, x_test, y_test = load_cifar10(
        args.data_root, args.conv_train_limit, args.conv_test_limit, args.device
    )

    b_targets = [
        float(token.strip())
        for token in args.target_b_values.split(',')
        if token.strip()
    ]
    if not b_targets:
        raise ValueError("target_b_values must contain at least one numeric value")
    eval_every = max(1, int(np.ceil(min(args.conv_train_limit, 50000) / 128)))

    # results[b_target] = list of per-seed dicts
    results = {b: [] for b in b_targets}

    for b_target in b_targets:
        etas, lams = schedule_target_constant_b(
            args.conv_lr_high, args.conv_wd, b_target, args.target_b_steps
        )
        _, B_arr = schedule_controls(etas, lams)
        print(f"\n  B_target={b_target}: LR range [{etas.min():.2e}, {etas.max():.2e}], B range [{B_arr.min():.4f}, {B_arr.max():.4f}]")

        for seed in range(args.n_seeds):
            actual_seed = args.seed_base + seed
            print(f"    seed={actual_seed}")
            torch.manual_seed(actual_seed)
            model = SmallConvBNNet(num_classes=10).to(args.device)
            hist = run_conv_schedule(
                model, x_train, y_train, x_test, y_test,
                etas, lams, batch_size=128, seed=actual_seed,
                log_every=0, eval_every=eval_every,
                schedule_name=f'B={b_target}_s{actual_seed}'
            )
            test_accs = hist.get('eval_test_acc', [])
            results[b_target].append({
                'best_acc': max(test_accs) if test_accs else 0.0,
                'final_acc': test_accs[-1] if test_accs else 0.0,
                'eta_final': float(etas[-1]),
                'B_min': float(B_arr.min()),
                'B_max': float(B_arr.max()),
            })

    # Plot: bar chart of best accuracy with error bars
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 3.8))
    b_vals = b_targets
    best_means = [np.mean([r['best_acc'] for r in results[b]]) for b in b_vals]
    best_stds = [np.std([r['best_acc'] for r in results[b]]) for b in b_vals]
    bar_labels = [format_b_target(b) for b in b_vals]

    colors_bar = ['#4e79a7' if b < 1.0 else '#e15759' if b > 1.0 else '#59a14f' for b in b_vals]
    ax.bar(bar_labels, best_means, yerr=best_stds,
           color=colors_bar, edgecolor='black', linewidth=0.5,
           capsize=4, error_kw={'linewidth': 1.2})
    ax.set_xlabel(r'Target $B_t$')
    ax.set_ylabel('Best test accuracy')
    ax.set_title(rf'Target-constant-$B_t$ sweep on CIFAR-10 ({args.n_seeds}-seed mean $\pm$ std)')
    ax.axhline(best_means[1], color='gray', linestyle=':', linewidth=0.8)  # B=1 reference
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_target_b_multiseed.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Summary
    summary = {}
    for b in b_vals:
        accs = [r['best_acc'] for r in results[b]]
        final_accs = [r['final_acc'] for r in results[b]]
        eta_finals = [r['eta_final'] for r in results[b]]
        summary[format_b_target(b)] = {
            'best_acc_mean': float(np.mean(accs)),
            'best_acc_std': float(np.std(accs)),
            'best_accs': [float(a) for a in accs],
            'final_acc_mean': float(np.mean(final_accs)),
            'final_acc_std': float(np.std(final_accs)),
            'eta_final_mean': float(np.mean(eta_finals)),
            'eta_final_std': float(np.std(eta_finals)),
            'B_min': float(results[b][0]['B_min']),
            'B_max': float(results[b][0]['B_max']),
        }
    with open(os.path.join(args.out_dir, 'target_b_multiseed_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=42)

    # MLP config
    parser.add_argument('--mlp_steps', type=int, default=120)
    parser.add_argument('--mlp_lr_high', type=float, default=0.5)
    parser.add_argument('--mlp_lr_low', type=float, default=0.05)
    parser.add_argument('--mlp_wd', type=float, default=0.05)
    parser.add_argument('--mlp_train_limit', type=int, default=10000)
    parser.add_argument('--mlp_test_limit', type=int, default=5000)

    # Conv config
    parser.add_argument('--conv_steps', type=int, default=10000)
    parser.add_argument('--conv_lr_high', type=float, default=0.5)
    parser.add_argument('--conv_lr_low', type=float, default=0.05)
    parser.add_argument('--conv_wd', type=float, default=0.05)
    parser.add_argument('--conv_train_limit', type=int, default=10000)
    parser.add_argument('--conv_test_limit', type=int, default=5000)

    # Target-B config
    parser.add_argument('--target_b_steps', type=int, default=10000)
    parser.add_argument('--target_b_values', type=str,
                        default='0.90,0.95,0.98,1.00,1.02,1.05,1.10')

    # What to run
    parser.add_argument('--skip_mlp', action='store_true')
    parser.add_argument('--skip_conv', action='store_true')
    parser.add_argument('--skip_target_b', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_mlp:
        run_mlp_multiseed(args)
    if not args.skip_conv:
        run_conv_multiseed(args)
    if not args.skip_target_b:
        run_target_b_multiseed(args)

    print("\n=== All multi-seed experiments complete ===")


if __name__ == '__main__':
    main()
