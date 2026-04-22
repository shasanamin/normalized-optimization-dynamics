"""
run_minibatch_sweep.py -- Test B_t regime robustness across batch sizes.

Trains BN MLP on MNIST under 3 schedules × 4 batch sizes × 5 seeds,
tracking expansion fraction and ratio residual per step.

Key claim: the B_t-controlled regime transitions (cosine suppression,
step-decay shock) survive at all batch sizes, with post-cosine expansion
fraction ≈ 0.02-0.03 across all settings.

Generates:
- fig_minibatch.png: 2×3 figure (expansion frac + ratio residual across batch sizes)
- minibatch_histories.json: full per-step histories
- minibatch_summary.json: aggregate stats
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
from src.theory import schedule_constant, schedule_step, schedule_cosine
from src.models import SimpleNormMLP
from src.training import run_mlp_schedule

import torch
import torch.nn as nn

DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(__file__), '..', 'data'),
)
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist(data_root, device='cpu'):
    """Load MNIST via torchvision, downloading automatically if needed."""
    import torchvision
    import torchvision.transforms as T

    train_ds = torchvision.datasets.MNIST(
        data_root, train=True, download=True, transform=T.ToTensor()
    )
    test_ds = torchvision.datasets.MNIST(
        data_root, train=False, download=True, transform=T.ToTensor()
    )

    x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).view(-1, 784)
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).view(-1, 784)
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    return x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)


def smooth(arr, window=10):
    if window <= 1 or len(arr) <= window:
        return np.array(arr)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='results/minibatch')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--lr_high', type=float, default=0.5)
    parser.add_argument('--lr_low', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    T = args.steps
    BATCH_SIZES = [512, 256, 128, 32]
    BS_LABELS = ['BS=512', 'BS=256', 'BS=128', 'BS=32']
    BS_COLORS = ['#333333', '#1f77b4', '#ff7f0e', '#d62728']
    SCHED_NAMES = ['Constant', 'Step Decay', 'Cosine Decay']
    SCHED_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

    schedules_raw = {
        'Constant': schedule_constant(args.lr_high, args.weight_decay, T),
        'Step Decay': schedule_step(args.lr_high, args.lr_low, args.weight_decay, T, T // 2),
        'Cosine Decay': schedule_cosine(args.lr_high, args.lr_low, args.weight_decay, T),
    }

    print(f"Loading MNIST from {args.data_root}...")
    x_train, y_train, x_test, y_test = load_mnist(args.data_root, args.device)
    print(f"  Train: {x_train.shape}, Device: {args.device}")

    # ---- Run experiments -----------------------------------------------
    # all_results[sname][bs_label] = list of seed histories
    all_results = {sname: {bl: [] for bl in BS_LABELS} for sname in SCHED_NAMES}

    total_runs = len(SCHED_NAMES) * len(BATCH_SIZES) * len(args.seeds)
    run_idx = 0

    for sname in SCHED_NAMES:
        etas, lams = schedules_raw[sname]
        for bs, bl in zip(BATCH_SIZES, BS_LABELS):
            for seed in args.seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {sname} | {bl} | seed={seed}")
                torch.manual_seed(seed)
                model = SimpleNormMLP(
                    input_dim=784, hidden_dim=args.hidden_dim, num_classes=10,
                    norm_kind='bn', eps=1e-5,
                ).to(args.device)
                hist = run_mlp_schedule(
                    model, x_train, y_train, x_test, y_test,
                    etas, lams,
                    batch_size=bs,
                    seed=seed,
                    log_every=60,
                    schedule_name=f'{sname}/{bl}',
                )
                all_results[sname][bl].append(hist)

    # ---- Save histories ------------------------------------------------
    json_path = os.path.join(args.out_dir, 'minibatch_histories.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f)
    print(f"\nSaved histories -> {json_path}")

    # ---- Summary stats -------------------------------------------------
    summary = {}
    for sname in SCHED_NAMES:
        summary[sname] = {}
        for bl in BS_LABELS:
            hists = all_results[sname][bl]
            exp_second_half = np.mean([np.mean(h['expansion_fraction'][T//2:]) for h in hists])
            rr_median = np.mean([np.median(h['ratio_residual']) for h in hists])
            best_acc = np.mean([np.max(h['test_acc']) for h in hists])
            summary[sname][bl] = {
                'expansion_second_half': float(exp_second_half),
                'ratio_residual_median': float(rr_median),
                'best_test_acc': float(best_acc),
            }
            print(f"  {sname:15s} {bl:12s}: expand_2H={exp_second_half:.3f}  "
                  f"rr={rr_median:.2e}  acc={best_acc:.3f}")

    summary_path = os.path.join(args.out_dir, 'minibatch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {summary_path}")

    # ---- Figure --------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5), sharex=True)

    WIN = 10  # mild smoothing for 120-step runs

    for col, sname in enumerate(SCHED_NAMES):
        ax_exp = axes[0, col]
        ax_rr = axes[1, col]

        for bs, bl, color in zip(BATCH_SIZES, BS_LABELS, BS_COLORS):
            hists = all_results[sname][bl]
            exp_m = np.mean([h['expansion_fraction'] for h in hists], axis=0)
            exp_s = np.std([h['expansion_fraction'] for h in hists], axis=0)
            rr_m = np.mean([h['ratio_residual'] for h in hists], axis=0)
            steps = np.arange(T)

            exp_sm = smooth(exp_m, WIN)
            rr_sm = smooth(rr_m, WIN)

            ax_exp.plot(steps, exp_sm, color=color, linewidth=1.2, label=bl)
            ax_exp.fill_between(steps,
                                smooth(exp_m - exp_s, WIN),
                                smooth(exp_m + exp_s, WIN),
                                color=color, alpha=0.12)

            ax_rr.plot(steps, rr_sm, color=color, linewidth=1.2, label=bl)

        ax_exp.set_ylim(-0.05, 1.05)
        ax_exp.set_title(sname, fontsize=11, fontweight='bold')
        if col == 0:
            ax_exp.set_ylabel('Expansion fraction', fontsize=9)
            ax_exp.legend(fontsize=7, loc='upper right')

        ax_rr.set_yscale('log')
        ax_rr.set_ylim(1e-8, 2)
        ax_rr.set_xlabel('Step', fontsize=9)
        ax_rr.grid(True, alpha=0.2)
        if col == 0:
            ax_rr.set_ylabel('Ratio residual', fontsize=9)

    fig.suptitle(
        'Minibatch robustness: $B_t$ regime transitions survive all batch sizes (BN-MLP on MNIST, 5-seed mean)',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = os.path.join(args.out_dir, 'fig_minibatch.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure -> {fig_path}")

    print("\n=== Minibatch sweep done ===")


if __name__ == '__main__':
    main()
