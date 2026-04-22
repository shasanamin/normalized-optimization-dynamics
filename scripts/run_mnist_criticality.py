"""
run_mnist_criticality.py -- Multi-seed near-critical controls on MNIST.

Compares two ways of staying near the contraction/expansion boundary:
removing weight decay entirely versus keeping positive weight decay while
explicitly enforcing target B_t schedules.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict

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
from src.models import SimpleNormMLP
from src.theory import schedule_constant, schedule_target_b_sequence, schedule_target_constant_b
from src.training import run_mlp_schedule


DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(__file__), '..', 'data'),
)
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_mnist(data_root, train_limit, test_limit, device):
    import torchvision
    import torchvision.transforms as T

    train_ds = torchvision.datasets.MNIST(
        data_root, train=True, download=True, transform=T.ToTensor()
    )
    test_ds = torchvision.datasets.MNIST(
        data_root, train=False, download=True, transform=T.ToTensor()
    )

    x_train = torch.stack(
        [train_ds[i][0] for i in range(min(train_limit, len(train_ds)))]
    ).view(-1, 784).to(device)
    y_train = torch.tensor(
        [train_ds[i][1] for i in range(min(train_limit, len(train_ds)))]
    ).to(device)
    x_test = torch.stack(
        [test_ds[i][0] for i in range(min(test_limit, len(test_ds)))]
    ).view(-1, 784).to(device)
    y_test = torch.tensor(
        [test_ds[i][1] for i in range(min(test_limit, len(test_ds)))]
    ).to(device)
    return x_train, y_train, x_test, y_test


def mean_std(arr_list):
    stacked = np.asarray(arr_list, dtype=float)
    return stacked.mean(axis=0), stacked.std(axis=0)


def summarize_schedule(histories, etas):
    best_accs = [float(np.max(hist['test_acc'])) for hist in histories]
    final_accs = [float(hist['test_acc'][-1]) for hist in histories]
    mean_expansions = [float(np.mean(hist['expansion_fraction'])) for hist in histories]
    return {
        'B_min': float(np.min(histories[0]['B_t'])),
        'B_max': float(np.max(histories[0]['B_t'])),
        'mean_expansion_mean': float(np.mean(mean_expansions)),
        'mean_expansion_std': float(np.std(mean_expansions)),
        'best_acc_mean': float(np.mean(best_accs)),
        'best_acc_std': float(np.std(best_accs)),
        'final_acc_mean': float(np.mean(final_accs)),
        'final_acc_std': float(np.std(final_accs)),
        'final_eta': float(etas[-1]),
        'best_accs': best_accs,
        'final_accs': final_accs,
    }


def plot_criticality_controls(results, schedule_order, out_path, n_seeds):
    fig, axes = plt.subplots(5, len(schedule_order), figsize=(4.0 * len(schedule_order), 11.0), sharex='col')

    for col, schedule_name in enumerate(schedule_order):
        payload = results[schedule_name]
        histories = payload['histories']
        steps = np.arange(len(payload['etas']) - 1)

        axes[0, col].plot(steps, payload['etas'][:-1], color='#1f77b4', linewidth=1.2)
        axes[0, col].set_title(schedule_name, fontsize=11)
        axes[0, col].set_ylabel(r'$\eta_t$')
        axes[0, col].grid(True, alpha=0.2)

        axes[1, col].plot(steps, histories[0]['B_t'], color='#ff7f0e', linewidth=1.2)
        axes[1, col].axhline(1.0, color='black', linestyle=':', linewidth=0.8)
        axes[1, col].set_ylabel(r'$B_t$')
        axes[1, col].grid(True, alpha=0.2)

        expansion_mean, expansion_std = mean_std([hist['expansion_fraction'] for hist in histories])
        axes[2, col].plot(steps, expansion_mean, color='#2ca02c', linewidth=1.2)
        axes[2, col].fill_between(steps, expansion_mean - expansion_std, expansion_mean + expansion_std,
                                  color='#2ca02c', alpha=0.2)
        axes[2, col].set_ylabel('Expansion')
        axes[2, col].set_ylim(-0.05, 1.05)
        axes[2, col].grid(True, alpha=0.2)

        train_loss_mean, train_loss_std = mean_std([hist['train_loss'] for hist in histories])
        axes[3, col].plot(steps, train_loss_mean, color='#9467bd', linewidth=1.2)
        axes[3, col].fill_between(steps, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                                  color='#9467bd', alpha=0.2)
        axes[3, col].set_ylabel('Train loss')
        axes[3, col].grid(True, alpha=0.2)

        test_acc_mean, test_acc_std = mean_std([hist['test_acc'] for hist in histories])
        axes[4, col].plot(steps, test_acc_mean, color='#d62728', linewidth=1.2)
        axes[4, col].fill_between(steps, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std,
                                  color='#d62728', alpha=0.2)
        axes[4, col].set_ylabel('Test acc.')
        axes[4, col].set_xlabel('Step')
        axes[4, col].grid(True, alpha=0.2)

    fig.suptitle(
        rf'Near-critical controls on MNIST ({n_seeds}-seed mean $\pm$ std)\n'
        r'Separating zero weight decay from positive-decay criticality',
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=42)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--lr0', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--train_limit', type=int, default=10000)
    parser.add_argument('--test_limit', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_every', type=int, default=0)
    parser.add_argument('--square_b_low', type=float, default=0.97)
    parser.add_argument('--square_b_high', type=float, default=1.03)
    parser.add_argument('--square_wave_half_period', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist(
        args.data_root, args.train_limit, args.test_limit, args.device
    )

    square_b = np.array([
        args.square_b_low if (t // args.square_wave_half_period) % 2 == 0 else args.square_b_high
        for t in range(args.steps)
    ], dtype=float)
    schedules = OrderedDict([
        ('No WD, constant ($\\lambda=0$)', schedule_constant(args.lr0, 0.0, args.steps)),
        ('WD, target $B=1.00$', schedule_target_constant_b(args.lr0, args.weight_decay, 1.00, args.steps)),
        ('WD, target $B=1.02$', schedule_target_constant_b(args.lr0, args.weight_decay, 1.02, args.steps)),
        ('WD, target $B=0.98$', schedule_target_constant_b(args.lr0, args.weight_decay, 0.98, args.steps)),
        ('WD, square-wave $B_t$', schedule_target_b_sequence(args.lr0, args.weight_decay, square_b)),
    ])

    results = OrderedDict()
    histories_json = OrderedDict()
    summary = OrderedDict()

    for schedule_name, (etas, lams) in schedules.items():
        print(f"\\n=== {schedule_name} ===")
        schedule_histories = []
        for seed_offset in range(args.n_seeds):
            actual_seed = args.seed_base + seed_offset
            print(f"  seed={actual_seed}")
            torch.manual_seed(actual_seed)
            model = SimpleNormMLP(norm_kind='bn', hidden_dim=128).to(args.device)
            hist = run_mlp_schedule(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                etas,
                lams,
                batch_size=args.batch_size,
                seed=actual_seed,
                log_every=args.log_every,
                schedule_name=f'{schedule_name}_s{actual_seed}',
            )
            schedule_histories.append(hist)

        results[schedule_name] = {
            'etas': etas,
            'lams': lams,
            'histories': schedule_histories,
        }
        histories_json[schedule_name] = schedule_histories
        summary[schedule_name] = summarize_schedule(schedule_histories, etas)

    fig_path = os.path.join(args.out_dir, 'fig_extended_criticality_controls.png')
    plot_criticality_controls(results, list(schedules.keys()), fig_path, args.n_seeds)
    print(f"Saved {fig_path}")

    with open(os.path.join(args.out_dir, 'mnist_criticality_multiseed_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, 'mnist_criticality_multiseed_histories.json'), 'w') as f:
        json.dump(histories_json, f)


if __name__ == '__main__':
    main()