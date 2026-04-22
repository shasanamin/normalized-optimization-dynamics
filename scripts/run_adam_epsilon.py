"""
run_adam_epsilon.py -- Adam epsilon-continuity study on the CIFAR-10 BN ConvNet.

This script measures how the epsilon=0 Adam theorem degrades as epsilon grows.
For each epsilon value it tracks:
- the recurrence residual against the epsilon=0 augmented-state law,
- the exact algebraic residual using the actual epsilon-perturbed direction,
- the scale-breaking gap between p_t^(epsilon) and the epsilon=0 direction,
- expansion fractions and exponent scans on a representative convolutional layer.

Usage:
    python scripts/run_adam_epsilon.py \
      --out_dir results/adam_epsilon \
      --data_root /path/to/data \
      --device cuda:0
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
from src.theory import schedule_constant, schedule_step, schedule_cosine
from src.models import SmallConvBNNet
from src.diagnostics import compute_adam_block_metrics, flatten_filter_blocks


DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(__file__), '..', 'data'),
)
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_EPS_VALUES = [1e-12, 1e-10, 1e-8, 1e-6]
EXPONENT_GRID = np.linspace(0.5, 3.0, 26, dtype=float)
EPS_COLORS = {
    '1e-12': '#1f77b4',
    '1e-10': '#2ca02c',
    '1e-08': '#ff7f0e',
    '1e-06': '#d62728',
}


def eps_label(value: float) -> str:
    return f'{value:.0e}'


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


def prepare_adam_step(
    model,
    lr,
    wd,
    beta1,
    beta2,
    eps,
    state_dict,
    step_count,
    tracked_param_names,
):
    tracked_zero = {}
    tracked_actual = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            if name not in state_dict:
                state_dict[name] = {
                    'm': torch.zeros_like(p),
                    'v': torch.zeros_like(p),
                }
            state = state_dict[name]
            state['m'].mul_(beta1).add_(g, alpha=1 - beta1)
            state['v'].mul_(beta2).addcmul_(g, g, value=1 - beta2)

            m_hat = state['m'] / (1 - beta1 ** step_count)
            v_hat = state['v'] / (1 - beta2 ** step_count)
            sqrt_v = v_hat.sqrt().clamp_min(1e-30)
            p_zero = m_hat / sqrt_v
            p_actual = m_hat / (sqrt_v + eps)

            if name in tracked_param_names.values():
                tracked_zero[name] = flatten_filter_blocks(p_zero.clone())
                tracked_actual[name] = flatten_filter_blocks(p_actual.clone())

            p.mul_(1.0 - lr * wd).add_(p_actual, alpha=-lr)

    return tracked_zero, tracked_actual


def run_adam_eps_conv(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    etas,
    lams,
    eps,
    beta1,
    beta2,
    batch_size=128,
    seed=7,
    log_every=2000,
    eval_every=79,
    schedule_name="",
):
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
        'train_loss': [], 'B_tilde': [],
        'eval_t': [], 'eval_epoch': [],
        'eval_train_loss': [], 'eval_train_acc': [],
        'eval_test_loss': [], 'eval_test_acc': [],
        'epsilon': eps,
        'exponents': EXPONENT_GRID.tolist(),
    }
    exponent_sums = {}
    for name in layer_names:
        history[f'{name}_expansion'] = []
        history[f'{name}_eps0_residual'] = []
        history[f'{name}_actual_direction_residual'] = []
        history[f'{name}_threshold_accuracy'] = []
        history[f'{name}_scale_break_gap'] = []
        history[f'{name}_c_median'] = []
        history[f'{name}_denom_zero_median'] = []
        history[f'{name}_denom_actual_median'] = []
        history[f'{name}_radius_residual'] = []
        exponent_sums[name] = np.zeros_like(EXPONENT_GRID)

    adam_state = {}
    model.train()

    for t in range(T):
        n = x_train.shape[0]
        if batch_size >= n:
            xb, yb = x_train, y_train
        else:
            idx = torch.randint(n, (batch_size,), generator=gen, device='cpu')
            xb, yb = x_train[idx], y_train[idx]

        snaps_before = {}
        for name, layer in layers.items():
            snaps_before[name] = flatten_filter_blocks(layer.weight.data.clone())

        model.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        loss.backward()

        p_zero, p_actual = prepare_adam_step(
            model,
            etas[t],
            lams[t],
            beta1,
            beta2,
            eps,
            adam_state,
            t + 1,
            tracked_param_names,
        )

        snaps_after = {}
        for name, layer in layers.items():
            snaps_after[name] = flatten_filter_blocks(layer.weight.data.clone())

        b_tilde = (etas[t + 1] / etas[t]) / (1 - etas[t + 1] * lams[t + 1])
        history['train_loss'].append(loss.item())
        history['B_tilde'].append(b_tilde)

        for name in layer_names:
            metrics = compute_adam_block_metrics(
                snaps_before[name],
                p_zero[tracked_param_names[name]],
                p_actual[tracked_param_names[name]],
                snaps_after[name],
                etas[t],
                lams[t],
                etas[t + 1],
                lams[t + 1],
                exponents=exponent_tensor,
            )
            history[f'{name}_expansion'].append(metrics['expansion_fraction'])
            history[f'{name}_eps0_residual'].append(metrics['eps0_residual_median'])
            history[f'{name}_actual_direction_residual'].append(metrics['actual_direction_residual_median'])
            history[f'{name}_threshold_accuracy'].append(metrics['threshold_accuracy'])
            history[f'{name}_scale_break_gap'].append(metrics['scale_break_gap_median'])
            history[f'{name}_c_median'].append(metrics['c_median'])
            history[f'{name}_denom_zero_median'].append(metrics['denom_zero_median'])
            history[f'{name}_denom_actual_median'].append(metrics['denom_actual_median'])
            history[f'{name}_radius_residual'].append(metrics['radius_residual_median'])
            exponent_sums[name] += np.array(metrics['exponent_scan'], dtype=float)

        if eval_every and (t % eval_every == 0 or t == T - 1):
            model.eval()
            with torch.no_grad():
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
            te_acc_str = f"{history['eval_test_acc'][-1]:.3f}" if history['eval_test_acc'] else 'N/A'
            print(f"  [{schedule_name}] step {t+1}/{T}  loss={loss.item():.4f}  test_acc={te_acc_str}")

    for name in layer_names:
        history[f'{name}_exponent_scan_mean'] = (exponent_sums[name] / max(T, 1)).tolist()

    return history


def mean_std_arr(arr_list):
    stacked = np.array(arr_list)
    return stacked.mean(axis=0), stacked.std(axis=0)


def smooth_arr(arr, window=200):
    if window <= 1 or len(arr) <= window:
        return np.array(arr)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--seed_base', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--lr_high', type=float, default=0.5)
    parser.add_argument('--lr_low', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--train_limit', type=int, default=10000)
    parser.add_argument('--test_limit', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps_values', type=float, nargs='+', default=DEFAULT_EPS_VALUES)
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

    all_results = {}
    for eps in args.eps_values:
        eps_key = eps_label(eps)
        all_results[eps_key] = {}
        for sname, (etas, lams) in schedules.items():
            all_results[eps_key][sname] = []
            for seed_idx in range(args.n_seeds):
                actual_seed = args.seed_base + seed_idx
                print(f"\n  eps={eps_key} | {sname} | seed={actual_seed}")
                torch.manual_seed(actual_seed)
                model = SmallConvBNNet(num_classes=10).to(args.device)
                history = run_adam_eps_conv(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    etas,
                    lams,
                    eps=eps,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    batch_size=args.batch_size,
                    seed=actual_seed,
                    log_every=2000,
                    eval_every=eval_every,
                    schedule_name=f'adam_eps_{eps_key}_{sname}_s{actual_seed}',
                )
                all_results[eps_key][sname].append(history)

    hist_path = os.path.join(args.out_dir, 'adam_epsilon_histories.json')
    serializable = {}
    for eps_key, sched_data in all_results.items():
        serializable[eps_key] = {}
        for sname, histories in sched_data.items():
            serializable[eps_key][sname] = []
            for history in histories:
                serialized = {}
                for key, value in history.items():
                    if isinstance(value, list):
                        serialized[key] = [float(x) if isinstance(x, (float, int, np.floating)) else x for x in value]
                    else:
                        serialized[key] = float(value) if isinstance(value, (float, int, np.floating)) else value
                serializable[eps_key][sname].append(serialized)
    with open(hist_path, 'w') as f:
        json.dump(serializable, f)
    print(f'Saved histories to {hist_path}')

    # Figure 1: epsilon-continuity of the theorem residual
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    eps_numeric = np.array(args.eps_values, dtype=float)
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        ax = axes[col]
        residual_means = []
        residual_stds = []
        gap_means = []
        for eps in args.eps_values:
            eps_key = eps_label(eps)
            hists = all_results[eps_key][sname]
            residuals = [np.median(h['middle conv2_eps0_residual']) for h in hists]
            gaps = [np.median(h['middle conv2_scale_break_gap']) for h in hists]
            residual_means.append(np.mean(residuals))
            residual_stds.append(np.std(residuals))
            gap_means.append(np.mean(gaps))
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
    path = os.path.join(args.out_dir, 'fig_adam_epsilon_residual.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')

    # Figure 2: schedule transitions persist across epsilon values
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        ax = axes[col]
        for eps in args.eps_values:
            eps_key = eps_label(eps)
            hists = all_results[eps_key][sname]
            exp_m, exp_s = mean_std_arr([h['middle conv2_expansion'] for h in hists])
            exp_ms = smooth_arr(exp_m, 200)
            exp_ss = smooth_arr(exp_s, 200)
            steps = np.arange(len(exp_ms))
            color = EPS_COLORS.get(eps_key, None)
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
    path = os.path.join(args.out_dir, 'fig_adam_epsilon_expansion.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')

    summary = {
        'eps_values': [float(eps) for eps in args.eps_values],
        'schedules': {},
        'aggregated_exponent_scan': {},
    }
    for sname in schedules:
        summary['schedules'][sname] = {}
        for eps in args.eps_values:
            eps_key = eps_label(eps)
            hists = all_results[eps_key][sname]
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

    for eps in args.eps_values:
        eps_key = eps_label(eps)
        curves = []
        for sname in schedules:
            hists = all_results[eps_key][sname]
            curves.extend([h['middle conv2_exponent_scan_mean'] for h in hists])
        exponent_mean = np.array(curves).mean(axis=0)
        best_idx = int(np.argmin(exponent_mean))
        summary['aggregated_exponent_scan'][eps_key] = {
            'exponent_grid': EXPONENT_GRID.tolist(),
            'exponent_scan_mean': exponent_mean.tolist(),
            'best_exponent': float(EXPONENT_GRID[best_idx]),
        }

    with open(os.path.join(args.out_dir, 'adam_epsilon_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {args.out_dir}/adam_epsilon_summary.json")
    print('\n=== Adam epsilon study complete ===')


if __name__ == '__main__':
    main()