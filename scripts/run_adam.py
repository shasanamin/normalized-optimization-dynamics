"""
run_adam.py -- Adam / decoupled-decay Adam robustness experiment on CIFAR-10 BN ConvNet.

Extends the SGDM experiment to test whether the B_t regime picture remains
predictive under adaptive optimizers. Tests:
  - SGD (no momentum): baseline (exact recurrence)
  - Adam with coupled WD: p <- (1-lr*wd)*p - lr * m_hat/(sqrt(v_hat)+eps)
  - Adam-style decoupled-decay variant: p <- (1-wd)*p - lr * m_hat/(sqrt(v_hat)+eps)

Usage:
    python scripts/run_adam.py --out_dir results/adam --data_root /path/to/data --device cuda:0
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
from src.diagnostics import flatten_filter_blocks

DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(__file__), '..', 'data'),
)
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------------------------------------------------
# Data loading (same as run_sgdm.py)
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
# Manual Adam / decoupled-decay Adam step functions
# -----------------------------------------------------------------------

def manual_adam_coupled_step(model, lr, wd, beta1, beta2, eps,
                             state_dict, step_count):
    """Adam with coupled weight decay: p <- (1 - lr*wd)*p - lr * m_hat/(sqrt(v_hat)+eps).

    This matches the theory's coupled WD assumption: a_t = 1 - eta_t * lambda_t.
    """
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
            s = state_dict[name]
            s['m'].mul_(beta1).add_(g, alpha=1 - beta1)
            s['v'].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            # Bias correction
            m_hat = s['m'] / (1 - beta1 ** step_count)
            v_hat = s['v'] / (1 - beta2 ** step_count)
            # Coupled WD + Adam step
            p.mul_(1.0 - lr * wd).add_(
                m_hat / (v_hat.sqrt() + eps), alpha=-lr
            )


def manual_adamw_step(model, lr, wd, beta1, beta2, eps,
                       state_dict, step_count):
    """Adam-style decoupled-decay variant: p <- (1 - wd)*p - lr * m_hat/(sqrt(v_hat)+eps).

    This uses the simple per-step shrinkage factor a_t = 1 - lambda_t that the
    paper uses for its decoupled-decay robustness diagnostic.
    """
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
            s = state_dict[name]
            s['m'].mul_(beta1).add_(g, alpha=1 - beta1)
            s['v'].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            m_hat = s['m'] / (1 - beta1 ** step_count)
            v_hat = s['v'] / (1 - beta2 ** step_count)
            # Decoupled WD then Adam step
            p.mul_(1.0 - wd).add_(
                m_hat / (v_hat.sqrt() + eps), alpha=-lr
            )


def manual_sgd_step(model, lr, wd):
    """Plain SGD + coupled weight decay (baseline)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.mul_(1.0 - lr * wd).add_(p.grad, alpha=-lr)


# -----------------------------------------------------------------------
# Blockwise diagnostics (supports both coupled and decoupled a_t)
# -----------------------------------------------------------------------

def compute_block_metrics_general(w_before, g_before, w_after,
                                   eta_t, lam_t, eta_next, lam_next,
                                   decoupled=False):
    """Compute per-block diagnostics, supporting coupled or decoupled WD.

    For coupled WD: a_t = 1 - eta_t * lam_t
    For the decoupled-decay variant: a_t = 1 - lam_t
    """
    if decoupled:
        a_t = 1.0 - lam_t
        a_next = 1.0 - lam_next
    else:
        a_t = 1.0 - eta_t * lam_t
        a_next = 1.0 - eta_next * lam_next

    B_t = eta_next / (eta_t * a_t * a_next)
    threshold = np.sqrt(max(B_t - 1.0, 0.0))

    r_before = torch.norm(w_before, dim=1)
    r_after = torch.norm(w_after, dim=1)

    # Effective directional stepsize
    Phi_before = eta_t / (a_t * r_before ** 2)
    Phi_after = eta_next / (a_next * r_after ** 2)

    # Tangent gradient (for SGD theory comparison)
    wg_dot = (w_before * g_before).sum(dim=1)
    g_norm = torch.norm(g_before, dim=1)
    u_before = w_before / r_before.unsqueeze(1)
    g_par = wg_dot / (r_before + 1e-30)
    g_perp = g_before - g_par.unsqueeze(1) * u_before
    gbar_norm = r_before * torch.norm(g_perp, dim=1)

    # R_t = Phi_t * ||gbar_t||
    R_t = Phi_before * gbar_norm

    # Ratio residual
    ratio_actual = Phi_after / (Phi_before + 1e-30)
    ratio_predicted = B_t / (1.0 + R_t ** 2)
    ratio_residual = torch.abs(ratio_actual - ratio_predicted)

    # Expansion
    expanding = (Phi_after > Phi_before).float()
    if B_t <= 1.0:
        threshold_correct = (1.0 - expanding)
    else:
        predicted_contract = (R_t >= threshold).float()
        actual_contract = 1.0 - expanding
        threshold_correct = (predicted_contract == actual_contract).float()

    # Orthogonality residual (informative for adaptive methods)
    orth_residual = torch.abs(wg_dot) / (r_before * g_norm + 1e-30)

    return {
        'ratio_residual_median': ratio_residual.median().item(),
        'expansion_fraction': expanding.mean().item(),
        'threshold_accuracy': threshold_correct.mean().item(),
        'orth_residual_median': orth_residual.median().item(),
        'B_t': B_t,
        'Phi_median': Phi_before.median().item(),
    }


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def run_adam_conv(model, x_train, y_train, x_test, y_test,
                  etas, lams, method='sgd',
                  beta1=0.9, beta2=0.999, eps=1e-8,
                  batch_size=128, seed=7,
                  log_every=2000, eval_every=79, schedule_name=""):
    """Train ConvNet with Adam / decoupled-decay Adam / SGD and track diagnostics."""
    T = len(etas) - 1
    gen = torch.Generator(device='cpu').manual_seed(seed)
    n_train = x_train.shape[0]
    steps_per_epoch = int(np.ceil(n_train / batch_size))

    layers = model.conv_layers()
    layer_names = list(layers.keys())
    decoupled = (method == 'adamw')

    history = {
        'train_loss': [], 'B_t': [],
        'eval_t': [], 'eval_epoch': [],
        'eval_train_loss': [], 'eval_train_acc': [],
        'eval_test_loss': [], 'eval_test_acc': [],
    }
    for name in layer_names:
        history[f'{name}_expansion'] = []
        history[f'{name}_ratio_residual'] = []
        history[f'{name}_orth_residual'] = []

    adam_state = {}
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

        # Optimizer step
        if method == 'sgd':
            manual_sgd_step(model, etas[t], lams[t])
        elif method == 'adam_coupled':
            manual_adam_coupled_step(model, etas[t], lams[t],
                                     beta1, beta2, eps, adam_state, t + 1)
        elif method == 'adamw':
            manual_adamw_step(model, etas[t], lams[t],
                               beta1, beta2, eps, adam_state, t + 1)

        # Snapshot after
        snaps_after = {}
        for name, layer in layers.items():
            snaps_after[name] = flatten_filter_blocks(layer.weight.data.clone())

        # Diagnostics
        if decoupled:
            B_t = etas[t + 1] / (etas[t] * (1 - lams[t]) * (1 - lams[t + 1]))
        else:
            B_t = etas[t + 1] / (etas[t] * (1 - etas[t] * lams[t]) * (1 - etas[t + 1] * lams[t + 1]))
        history['train_loss'].append(loss.item())
        history['B_t'].append(B_t)

        for name in layer_names:
            m = compute_block_metrics_general(
                snaps_before[name], grads[name], snaps_after[name],
                etas[t], lams[t], etas[t + 1], lams[t + 1],
                decoupled=decoupled
            )
            history[f'{name}_expansion'].append(m['expansion_fraction'])
            history[f'{name}_ratio_residual'].append(m['ratio_residual_median'])
            history[f'{name}_orth_residual'].append(m['orth_residual_median'])

        # Eval
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
            te_str = f"{history['eval_test_acc'][-1]:.3f}" if history['eval_test_acc'] else "N/A"
            print(f"  [{schedule_name}] step {t+1}/{T}  loss={loss.item():.4f}  test_acc={te_str}")

    return history


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
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--lr_high', type=float, default=0.5)
    parser.add_argument('--lr_low', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--train_limit', type=int, default=10000)
    parser.add_argument('--test_limit', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
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

    methods = {
        'SGD': 'sgd',
        'Adam (coupled WD)': 'adam_coupled',
        'AdamW (decoupled WD)': 'adamw',
    }

    all_results = {}
    for method_label, method_key in methods.items():
        all_results[method_label] = {}
        for sname, (etas, lams) in schedules.items():
            all_results[method_label][sname] = []
            for seed_idx in range(args.n_seeds):
                actual_seed = args.seed_base + seed_idx
                print(f"\n  {method_label} | {sname} | seed={actual_seed}")
                torch.manual_seed(actual_seed)
                model = SmallConvBNNet(num_classes=10).to(args.device)
                hist = run_adam_conv(
                    model, x_train, y_train, x_test, y_test,
                    etas, lams, method=method_key,
                    beta1=args.beta1, beta2=args.beta2, eps=args.eps,
                    batch_size=args.batch_size, seed=actual_seed,
                    log_every=2000, eval_every=eval_every,
                    schedule_name=f'{method_label}_{sname}_s{actual_seed}'
                )
                all_results[method_label][sname].append(hist)

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------
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
    hist_path = os.path.join(args.out_dir, 'adam_histories.json')
    serializable = {}
    for ml in all_results:
        serializable[ml] = {}
        for sn in all_results[ml]:
            serializable[ml][sn] = []
            for h in all_results[ml][sn]:
                sh = {}
                for k, v in h.items():
                    if isinstance(v, list):
                        sh[k] = [float(x) if isinstance(x, (float, int, np.floating)) else x for x in v]
                    else:
                        sh[k] = v
                serializable[ml][sn].append(sh)
    with open(hist_path, 'w') as f:
        json.dump(serializable, f)
    print(f"Saved histories to {hist_path}")

    method_labels = list(methods.keys())

    # Figure 1: B_t + expansion fraction (3 rows x 3 cols)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex='col')
    for row, method_label in enumerate(method_labels):
        for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
            ax = axes[row, col]
            etas, lams = schedules[sname]
            _, B_arr = schedule_controls(etas, lams)
            steps = np.arange(len(B_arr))
            hists = all_results[method_label][sname]

            ax.plot(steps, B_arr, color='gray', linestyle='--', linewidth=0.8, label=r'$B_t$ (coupled)')
            ax.axhline(1.0, color='black', linestyle=':', linewidth=0.5)

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
                ax.set_ylabel(method_label + '\n' + r'$B_t$', fontsize=8)
            if col == 2:
                ax2.set_ylabel('Expansion frac.', fontsize=9)
            if row == 2:
                ax.set_xlabel('Step')

    fig.suptitle(r'$B_t$-driven regime transitions: SGD vs Adam vs AdamW (conv2, 5-seed)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_adam_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {path}")

    # Figure 2: Test accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    STYLES = {'SGD': ('--', '#555555'), 'Adam (coupled WD)': ('-', '#d62728'),
              'AdamW (decoupled WD)': ('-', '#9467bd')}
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        ax = axes[col]
        for method_label in method_labels:
            hists = all_results[method_label][sname]
            eval_epochs = hists[0]['eval_epoch']
            acc_m, acc_s = mean_std_arr([h['eval_test_acc'] for h in hists])
            ls, c = STYLES.get(method_label, ('-', 'gray'))
            short_name = method_label.split(' ')[0]
            ax.plot(eval_epochs, acc_m, ls, color=c, linewidth=1.2, label=short_name)
            ax.fill_between(eval_epochs, acc_m - acc_s, acc_m + acc_s,
                            color=c, alpha=0.12)
        ax.set_title(sname)
        ax.set_xlabel('Epoch')
        if col == 0:
            ax.set_ylabel('Test accuracy')
            ax.legend(fontsize=8)
    fig.suptitle('Test accuracy: SGD vs Adam vs AdamW (5-seed)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_adam_accuracy.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # Figure 3: Ratio residual comparison (the key plot)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for col, sname in enumerate(['Constant', 'Step', 'Cosine']):
        ax = axes[col]
        for method_label in method_labels:
            hists = all_results[method_label][sname]
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
            ls, c = STYLES.get(method_label, ('-', 'gray'))
            short_name = method_label.split(' ')[0]
            ax.plot(steps, res_m_smooth, ls, color=c, linewidth=1,
                    label=short_name)
        ax.set_title(sname)
        ax.set_xlabel('Step')
        if col == 0:
            ax.set_ylabel('Ratio residual (smoothed)')
            ax.legend(fontsize=8)
        ax.set_yscale('log')
    fig.suptitle('Recurrence ratio residual: SGD (exact) vs Adam/AdamW (approximate)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'fig_adam_ratio_residual.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")

    # Summary JSON
    summary = {}
    for method_label in method_labels:
        summary[method_label] = {}
        for sname in schedules:
            hists = all_results[method_label][sname]
            accs = [max(h['eval_test_acc']) for h in hists]
            ratio_res = [np.median(h['middle conv2_ratio_residual']) for h in hists]
            orth_res = [np.median(h['middle conv2_orth_residual']) for h in hists]
            summary[method_label][sname] = {
                'best_acc_mean': float(np.mean(accs)),
                'best_acc_std': float(np.std(accs)),
                'ratio_residual_median_mean': float(np.mean(ratio_res)),
                'orth_residual_median_mean': float(np.mean(orth_res)),
            }

    with open(os.path.join(args.out_dir, 'adam_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {args.out_dir}/adam_summary.json")
    print("\n=== Adam experiment complete ===")


if __name__ == '__main__':
    main()
