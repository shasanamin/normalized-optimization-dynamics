"""
training.py -- Training loops with schedule support and blockwise diagnostics.

Provides unified training infrastructure for both MLP and ConvNet experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

from .diagnostics import compute_block_metrics, flatten_filter_blocks


def manual_sgd_step(model: nn.Module, lr: float, wd: float) -> None:
    """Manual SGD + coupled weight decay update (no momentum).

    Implements: p <- (1 - lr * wd) * p - lr * grad.
    This is the coupled weight-decay form used throughout the paper.
    """
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.mul_(1.0 - lr * wd).add_(p.grad, alpha=-lr)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             batch_size: int = 512) -> Tuple[float, float]:
    """Evaluate loss and accuracy on a dataset.

    Args:
        model: neural network
        x: input tensor
        y: target labels
        batch_size: evaluation batch size (for memory)

    Returns:
        (loss, accuracy) tuple
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n = x.shape[0]

    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        total_loss += nn.functional.cross_entropy(
            logits, yb, reduction='sum'
        ).item()
        correct += (logits.argmax(dim=1) == yb).sum().item()

    model.train()
    return total_loss / n, correct / n


def sample_batch(x: torch.Tensor, y: torch.Tensor,
                 batch_size: Optional[int],
                 generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a training minibatch (or return full dataset if batch_size is None/0)."""
    n = x.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        return x, y
    idx = torch.randint(n, (batch_size,), generator=generator, device='cpu')
    return x[idx], y[idx]


def run_mlp_schedule(
    model: nn.Module,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_test: torch.Tensor, y_test: torch.Tensor,
    etas: np.ndarray, lams: np.ndarray,
    batch_size: Optional[int] = None,
    seed: int = 7,
    log_every: int = 20,
    schedule_name: str = "",
) -> Dict[str, list]:
    """Train an MLP with blockwise diagnostics on the first linear layer.

    Args:
        model: SimpleNormMLP instance
        x_train, y_train: training data
        x_test, y_test: test data
        etas: learning rate schedule array
        lams: weight decay schedule array
        batch_size: minibatch size (None for full batch)
        seed: random seed
        log_every: print progress interval
        schedule_name: label for logging

    Returns:
        History dict with per-step diagnostics.
    """
    T = len(etas) - 1
    gen = torch.Generator(device='cpu').manual_seed(seed)

    history = {
        'train_loss': [], 'test_loss': [], 'test_acc': [],
        'orth_residual': [], 'pyth_residual': [], 'ratio_residual': [],
        'threshold_accuracy': [], 'expansion_fraction': [], 'B_t': [],
    }

    model.train()
    for t in range(T):
        xb, yb = sample_batch(x_train, y_train, batch_size, gen)

        # Snapshot weights before
        w_before = model.fc1.weight.data.clone()

        # Forward and backward
        model.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        loss.backward()

        g_before = model.fc1.weight.grad.data.clone()

        # SGD + WD step
        manual_sgd_step(model, etas[t], lams[t])

        w_after = model.fc1.weight.data.clone()

        # Blockwise diagnostics
        metrics = compute_block_metrics(
            w_before, g_before, w_after,
            etas[t], lams[t], etas[t + 1], lams[t + 1]
        )

        history['train_loss'].append(loss.item())
        for key in ['orth_residual', 'pyth_residual', 'ratio_residual',
                     'threshold_accuracy', 'expansion_fraction', 'B_t']:
            suffix = '_median' if key in ['orth_residual', 'pyth_residual', 'ratio_residual'] else ''
            history[key].append(metrics.get(key + suffix, metrics.get(key)))

        # Test evaluation
        test_loss, test_acc = evaluate(model, x_test, y_test)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if log_every and (t + 1) % log_every == 0:
            print(f"  [{schedule_name}] step {t+1}/{T}  "
                  f"loss={loss.item():.4f}  acc={test_acc:.3f}  "
                  f"B_t={metrics['B_t']:.4f}  expand={metrics['expansion_fraction']:.3f}")

    return history


def run_conv_schedule(
    model: nn.Module,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_test: torch.Tensor, y_test: torch.Tensor,
    etas: np.ndarray, lams: np.ndarray,
    batch_size: int = 128,
    seed: int = 7,
    log_every: int = 100,
    eval_every: int = 20,
    schedule_name: str = "",
) -> Dict[str, list]:
    """Train a ConvNet with filterwise blockwise diagnostics.

    Instruments all convolutional layers returned by model.conv_layers().

    Returns:
        History dict with per-step per-layer diagnostics + evaluation curves.
    """
    T = len(etas) - 1
    gen = torch.Generator(device='cpu').manual_seed(seed)
    n_train = x_train.shape[0]
    steps_per_epoch = int(np.ceil(n_train / batch_size))

    layers = model.conv_layers()
    layer_names = list(layers.keys())

    history = {
        'train_loss': [], 'B_t': [],
        'eval_t': [], 'eval_epoch': [],
        'eval_train_loss': [], 'eval_train_acc': [],
        'eval_test_loss': [], 'eval_test_acc': [],
    }
    for name in layer_names:
        history[f'{name}_expansion'] = []
        history[f'{name}_threshold_acc'] = []
        history[f'{name}_ratio_residual'] = []

    model.train()
    for t in range(T):
        xb, yb = sample_batch(x_train, y_train, batch_size, gen)

        # Snapshot conv weights before
        snapshots_before = {}
        for name, layer in layers.items():
            snapshots_before[name] = flatten_filter_blocks(
                layer.weight.data.clone()
            )

        # Forward + backward
        model.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        loss.backward()

        grads_before = {}
        for name, layer in layers.items():
            grads_before[name] = flatten_filter_blocks(
                layer.weight.grad.data.clone()
            )

        # SGD + WD step
        manual_sgd_step(model, etas[t], lams[t])

        # Snapshot after
        snapshots_after = {}
        for name, layer in layers.items():
            snapshots_after[name] = flatten_filter_blocks(
                layer.weight.data.clone()
            )

        # Per-layer diagnostics
        history['train_loss'].append(loss.item())
        history['B_t'].append(
            etas[t + 1] / (etas[t] * (1 - etas[t] * lams[t]) * (1 - etas[t + 1] * lams[t + 1]))
        )

        for name in layer_names:
            m = compute_block_metrics(
                snapshots_before[name], grads_before[name],
                snapshots_after[name],
                etas[t], lams[t], etas[t + 1], lams[t + 1]
            )
            history[f'{name}_expansion'].append(m['expansion_fraction'])
            history[f'{name}_threshold_acc'].append(m['threshold_accuracy'])
            history[f'{name}_ratio_residual'].append(m['ratio_residual_median'])

        # Periodic full-dataset evaluation
        if eval_every and (t % eval_every == 0 or t == T - 1):
            tr_loss, tr_acc = evaluate(model, x_train, y_train)
            te_loss, te_acc = evaluate(model, x_test, y_test)
            history['eval_t'].append(t)
            history['eval_epoch'].append(t / steps_per_epoch)
            history['eval_train_loss'].append(tr_loss)
            history['eval_train_acc'].append(tr_acc)
            history['eval_test_loss'].append(te_loss)
            history['eval_test_acc'].append(te_acc)

        if log_every and (t + 1) % log_every == 0:
            print(f"  [{schedule_name}] step {t+1}/{T}  loss={loss.item():.4f}")

    return history
