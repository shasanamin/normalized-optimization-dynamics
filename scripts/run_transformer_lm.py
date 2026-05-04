#!/usr/bin/env python
"""Run Transformer / LayerNorm language-model diagnostics.

This is the cleaned release entry point for the collaborator-supplied GPT-2
experiments. It avoids cluster-specific paths and logging services, and writes
histories in the same shape consumed by ``replot_transformer_lm.py``.

Example:
    python scripts/run_transformer_lm.py \\
      --dataset wikitext --model small_gpt2 \\
      --data_root /scratch/$USER/wikitext_gpt2 \\
      --out_dir results/transformer_lm/wikitext_small_gpt2
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{os.environ.get('USER', 'user')}"),
)

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataloading import load_openweb, load_wikitext
from src.diagnostics import flatten_filter_blocks
from src.models import gpt2_noaffine, small_gpt2_noaffine
from src.theory import schedule_constant, schedule_cosine, schedule_step


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def default_lrs(dataset: str) -> Tuple[float, float]:
    """Learning-rate defaults matching the supplied reference histories."""
    if dataset == "openweb":
        return 2.5e-4, 1.0e-5
    return 5.0e-5, 5.0e-6


def manual_sgd_step(model: nn.Module, lr: float, wd: float) -> None:
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.mul_(1.0 - lr * wd).add_(p.grad, alpha=-lr)


def manual_sgdm_step(
    model: nn.Module,
    lr: float,
    wd: float,
    momentum: float,
    velocity: Dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if name not in velocity:
                velocity[name] = torch.zeros_like(p)
            v = velocity[name]
            v.mul_(momentum).add_(p.grad)
            p.mul_(1.0 - lr * wd).add_(v, alpha=-lr)


def manual_adam_coupled_step(
    model: nn.Module,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
    eps: float,
    state: Dict[str, dict],
    step_count: int,
) -> None:
    """Adam with coupled weight decay: p <- (1-lr*wd)p - lr*m/sqrt(v)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if name not in state:
                state[name] = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            s = state[name]
            s["m"].mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
            s["v"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
            m_hat = s["m"] / (1.0 - beta1 ** step_count)
            v_hat = s["v"] / (1.0 - beta2 ** step_count)
            p.mul_(1.0 - lr * wd).add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


def manual_adamw_step(
    model: nn.Module,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
    eps: float,
    state: Dict[str, dict],
    step_count: int,
) -> None:
    """AdamW-style decoupled decay matching the reference diagnostic."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if name not in state:
                state[name] = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            s = state[name]
            s["m"].mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
            s["v"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
            m_hat = s["m"] / (1.0 - beta1 ** step_count)
            v_hat = s["v"] / (1.0 - beta2 ** step_count)
            p.mul_(1.0 - wd).add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


def compute_block_metrics_general(
    w_before: torch.Tensor,
    g_before: torch.Tensor,
    w_after: torch.Tensor,
    eta_t: float,
    lam_t: float,
    eta_next: float,
    lam_next: float,
    decoupled: bool = False,
) -> dict:
    if decoupled:
        a_t = 1.0 - lam_t
        a_next = 1.0 - lam_next
    else:
        a_t = 1.0 - eta_t * lam_t
        a_next = 1.0 - eta_next * lam_next

    b_t = eta_next / (eta_t * a_t * a_next)
    threshold = np.sqrt(max(b_t - 1.0, 0.0))

    r_before = torch.norm(w_before, dim=1)
    r_after = torch.norm(w_after, dim=1)
    phi_before = eta_t / (a_t * r_before ** 2)
    phi_after = eta_next / (a_next * r_after ** 2)

    wg_dot = (w_before * g_before).sum(dim=1)
    g_norm = torch.norm(g_before, dim=1)
    u_before = w_before / r_before.unsqueeze(1)
    g_par = wg_dot / (r_before + 1e-30)
    g_perp = g_before - g_par.unsqueeze(1) * u_before
    gbar_norm = r_before * torch.norm(g_perp, dim=1)
    r_t = phi_before * gbar_norm

    ratio_actual = phi_after / (phi_before + 1e-30)
    ratio_predicted = b_t / (1.0 + r_t ** 2)
    ratio_residual = torch.abs(ratio_actual - ratio_predicted)

    expanding = (phi_after > phi_before).float()
    if b_t <= 1.0:
        threshold_correct = 1.0 - expanding
    else:
        predicted_contract = (r_t >= threshold).float()
        threshold_correct = (predicted_contract == (1.0 - expanding)).float()

    orth_residual = torch.abs(wg_dot) / (r_before * g_norm + 1e-30)
    return {
        "B_t": b_t,
        "Phi_median": phi_before.median().item(),
        "ratio_residual_median": ratio_residual.median().item(),
        "expansion_fraction": expanding.mean().item(),
        "threshold_accuracy": threshold_correct.mean().item(),
        "orth_residual_median": orth_residual.median().item(),
    }


@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    loader,
    device: str,
    max_tokens: int = 5000,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    n_tokens = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb, yb)
        logits = out["logits"]
        token_count = yb.numel()
        total_loss += out["loss"].item() * token_count
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        n_tokens += token_count
        if max_tokens and n_tokens >= max_tokens:
            break
    if was_training:
        model.train()
    return total_loss / max(n_tokens, 1), correct / max(n_tokens, 1)


def next_from_iterator(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def run_lm_schedule(
    model: nn.Module,
    train_loader,
    val_loader,
    train_size: int,
    etas: np.ndarray,
    lams: np.ndarray,
    optimizer: str,
    device: str,
    batch_size: int,
    seed: int,
    momentum: float,
    beta1: float,
    beta2: float,
    eps: float,
    eval_every: int,
    max_eval_tokens: int,
    log_every: int,
    schedule_name: str,
) -> dict:
    del seed
    steps_per_epoch = int(np.ceil(train_size / batch_size))
    layers = model.tracked_layers()
    layer_names = list(layers.keys())
    decoupled = optimizer == "adamw"

    history = {
        "train_loss": [],
        "B_t": [],
        "eval_t": [],
        "eval_epoch": [],
        "eval_train_loss": [],
        "eval_train_acc": [],
        "eval_test_loss": [],
        "eval_test_acc": [],
    }
    for name in layer_names:
        history[f"{name}_expansion"] = []
        history[f"{name}_ratio_residual"] = []
        history[f"{name}_threshold_acc"] = []
        history[f"{name}_orth_residual"] = []

    velocity = {}
    adam_state = {}
    iterator = iter(train_loader)
    model.train()

    for t in range(len(etas) - 1):
        (xb, yb), iterator = next_from_iterator(train_loader, iterator)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        snaps_before = {
            name: flatten_filter_blocks(layer.weight.data.clone())
            for name, layer in layers.items()
        }

        model.zero_grad()
        loss = model(xb, yb)["loss"]
        loss.backward()

        grads = {
            name: flatten_filter_blocks(layer.weight.grad.data.clone())
            for name, layer in layers.items()
        }

        if optimizer == "sgd":
            manual_sgd_step(model, etas[t], lams[t])
        elif optimizer == "sgdm":
            manual_sgdm_step(model, etas[t], lams[t], momentum, velocity)
        elif optimizer == "adam":
            manual_adam_coupled_step(
                model, etas[t], lams[t], beta1, beta2, eps, adam_state, t + 1
            )
        elif optimizer == "adamw":
            manual_adamw_step(
                model, etas[t], lams[t], beta1, beta2, eps, adam_state, t + 1
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        snaps_after = {
            name: flatten_filter_blocks(layer.weight.data.clone())
            for name, layer in layers.items()
        }

        if decoupled:
            b_t = etas[t + 1] / (etas[t] * (1.0 - lams[t]) * (1.0 - lams[t + 1]))
        else:
            b_t = etas[t + 1] / (
                etas[t]
                * (1.0 - etas[t] * lams[t])
                * (1.0 - etas[t + 1] * lams[t + 1])
            )

        history["train_loss"].append(loss.item())
        history["B_t"].append(b_t)

        for name in layer_names:
            metrics = compute_block_metrics_general(
                snaps_before[name],
                grads[name],
                snaps_after[name],
                etas[t],
                lams[t],
                etas[t + 1],
                lams[t + 1],
                decoupled=decoupled,
            )
            history[f"{name}_expansion"].append(metrics["expansion_fraction"])
            history[f"{name}_ratio_residual"].append(metrics["ratio_residual_median"])
            history[f"{name}_threshold_acc"].append(metrics["threshold_accuracy"])
            history[f"{name}_orth_residual"].append(metrics["orth_residual_median"])

        if eval_every and (t % eval_every == 0 or t == len(etas) - 2):
            tr_loss, tr_acc = evaluate_lm(model, train_loader, device, max_eval_tokens)
            te_loss, te_acc = evaluate_lm(model, val_loader, device, max_eval_tokens)
            history["eval_t"].append(t)
            history["eval_epoch"].append(t / steps_per_epoch)
            history["eval_train_loss"].append(tr_loss)
            history["eval_train_acc"].append(tr_acc)
            history["eval_test_loss"].append(te_loss)
            history["eval_test_acc"].append(te_acc)

        if log_every and (t + 1) % log_every == 0:
            acc = history["eval_test_acc"][-1] if history["eval_test_acc"] else float("nan")
            print(
                f"  [{schedule_name}] step {t + 1}/{len(etas) - 1} "
                f"loss={loss.item():.4f} eval_acc={acc:.3f}"
            )

    return history


def make_model(model_name: str, vocab_size: int, block_size: int, device: str) -> nn.Module:
    if model_name in {"small_gpt2", "tiny_gpt2", "tinygpt2"}:
        return small_gpt2_noaffine(vocab_size, max_seq_len=block_size).to(device)
    if model_name == "gpt2":
        return gpt2_noaffine(vocab_size, max_seq_len=block_size).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def serialize_history(data):
    if isinstance(data, dict):
        return {k: serialize_history(v) for k, v in data.items()}
    if isinstance(data, list):
        return [serialize_history(v) for v in data]
    if isinstance(data, (float, int, np.floating, np.integer)):
        return float(data)
    return data


def optimizer_group_name(optimizer: str, momentum: float) -> str:
    if optimizer == "sgd":
        return "SGD"
    if optimizer == "sgdm":
        return f"SGDM ($\\mu$={momentum:g})"
    if optimizer == "adam":
        return "Adam (coupled WD)"
    if optimizer == "adamw":
        return "AdamW (decoupled WD)"
    raise ValueError(optimizer)


def build_schedules(args):
    lr_high = args.lr_high
    lr_low = args.lr_low
    if lr_high is None or lr_low is None:
        default_high, default_low = default_lrs(args.dataset)
        lr_high = default_high if lr_high is None else lr_high
        lr_low = default_low if lr_low is None else lr_low
    return {
        "Constant": schedule_constant(lr_high, args.weight_decay, args.total_steps),
        "Step": schedule_step(
            lr_high, lr_low, args.weight_decay, args.total_steps, args.total_steps // 2
        ),
        "Cosine": schedule_cosine(lr_high, lr_low, args.weight_decay, args.total_steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wikitext", "openweb"], required=True)
    parser.add_argument(
        "--model",
        choices=["small_gpt2", "tiny_gpt2", "tinygpt2", "gpt2"],
        required=True,
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--lr_high", type=float, default=None)
    parser.add_argument("--lr_low", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["sgd", "sgdm", "adam", "adamw"],
        default=["sgd", "sgdm", "adam", "adamw"],
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=["Constant", "Step", "Cosine"],
        default=["Constant", "Step", "Cosine"],
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--max_eval_tokens", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=2000)
    parser.add_argument("--no_prepare_data", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    load_fn = load_wikitext if args.dataset == "wikitext" else load_openweb
    tokenizer, train_loader, val_loader, train_dataset, _ = load_fn(
        args.data_root,
        args.batch_size,
        block_size=args.block_size,
        num_workers=args.num_workers,
        prepare_if_missing=not args.no_prepare_data,
    )
    schedules = build_schedules(args)
    eval_every = args.eval_every
    if eval_every is None:
        eval_every = max(1, int(np.ceil(len(train_dataset) / args.batch_size)))

    for optimizer in args.optimizers:
        grouped = {}
        for schedule_name in args.schedules:
            grouped[schedule_name] = []
            etas, lams = schedules[schedule_name]
            for seed_idx in range(args.n_seeds):
                actual_seed = args.seed_base + seed_idx
                print(f"\n{optimizer} | {schedule_name} | seed={actual_seed}")
                torch.manual_seed(actual_seed)
                model = make_model(
                    args.model, tokenizer.vocab_size, args.block_size, args.device
                )
                hist = run_lm_schedule(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_size=len(train_dataset),
                    etas=etas,
                    lams=lams,
                    optimizer=optimizer,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=actual_seed,
                    momentum=args.momentum,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    eps=args.eps,
                    eval_every=eval_every,
                    max_eval_tokens=args.max_eval_tokens,
                    log_every=args.log_every,
                    schedule_name=f"{optimizer}_{schedule_name}_s{actual_seed}",
                )
                grouped[schedule_name].append(hist)

        group_name = optimizer_group_name(optimizer, args.momentum)
        if optimizer == "sgd":
            out_path = os.path.join(args.out_dir, "multiseed", "conv_histories.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(serialize_history(grouped), f)
        elif optimizer == "sgdm":
            out_path = os.path.join(args.out_dir, "sgdm", "sgdm_histories.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(serialize_history({group_name: grouped}), f)
        else:
            existing = {}
            out_path = os.path.join(args.out_dir, "adam", "adam_histories.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.exists(out_path):
                with open(out_path) as f:
                    existing = json.load(f)
            existing[group_name] = serialize_history(grouped)
            with open(out_path, "w") as f:
                json.dump(existing, f)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
