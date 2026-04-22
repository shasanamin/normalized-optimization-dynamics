"""
diagnostics.py -- Blockwise diagnostics for the paper's exact recurrences.

Provides three families of diagnostics:
- SGD diagnostics in the simplified tangent-gradient form used for Theorem 1.
- Exact SGDM augmented-state diagnostics based on z_t = r_t m_{t+1}.
- Adam-type diagnostics comparing the actual update to the epsilon=0 theorem.
"""

import torch
import numpy as np
from typing import Dict


EPS = 1e-30


def _row_norms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.norm(tensor, dim=1)


def _median_or_nan(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float('nan')
    return values.median().item()


def _masked_mean_or_nan(values: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.any():
        return values[mask].mean().item()
    return float('nan')


def _exponent_scan(
    ratio_actual: torch.Tensor,
    forcing: float,
    denom_root: torch.Tensor,
    exponents: torch.Tensor,
) -> torch.Tensor:
    denom = denom_root.unsqueeze(1).clamp_min(EPS)
    predicted = forcing / torch.pow(denom, exponents.unsqueeze(0))
    residual = torch.abs(ratio_actual.unsqueeze(1) - predicted)
    return residual.median(dim=0).values


def compute_block_metrics(
    w_before: torch.Tensor,
    g_before: torch.Tensor,
    w_after: torch.Tensor,
    eta_t: float,
    lam_t: float,
    eta_next: float,
    lam_next: float,
) -> Dict[str, float]:
    """Compute per-block diagnostics for scale-invariant verification.

    Each row of w_before is treated as an independent scale-invariant block.

    Args:
        w_before: weight matrix before the update, shape (num_blocks, block_dim)
        g_before: gradient of the loss w.r.t. w, same shape
        w_after: weight matrix after the SGD+WD update
        eta_t: learning rate at step t
        lam_t: weight decay at step t
        eta_next: learning rate at step t+1
        lam_next: weight decay at step t+1

    Returns:
        Dict with scalar diagnostics (median over blocks).
    """
    a_t = 1.0 - eta_t * lam_t
    a_next = 1.0 - eta_next * lam_next
    B_t = eta_next / (eta_t * a_t * a_next)
    threshold = np.sqrt(max(B_t - 1.0, 0.0))

    # Per-block norms
    r_before = torch.norm(w_before, dim=1)  # (num_blocks,)
    r_after = torch.norm(w_after, dim=1)

    # Orthogonality: |<w, g>| / (||w|| * ||g||)
    wg_dot = (w_before * g_before).sum(dim=1)
    g_norm = torch.norm(g_before, dim=1)
    orth_residual = torch.abs(wg_dot) / (r_before * g_norm + EPS)

    # Perpendicular gradient component
    u_before = w_before / r_before.unsqueeze(1)
    g_par = wg_dot / (r_before + EPS)
    g_perp = g_before - g_par.unsqueeze(1) * u_before
    g_perp_norm = torch.norm(g_perp, dim=1)

    # gbar = r * g_perp (the scale-free tangent gradient norm)
    gbar_norm = r_before * g_perp_norm

    # Pythagorean residual
    alpha_t = eta_t / (r_before ** 2)
    predicted_r2 = r_before ** 2 * (a_t ** 2 + alpha_t ** 2 * gbar_norm ** 2)
    pyth_residual = torch.abs(r_after ** 2 - predicted_r2) / (r_after ** 2 + EPS)

    # Effective directional stepsize
    Phi_before = eta_t / (a_t * r_before ** 2)
    Phi_after = eta_next / (a_next * r_after ** 2)

    # R_t = Phi_t * ||gbar_t||
    R_t = Phi_before * gbar_norm

    # Ratio residual: |Phi_{t+1}/Phi_t - B_t/(1+R_t^2)|
    ratio_actual = Phi_after / (Phi_before + EPS)
    ratio_predicted = B_t / (1.0 + R_t ** 2)
    ratio_residual = torch.abs(ratio_actual - ratio_predicted)

    # Threshold prediction accuracy
    expanding = (Phi_after > Phi_before).float()
    if B_t <= 1.0:
        # Theory predicts contraction always
        threshold_correct = (1.0 - expanding)
    else:
        # Theory predicts contraction iff R_t >= threshold
        predicted_contract = (R_t >= threshold).float()
        actual_contract = 1.0 - expanding
        threshold_correct = (predicted_contract == actual_contract).float()

    expansion_frac = expanding.mean().item()

    return {
        'orth_residual_median': orth_residual.median().item(),
        'pyth_residual_median': pyth_residual.median().item(),
        'ratio_residual_median': ratio_residual.median().item(),
        'threshold_accuracy': threshold_correct.mean().item(),
        'expansion_fraction': expansion_frac,
        'B_t': B_t,
        'Phi_median': Phi_before.median().item(),
        'R_median': R_t.median().item(),
    }


def flatten_filter_blocks(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape a 4D convolutional weight to 2D for blockwise diagnostics.

    Each output channel becomes one row (one scale-invariant block).

    Args:
        tensor: shape (out_channels, in_channels, kH, kW)

    Returns:
        tensor of shape (out_channels, in_channels * kH * kW)
    """
    return tensor.view(tensor.shape[0], -1)


def compute_sgdm_block_metrics(
    w_before: torch.Tensor,
    momentum_after: torch.Tensor,
    w_after: torch.Tensor,
    eta_t: float,
    lam_t: float,
    eta_next: float,
    lam_next: float,
    exponents: torch.Tensor = None,
) -> Dict[str, float]:
    """Compute exact SGDM augmented-state diagnostics per tracked block.

    Args:
        w_before: tracked block weights at step t.
        momentum_after: m_{t+1} after the momentum update, before the weight update.
        w_after: tracked block weights at step t+1.
    """
    a_t = 1.0 - eta_t * lam_t
    a_next = 1.0 - eta_next * lam_next
    B_t = eta_next / (eta_t * a_t * a_next)

    r_before = _row_norms(w_before)
    r_after = _row_norms(w_after)
    u_before = w_before / r_before.unsqueeze(1)

    phi_before = eta_t / (a_t * r_before ** 2)
    phi_after = eta_next / (a_next * r_after ** 2)

    z_t = r_before.unsqueeze(1) * momentum_after
    diff = u_before - phi_before.unsqueeze(1) * z_t
    denom_root = _row_norms(diff)
    denom_sq = denom_root ** 2

    ratio_actual = phi_after / (phi_before + EPS)
    ratio_predicted = B_t / denom_sq.clamp_min(EPS)
    ratio_residual = torch.abs(ratio_actual - ratio_predicted)

    radius_residual = torch.abs(r_after - a_t * r_before * denom_root) / (r_after + EPS)

    c_t = (u_before * z_t).sum(dim=1)
    s_t = z_t - c_t.unsqueeze(1) * u_before
    s_norm = _row_norms(s_t)
    denom_split = (1.0 - phi_before * c_t) ** 2 + (phi_before ** 2) * (s_norm ** 2)
    split_residual = torch.abs(denom_sq - denom_split)

    expanding = (phi_after > phi_before).float()
    predicted_contract = (denom_sq >= B_t).float()
    actual_contract = 1.0 - expanding
    threshold_correct = (predicted_contract == actual_contract).float()

    c_pos = c_t > 0.0
    pump_term = -2.0 * phi_before * c_t
    tangent_term = (phi_before ** 2) * (s_norm ** 2)

    result = {
        'B_t': B_t,
        'Phi_median': phi_before.median().item(),
        'ratio_residual_median': ratio_residual.median().item(),
        'radius_residual_median': radius_residual.median().item(),
        'split_residual_median': split_residual.median().item(),
        'threshold_accuracy': threshold_correct.mean().item(),
        'expansion_fraction': expanding.mean().item(),
        'c_median': c_t.median().item(),
        'abs_c_median': c_t.abs().median().item(),
        's_norm_median': s_norm.median().item(),
        'denom_median': denom_sq.median().item(),
        'pump_term_median': pump_term.median().item(),
        'tangent_term_median': tangent_term.median().item(),
        'c_positive_fraction': c_pos.float().mean().item(),
        'expansion_if_c_positive': _masked_mean_or_nan(expanding, c_pos),
        'expansion_if_c_nonpositive': _masked_mean_or_nan(expanding, ~c_pos),
    }

    if exponents is not None:
        scan = _exponent_scan(ratio_actual, B_t, denom_root, exponents)
        result['exponent_scan'] = scan.detach().cpu().tolist()

    return result


def compute_adam_block_metrics(
    w_before: torch.Tensor,
    p_zero_eps: torch.Tensor,
    p_actual: torch.Tensor,
    w_after: torch.Tensor,
    eta_t: float,
    lam_t: float,
    eta_next: float,
    lam_next: float,
    exponents: torch.Tensor = None,
) -> Dict[str, float]:
    """Compute Adam-type diagnostics relative to the epsilon=0 theorem.

    The exact epsilon=0 theorem uses Psi_t = eta_t / (a_t r_t) and the
    scale-free direction p_zero_eps. For epsilon > 0 we compare the actual
    update to that reference law, while also checking the trivial exact algebra
    using the actual update direction p_actual.
    """
    a_t = 1.0 - eta_t * lam_t
    a_next = 1.0 - eta_next * lam_next
    b_tilde = (eta_next / eta_t) / a_next

    r_before = _row_norms(w_before)
    r_after = _row_norms(w_after)
    u_before = w_before / r_before.unsqueeze(1)

    psi_before = eta_t / (a_t * r_before)
    psi_after = eta_next / (a_next * r_after)
    ratio_actual = psi_after / (psi_before + EPS)

    denom_zero = _row_norms(u_before - psi_before.unsqueeze(1) * p_zero_eps)
    denom_actual = _row_norms(u_before - psi_before.unsqueeze(1) * p_actual)

    ratio_pred_zero = b_tilde / denom_zero.clamp_min(EPS)
    ratio_pred_actual = b_tilde / denom_actual.clamp_min(EPS)

    zero_residual = torch.abs(ratio_actual - ratio_pred_zero)
    actual_residual = torch.abs(ratio_actual - ratio_pred_actual)
    radius_residual = torch.abs(r_after - a_t * r_before * denom_actual) / (r_after + EPS)

    c_t = (u_before * p_zero_eps).sum(dim=1)
    s_t = p_zero_eps - c_t.unsqueeze(1) * u_before
    s_norm = _row_norms(s_t)
    p_gap = _row_norms(p_actual - p_zero_eps) / (_row_norms(p_zero_eps) + EPS)

    expanding = (psi_after > psi_before).float()
    predicted_contract = (denom_zero >= b_tilde).float()
    actual_contract = 1.0 - expanding
    threshold_correct = (predicted_contract == actual_contract).float()

    result = {
        'B_tilde': b_tilde,
        'Psi_median': psi_before.median().item(),
        'eps0_residual_median': zero_residual.median().item(),
        'actual_direction_residual_median': actual_residual.median().item(),
        'radius_residual_median': radius_residual.median().item(),
        'expansion_fraction': expanding.mean().item(),
        'threshold_accuracy': threshold_correct.mean().item(),
        'c_median': c_t.median().item(),
        'abs_c_median': c_t.abs().median().item(),
        's_norm_median': s_norm.median().item(),
        'denom_zero_median': denom_zero.median().item(),
        'denom_actual_median': denom_actual.median().item(),
        'scale_break_gap_median': p_gap.median().item(),
    }

    if exponents is not None:
        scan = _exponent_scan(ratio_actual, b_tilde, denom_zero, exponents)
        result['exponent_scan'] = scan.detach().cpu().tolist()

    return result
