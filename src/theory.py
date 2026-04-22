"""
theory.py -- Exact theory for scale-invariant optimization.

Implements the core mathematics from the paper:
- Schedule factor B_t computation
- Exact 2D isotropic map (Theorem 4)
- Fixed-point analysis (Proposition 5)
- Jacobian and spiral-source verification (Proposition 7)
- Problem construction for the normalized-linear model
"""

import numpy as np
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------

def schedule_constant(eta: float, lam: float, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Constant learning rate and weight decay for T+1 steps."""
    return np.full(T + 1, eta), np.full(T + 1, lam)


def schedule_step(eta_hi: float, eta_lo: float, lam: float, T: int,
                  drop_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Step decay: eta_hi before drop_t, eta_lo from drop_t onward."""
    etas = np.where(np.arange(T + 1) < drop_t, eta_hi, eta_lo)
    return etas, np.full(T + 1, lam)


def schedule_cosine(eta_max: float, eta_min: float, lam: float,
                    T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine annealing from eta_max to eta_min over T steps."""
    ts = np.arange(T + 1)
    etas = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * ts / T))
    return etas, np.full(T + 1, lam)


def schedule_exponential(eta0: float, rho: float, lam: float,
                         T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exponential decay eta_t = eta0 * rho^t for T+1 steps."""
    ts = np.arange(T + 1)
    etas = eta0 * (rho ** ts)
    return etas, np.full(T + 1, lam)


def schedule_target_constant_b(eta0: float, lam: float, b_target: float,
                               total_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate LR schedule enforcing constant B_t = b_target.

    Uses the recursion:
        c_t = b_target * eta_t * (1 - lam * eta_t)
        eta_{t+1} = c_t / (1 + lam * c_t)
    """
    etas = np.zeros(total_steps + 1)
    etas[0] = eta0
    for t in range(total_steps):
        c = b_target * etas[t] * (1.0 - lam * etas[t])
        if c <= 0 or 1.0 + lam * c <= 0:
            etas[t + 1:] = etas[t]
            break
        etas[t + 1] = c / (1.0 + lam * c)
    return etas, np.full(total_steps + 1, lam)


def schedule_target_b_sequence(eta0: float, lam: float, b_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an LR schedule enforcing a prescribed sequence of B_t values.

    Given desired schedule factors b_sequence[t] = B_t and constant weight decay
    lam, solve the implicit relation

        eta_{t+1} = B_t eta_t (1 - lam * eta_t) (1 - lam * eta_{t+1})

    for eta_{t+1} at each step.
    """
    b_sequence = np.asarray(b_sequence, dtype=float)
    total_steps = len(b_sequence)
    etas = np.zeros(total_steps + 1)
    etas[0] = eta0
    for t in range(total_steps):
        c = b_sequence[t] * etas[t] * (1.0 - lam * etas[t])
        if c <= 0 or 1.0 + lam * c <= 0:
            etas[t + 1:] = etas[t]
            break
        etas[t + 1] = c / (1.0 + lam * c)
    return etas, np.full(total_steps + 1, lam)


def decoupled_lambda_for_constant_b_horizon(
    eta0: float,
    eta_final: float,
    b_target: float,
    total_steps: int,
) -> float:
    """Solve for constant decoupled shrinkage matching a target-B horizon.

    For the simple decoupled-decay analogue with

        eta_{t+1} = B_t eta_t (1 - lam)^2,

    a constant target B_t = b_target implies

        eta_T = eta_0 [b_target (1 - lam)^2]^T.

    This helper chooses lam so that the resulting schedule lands at eta_final
    after total_steps updates.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if eta0 <= 0.0 or eta_final <= 0.0:
        raise ValueError("eta0 and eta_final must both be positive")
    if b_target <= 0.0:
        raise ValueError("b_target must be positive")

    growth_per_step = (eta_final / eta0) ** (1.0 / total_steps)
    shrink_sq = growth_per_step / b_target
    if shrink_sq <= 0.0 or shrink_sq >= 1.0:
        raise ValueError(
            "Requested horizon does not yield a valid constant decoupled shrinkage; "
            "need 0 < (eta_final / eta0)^(1/T) / b_target < 1."
        )
    return 1.0 - np.sqrt(shrink_sq)


def schedule_target_constant_b_decoupled(
    eta0: float,
    lam: float,
    b_target: float,
    total_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an LR schedule enforcing constant B_t for decoupled shrinkage.

    For the simple decoupled-decay analogue with a_t = 1 - lam, the schedule
    factor is

        B_t = eta_{t+1} / (eta_t (1 - lam)^2).

    Holding lam fixed therefore gives the closed-form recursion

        eta_{t+1} = B_t eta_t (1 - lam)^2.
    """
    if not (0.0 <= lam < 1.0):
        raise ValueError("lam must satisfy 0 <= lam < 1")
    if b_target <= 0.0:
        raise ValueError("b_target must be positive")

    etas = np.zeros(total_steps + 1)
    etas[0] = eta0
    multiplier = b_target * (1.0 - lam) ** 2
    for t in range(total_steps):
        etas[t + 1] = etas[t] * multiplier
    return etas, np.full(total_steps + 1, lam)


def schedule_target_constant_b_decoupled_horizon(
    eta0: float,
    eta_final: float,
    b_target: float,
    total_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decoupled constant-B schedule with lam chosen to match a target horizon."""
    lam = decoupled_lambda_for_constant_b_horizon(
        eta0=eta0,
        eta_final=eta_final,
        b_target=b_target,
        total_steps=total_steps,
    )
    return schedule_target_constant_b_decoupled(
        eta0=eta0,
        lam=lam,
        b_target=b_target,
        total_steps=total_steps,
    )


def schedule_cosine_then_hold(eta_max: float, eta_min: float, lam: float,
                              total_steps: int,
                              decay_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine decay for first decay_steps, then hold at eta_min."""
    etas = np.full(total_steps + 1, eta_min)
    ts = np.arange(decay_steps + 1)
    etas[:decay_steps + 1] = (
        eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * ts / decay_steps))
    )
    return etas, np.full(total_steps + 1, lam)


# ---------------------------------------------------------------------------
# Schedule factor B_t
# ---------------------------------------------------------------------------

def schedule_controls(etas: np.ndarray,
                      lams: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute decay factors a_t and schedule factors B_t.

    Returns:
        a: array of a_t = 1 - eta_t * lam_t
        B: array of B_t = eta_{t+1} / (eta_t * a_t * a_{t+1})
    """
    a = 1.0 - etas * lams
    B = etas[1:] / (etas[:-1] * a[:-1] * a[1:])
    return a, B


# ---------------------------------------------------------------------------
# Exact 2D isotropic map
# ---------------------------------------------------------------------------

def isotropic_map_step(q: float, Phi: float, a: float) -> Tuple[float, float]:
    """One step of the exact 2D isotropic map (Theorem 4).

    Args:
        q: alignment with target (inner product)
        Phi: effective directional stepsize
        a: decay factor (1 - eta*lambda)

    Returns:
        q_next, Phi_next
    """
    s_sq = 1.0 - q * q  # = ||gbar||^2
    R = Phi * np.sqrt(s_sq)
    denom_sq = 1.0 + R * R
    denom = np.sqrt(denom_sq)
    q_next = (q + Phi * s_sq) / denom
    Phi_next = Phi / (a * a * denom_sq)
    return q_next, Phi_next


def isotropic_map_step_scheduled(q: float, Phi: float,
                                 B_t: float) -> Tuple[float, float]:
    """Scheduled variant of the exact 2D isotropic map.

    Uses B_t directly instead of a^2, since the schedule factor
    already encodes the learning rate ratio and decay factors.
    """
    s_sq = 1.0 - q * q
    R = Phi * np.sqrt(s_sq)
    denom_sq = 1.0 + R * R
    denom = np.sqrt(denom_sq)
    q_next = (q + Phi * s_sq) / denom
    Phi_next = B_t * Phi / denom_sq
    return q_next, Phi_next


def run_2d_map_scheduled(q0: float, Phi0: float, etas: np.ndarray,
                         lams: np.ndarray) -> Dict[str, np.ndarray]:
    """Run the exact scheduled 2D isotropic map.

    Returns dict with time series: q, Phi, F, R, B, delta_log_phi_{lhs,rhs}.
    """
    a, B = schedule_controls(etas, lams)
    T = len(B)

    q_arr = np.zeros(T + 1)
    Phi_arr = np.zeros(T + 1)
    q_arr[0], Phi_arr[0] = q0, Phi0

    for t in range(T):
        q_arr[t + 1], Phi_arr[t + 1] = isotropic_map_step_scheduled(
            q_arr[t], Phi_arr[t], B[t]
        )

    F = 1.0 - q_arr
    s_sq = 1.0 - q_arr ** 2
    R = Phi_arr * np.sqrt(s_sq)

    # Verify additive log identity
    delta_log_lhs = np.diff(np.log(Phi_arr))
    delta_log_rhs = np.log(B) - np.log(1.0 + R[:-1] ** 2)

    return {
        'q': q_arr, 'Phi': Phi_arr, 'F': F, 'R': R,
        'B': B, 'a': a, 'etas': etas, 'lams': lams,
        'delta_log_phi_lhs': delta_log_lhs,
        'delta_log_phi_rhs': delta_log_rhs,
    }


# ---------------------------------------------------------------------------
# Fixed-point and Jacobian analysis
# ---------------------------------------------------------------------------

def fixed_point_values(a: float) -> Dict[str, float]:
    """Exact fixed point of the 2D isotropic map (Proposition 5).

    Returns: q_star, s_star, Phi_star, g_star, and the switching surface product.
    """
    q_star = np.sqrt((1.0 + a) / 2.0)
    s_star = np.sqrt((1.0 - a) / 2.0)
    Phi_star = np.sqrt(2.0 * (1.0 + a)) / a
    g_star = s_star  # ||gbar|| = sqrt(1 - q^2)
    gPhi_star = g_star * Phi_star  # should equal sqrt(a^{-2} - 1)
    return {
        'q_star': q_star, 's_star': s_star, 'Phi_star': Phi_star,
        'g_star': g_star, 'gPhi_star': gPhi_star,
        'switching_surface': np.sqrt(1.0 / (a * a) - 1.0),
    }


def fixed_point_jacobian(a: float) -> np.ndarray:
    """Exact 2x2 Jacobian at the fixed point (Proposition 6)."""
    J = np.array([
        [a**2 + a - 1.0, a**2 * (a - 1.0) / 2.0],
        [4.0 * a + 8.0 + 4.0 / a, 2.0 * a**2 - 1.0],
    ])
    return J


def trace_det_disc(a: float) -> Tuple[float, float, float]:
    """Trace, determinant, and discriminant of the Jacobian.

    Returns:
        tr = 3a^2 + a - 2
        det = 1 + a - a^2
        disc = a(a-1)(9a^2 + 15a + 8)  [negative for a in (0,1)]
    """
    tr = 3.0 * a**2 + a - 2.0
    det = 1.0 + a - a**2
    disc = a * (a - 1.0) * (9.0 * a**2 + 15.0 * a + 8.0)
    return tr, det, disc


def eigenvalue_modulus(a: float) -> float:
    """Eigenvalue modulus at the fixed point: sqrt(det) = sqrt(1+a-a^2)."""
    return np.sqrt(1.0 + a - a**2)


# ---------------------------------------------------------------------------
# Problem construction for the normalized-linear model
# ---------------------------------------------------------------------------

def make_anisotropic_problem(n: int = 200, d: int = 5,
                             seed: int = 42) -> Dict:
    """Centered random design with controllable condition number."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d))
    A -= A.mean(axis=0, keepdims=True)
    Sigma = A.T @ A / n
    eigvals = np.linalg.eigvalsh(Sigma)
    kappa = eigvals[-1] / eigvals[0]

    # Realizable normalized target
    coeff = rng.standard_normal(d)
    y = A @ coeff
    y *= np.sqrt(n) / np.linalg.norm(y)

    return {'A': A, 'Sigma': Sigma, 'y': y, 'kappa': kappa}


def make_isotropic_problem(n: int = 31, d: int = 30,
                           seed: int = 42) -> Dict:
    """Centered QR-based construction ensuring Sigma = I_d."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, d))
    raw -= raw.mean(axis=0, keepdims=True)
    Q, R_mat = np.linalg.qr(raw, mode='reduced')
    A = Q * np.sqrt(n)  # ensures A^T A / n = I_d
    Sigma = A.T @ A / n

    beta = rng.standard_normal(d)
    beta /= np.linalg.norm(beta)
    y = A @ beta

    return {'A': A, 'Sigma': Sigma, 'y': y, 'beta': beta}


# ---------------------------------------------------------------------------
# Full d-dimensional simulation
# ---------------------------------------------------------------------------

def tangent_gradient(u: np.ndarray, A: np.ndarray, Sigma: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
    """Scale-free tangent gradient gbar(u)."""
    n = A.shape[0]
    sig = np.sqrt(u @ Sigma @ u)
    c = A.T @ y / n  # = Sigma @ beta
    q = (u @ c) / sig
    return -c / sig + q * (Sigma @ u) / sig**2


def sphere_loss(u: np.ndarray, A: np.ndarray, Sigma: np.ndarray,
                y: np.ndarray) -> float:
    """Sphere loss F(u) = (1/2n)||Au/sigma(u) - y||^2."""
    n = A.shape[0]
    sig = np.sqrt(u @ Sigma @ u)
    z_hat = A @ u / sig
    return 0.5 * np.sum((z_hat - y)**2) / n


def run_full_simulation(A: np.ndarray, Sigma: np.ndarray, y: np.ndarray,
                        etas: np.ndarray, lams: np.ndarray,
                        w0: np.ndarray) -> Dict[str, np.ndarray]:
    """Full d-dimensional GD+WD simulation on the normalized-linear model.

    Returns dict with complete time series for verification.
    """
    T = len(etas) - 1
    w = w0.copy()

    results = {k: [] for k in ['F', 'q', 'Phi', 'g', 'r', 'R', 'B']}
    a_arr, B_arr = schedule_controls(etas, lams)

    for t in range(T + 1):
        r = np.linalg.norm(w)
        u = w / r
        F = sphere_loss(u, A, Sigma, y)
        gb = tangent_gradient(u, A, Sigma, y)
        g = np.linalg.norm(gb)
        a_t = a_arr[t]
        Phi = etas[t] / (a_t * r**2)
        R = Phi * g

        results['F'].append(F)
        results['q'].append(1.0 - F)
        results['Phi'].append(Phi)
        results['g'].append(g)
        results['r'].append(r)
        results['R'].append(R)
        if t < T:
            results['B'].append(B_arr[t])

        if t < T:
            w = a_t * w - etas[t] * gb / r

    return {k: np.array(v) for k, v in results.items()}
