"""Monte Carlo estimation of the realized variance under Heston."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .heston_fft import HestonParams, validate_heston_params


@dataclass(frozen=True)
class MonteCarloResult:
    K_var: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    n_paths: int
    n_steps: int
    mean_terminal_price: float
    mean_terminal_variance: float


def simulate_heston_paths(
    params: HestonParams | Dict[str, float],
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T: float = 4.0 / 12.0,
    n_steps: int = 84,
    n_paths: int = 20000,
    seed: int = 12345,
    full_truncation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Heston stock and variance paths with Euler full truncation."""

    params = validate_heston_params(params)
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    v = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = S0
    v[:, 0] = params.v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2_independent = rng.standard_normal(n_paths)
        z2 = params.rho * z1 + np.sqrt(1.0 - params.rho**2) * z2_independent

        v_pos = np.maximum(v[:, t], 0.0) if full_truncation else v[:, t]
        v_next = (
            v[:, t]
            + params.kappa * (params.theta - v_pos) * dt
            + params.sigma * np.sqrt(np.maximum(v_pos, 0.0)) * sqrt_dt * z2
        )
        if full_truncation:
            v_next = np.maximum(v_next, 0.0)

        # Log-Euler update keeps S positive.
        S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1)
        v[:, t + 1] = v_next
    return S, v


def realised_variance(S_paths: np.ndarray, annualisation: float = 252.0) -> np.ndarray:
    if np.any(S_paths <= 0):
        raise ValueError("Stock paths must be positive for log returns.")
    n_steps = S_paths.shape[1] - 1
    log_returns = np.diff(np.log(S_paths), axis=1)
    return (annualisation / n_steps) * np.sum(log_returns * log_returns, axis=1)


def estimate_variance_swap_mc(realised_variances: np.ndarray) -> Dict[str, float]:
    rv = np.asarray(realised_variances, dtype=float)
    if rv.ndim != 1 or len(rv) < 2:
        raise ValueError("realised_variances must be a one-dimensional array with at least two values.")
    mean = float(np.mean(rv))
    se = float(np.std(rv, ddof=1) / np.sqrt(len(rv)))
    return {
        "K_var": mean,
        "standard_error": se,
        "ci95_lower": mean - 1.96 * se,
        "ci95_upper": mean + 1.96 * se,
    }


def run_monte_carlo_variance_swap(
    params: HestonParams | Dict[str, float],
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T: float = 4.0 / 12.0,
    n_steps: int = 84,
    n_paths: int = 20000,
    seed: int = 12345,
) -> Tuple[MonteCarloResult, np.ndarray, np.ndarray, np.ndarray]:
    S, v = simulate_heston_paths(params, S0=S0, r=r, q=q, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    rv = realised_variance(S)
    est = estimate_variance_swap_mc(rv)
    result = MonteCarloResult(
        K_var=est["K_var"],
        standard_error=est["standard_error"],
        confidence_interval_95=(est["ci95_lower"], est["ci95_upper"]),
        n_paths=n_paths,
        n_steps=n_steps,
        mean_terminal_price=float(np.mean(S[:, -1])),
        mean_terminal_variance=float(np.mean(v[:, -1])),
    )
    return result, S, v, rv
