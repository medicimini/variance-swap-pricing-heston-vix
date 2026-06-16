"""Heston model call pricing with the Carr-Madan FFT formula."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def make_heston_params(v0: float, kappa: float, theta: float, sigma: float, rho: float) -> HestonParams:
    params = HestonParams(float(v0), float(kappa), float(theta), float(sigma), float(rho))
    validate_heston_params(params)
    return params


def validate_heston_params(params: HestonParams | Dict[str, float]) -> HestonParams:
    if isinstance(params, dict):
        params = HestonParams(**{k: float(params[k]) for k in ["v0", "kappa", "theta", "sigma", "rho"]})
    checks = {
        "v0 > 0": params.v0 > 0,
        "kappa > 0": params.kappa > 0,
        "theta > 0": params.theta > 0,
        "sigma > 0": params.sigma > 0,
        "-1 < rho < 1": -1 < params.rho < 1,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("Invalid Heston parameters: " + ", ".join(failed))
    return params


def feller_margin(params: HestonParams | Dict[str, float]) -> float:
    params = validate_heston_params(params)
    return 2.0 * params.kappa * params.theta - params.sigma**2


def heston_characteristic_function(
    u: np.ndarray | complex,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: HestonParams | Dict[str, float],
) -> np.ndarray:
    """Risk-neutral characteristic function of log(S_T) under Heston.

    Uses the common "little Heston trap" representation through the ratio ``g``.
    """

    params = validate_heston_params(params)
    u = np.asarray(u, dtype=np.complex128)
    i = 1j

    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )

    b = kappa - rho * sigma * i * u
    d = np.sqrt(b * b + sigma * sigma * (i * u + u * u))
    g = (b - d) / (b + d)
    exp_neg_dT = np.exp(-d * T)

    # Avoid log of exact zero in pathological parameter points.
    denominator = 1.0 - g * exp_neg_dT
    numerator = 1.0 - g

    C = (
        i * u * (np.log(S0) + (r - q) * T)
        + (kappa * theta / (sigma * sigma))
        * ((b - d) * T - 2.0 * np.log(denominator / numerator))
    )
    D = ((b - d) / (sigma * sigma)) * ((1.0 - exp_neg_dT) / denominator)
    return np.exp(C + D * v0)


def carr_madan_fft_call_grid(
    T: float,
    params: HestonParams | Dict[str, float],
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    alpha: float = 1.5,
    n: int = 4096,
    eta: float = 0.25,
) -> pd.DataFrame:
    """Return a grid of strikes and call prices using Carr-Madan FFT."""

    if T <= 0:
        raise ValueError("Maturity T must be positive.")
    if alpha <= 0:
        raise ValueError("Carr-Madan damping alpha must be positive.")
    if n <= 0 or n & (n - 1) != 0:
        raise ValueError("n must be a positive power of two for FFT.")
    if eta <= 0:
        raise ValueError("eta must be positive.")

    params = validate_heston_params(params)
    j = np.arange(n)
    v = eta * j
    lambd = 2.0 * np.pi / (n * eta)
    b = 0.5 * n * lambd
    k = -b + lambd * j

    shifted_u = v - (alpha + 1.0) * 1j
    phi = heston_characteristic_function(shifted_u, T, S0, r, q, params)
    denominator = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * T) * phi / denominator

    weights = np.ones(n)
    weights[0] = 0.5
    fft_input = np.exp(1j * b * v) * psi * eta * weights
    fft_values = np.fft.fft(fft_input)

    strikes = np.exp(k)
    call_prices = np.exp(-alpha * k) * np.real(fft_values) / np.pi
    grid = pd.DataFrame({"Strike": strikes, "CallPrice": call_prices})
    grid = grid[np.isfinite(grid["CallPrice"]) & (grid["Strike"] > 0)].copy()
    grid["CallPrice"] = grid["CallPrice"].clip(lower=0.0)
    return grid.sort_values("Strike").reset_index(drop=True)


def heston_call_prices_for_maturity(
    strikes: Iterable[float],
    T: float,
    params: HestonParams | Dict[str, float],
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    alpha: float = 1.5,
    n: int = 4096,
    eta: float = 0.25,
) -> np.ndarray:
    strikes = np.asarray(list(strikes), dtype=float)
    if np.any(strikes <= 0):
        raise ValueError("All interpolation strikes must be positive.")

    grid = carr_madan_fft_call_grid(T, params, S0=S0, r=r, q=q, alpha=alpha, n=n, eta=eta)
    min_k, max_k = grid["Strike"].min(), grid["Strike"].max()
    if strikes.min() < min_k or strikes.max() > max_k:
        raise ValueError(
            f"Requested strikes [{strikes.min()}, {strikes.max()}] fall outside FFT grid [{min_k}, {max_k}]."
        )
    interpolator = interp1d(
        grid["Strike"].to_numpy(),
        grid["CallPrice"].to_numpy(),
        kind="linear",
        bounds_error=True,
    )
    return np.asarray(interpolator(strikes), dtype=float)


def price_option_dataframe(
    option_df: pd.DataFrame,
    params: HestonParams | Dict[str, float],
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    alpha: float = 1.5,
    n: int = 4096,
    eta: float = 0.25,
) -> pd.DataFrame:
    """Add Heston model prices and pricing errors to an option DataFrame."""

    required = {"Strike", "Maturity", "Price"}
    if not required.issubset(option_df.columns):
        raise ValueError(f"option_df must contain columns {sorted(required)}")

    priced_blocks = []
    for T, block in option_df.groupby("Maturity", sort=True):
        model_prices = heston_call_prices_for_maturity(
            block["Strike"].to_numpy(), float(T), params, S0=S0, r=r, q=q, alpha=alpha, n=n, eta=eta
        )
        priced = block.copy()
        priced["ModelPrice"] = model_prices
        priced["PricingError"] = priced["ModelPrice"] - priced["Price"]
        priced_blocks.append(priced)
    return pd.concat(priced_blocks, ignore_index=True)


def pricing_diagnostics(priced_df: pd.DataFrame) -> Dict[str, float]:
    err = priced_df["PricingError"].to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_error": float(np.mean(err)),
    }
