"""Calibration of Heston parameters to option prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .heston_fft import HestonParams, feller_margin, make_heston_params, price_option_dataframe, pricing_diagnostics


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    objective_value: float
    success: bool
    message: str
    n_iterations: int
    diagnostics: Dict[str, float]
    priced_df: pd.DataFrame


def _objective(x, option_df, S0, r, q, alpha, n, eta) -> float:
    v0, kappa, theta, sigma, rho = x
    if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or not (-0.999 < rho < 0.999):
        return 1e8
    try:
        params = make_heston_params(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        priced = price_option_dataframe(option_df, params, S0=S0, r=r, q=q, alpha=alpha, n=n, eta=eta)
        errors = priced["ModelPrice"].to_numpy() - priced["Price"].to_numpy()
        return float(np.sqrt(np.mean(errors * errors)))
    except Exception:
        return 1e8


def calibrate_heston(
    option_df: pd.DataFrame,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    alpha: float = 1.5,
    n: int = 4096,
    eta: float = 0.25,
    maxiter: int = 60,
    initial_guess: Iterable[float] = (0.04, 2.0, 0.04, 0.5, -0.5),
) -> CalibrationResult:
    """Calibrate Heston parameters by minimizing RMSE of call prices.

    Parameter vector order is ``v0, kappa, theta, sigma, rho``.
    """

    bounds = [
        (1e-5, 1.0),   # v0
        (1e-3, 20.0),  # kappa
        (1e-5, 1.0),   # theta
        (1e-3, 5.0),   # sigma
        (-0.999, 0.999),
    ]
    result = minimize(
        _objective,
        np.asarray(initial_guess, dtype=float),
        args=(option_df, S0, r, q, alpha, n, eta),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "maxfun": 600, "ftol": 1e-9, "maxls": 20},
    )

    params = make_heston_params(
        v0=result.x[0], kappa=result.x[1], theta=result.x[2], sigma=result.x[3], rho=result.x[4]
    )
    priced = price_option_dataframe(option_df, params, S0=S0, r=r, q=q, alpha=alpha, n=n, eta=eta)
    diagnostics = pricing_diagnostics(priced)
    diagnostics["feller_margin"] = float(feller_margin(params))

    return CalibrationResult(
        params=params,
        objective_value=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        n_iterations=int(result.nit),
        diagnostics=diagnostics,
        priced_df=priced,
    )
