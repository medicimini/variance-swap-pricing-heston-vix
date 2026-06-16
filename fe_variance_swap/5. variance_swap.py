"""Variance swap strike calculations."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .heston_fft import HestonParams, validate_heston_params


def analytical_heston_variance_strike(params: HestonParams | Dict[str, float], T: float = 4.0 / 12.0) -> Dict[str, float]:
    """Approximate fair variance strike under Heston.

    K_var = ((1 - exp(-kappa T)) / (kappa T)) * (v0 - theta) + theta
    """

    params = validate_heston_params(params)
    if T <= 0:
        raise ValueError("T must be strictly positive.")
    weight = (1.0 - np.exp(-params.kappa * T)) / (params.kappa * T)
    k_var = weight * (params.v0 - params.theta) + params.theta
    if k_var <= 0:
        raise ValueError("Analytical variance strike is non-positive; calibration is probably unstable.")
    return {
        "K_var": float(k_var),
        "vol_strike": float(np.sqrt(k_var)),
        "mean_reversion_weight": float(weight),
    }
