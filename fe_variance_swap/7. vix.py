"""VIX-style model-free variance swap calculation from raw option prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .data import maturity_slice


@dataclass(frozen=True)
class VixVarianceResult:
    K_var: float
    vol_strike: float
    forward: float
    K0: float
    maturity: float
    n_options: int
    replication_table: pd.DataFrame


def compute_delta_k(strikes: np.ndarray) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    if np.any(np.diff(strikes) <= 0):
        raise ValueError("Strikes must be strictly increasing.")
    delta = np.empty_like(strikes)
    delta[0] = strikes[1] - strikes[0]
    delta[-1] = strikes[-1] - strikes[-2]
    delta[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])
    return delta


def call_to_put(call_price: np.ndarray, strike: np.ndarray, S0: float, r: float, q: float, T: float) -> np.ndarray:
    return call_price - S0 * np.exp(-q * T) + strike * np.exp(-r * T)


def compute_vix_variance(
    option_df: pd.DataFrame,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T: float = 4.0 / 12.0,
    tol: float = 1e-8,
) -> VixVarianceResult:
    """Estimate variance strike using the VIX-style option replication formula.

    The raw workbook contains call prices only. For strikes below the forward,
    put prices are inferred by put-call parity.
    """

    df_T = maturity_slice(option_df, T, tol=tol)
    strikes = df_T["Strike"].to_numpy(dtype=float)
    calls = df_T["Price"].to_numpy(dtype=float)
    F = float(S0 * np.exp((r - q) * T))
    below_forward = strikes[strikes < F]
    if len(below_forward) == 0:
        raise ValueError("Cannot identify K0 because no strike is below the forward.")
    K0 = float(np.max(below_forward))

    delta_k = compute_delta_k(strikes)
    puts = call_to_put(calls, strikes, S0, r, q, T)
    q_prices = np.where(strikes < F, puts, calls)
    if np.any(q_prices < -1e-10):
        raise ValueError("Synthetic OTM option prices contain negative values; check data/parity assumptions.")
    q_prices = np.maximum(q_prices, 0.0)

    contribution = (delta_k / (strikes * strikes)) * q_prices
    variance = (2.0 * np.exp(r * T) / T) * float(np.sum(contribution)) - (1.0 / T) * ((F / K0) - 1.0) ** 2
    if variance <= 0:
        raise ValueError(f"VIX-style variance estimate is non-positive: {variance}")

    table = df_T.copy()
    table["Forward"] = F
    table["K0"] = K0
    table["DeltaK"] = delta_k
    table["SyntheticPut"] = puts
    table["Q"] = q_prices
    table["Contribution"] = contribution
    table["OptionTypeUsed"] = np.where(strikes < F, "Put via parity", "Call")

    return VixVarianceResult(
        K_var=float(variance),
        vol_strike=float(np.sqrt(variance)),
        forward=F,
        K0=K0,
        maturity=T,
        n_options=int(len(df_T)),
        replication_table=table,
    )
