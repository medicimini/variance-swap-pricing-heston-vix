"""Central configuration for the Financial Engineering variance swap assignment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssignmentConfig:
    """Market and contract assumptions fixed by the assignment."""

    spot: float = 100.0
    rate: float = 0.05
    dividend_yield: float = 0.0
    variance_swap_maturity: float = 4.0 / 12.0
    trading_days_per_year: int = 252
    mc_steps: int = 84


CONFIG = AssignmentConfig()
