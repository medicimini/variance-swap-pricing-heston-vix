"""End-to-end runner for the variance swap assignment.

Run from the project root:
    python run_assignment.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fe_variance_swap.config import CONFIG
from fe_variance_swap.data import load_option_data, validate_option_data
from fe_variance_swap.calibration import calibrate_heston
from fe_variance_swap.variance_swap import analytical_heston_variance_strike
from fe_variance_swap.monte_carlo import run_monte_carlo_variance_swap
from fe_variance_swap.vix import compute_vix_variance
from fe_variance_swap.plots import (
    save_calibration_plot,
    save_realised_variance_histogram,
    save_sample_paths_plot,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "option_data.xlsx"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    option_df = load_option_data(DATA_PATH)
    validation = validate_option_data(option_df, target_maturity=CONFIG.variance_swap_maturity)

    calibration = calibrate_heston(
        option_df,
        S0=CONFIG.spot,
        r=CONFIG.rate,
        q=CONFIG.dividend_yield,
        alpha=1.5,
        n=1024,
        eta=0.25,
        maxiter=60,
    )
    params = calibration.params
    analytical = analytical_heston_variance_strike(params, T=CONFIG.variance_swap_maturity)
    mc_result, S_paths, v_paths, rv = run_monte_carlo_variance_swap(
        params,
        S0=CONFIG.spot,
        r=CONFIG.rate,
        q=CONFIG.dividend_yield,
        T=CONFIG.variance_swap_maturity,
        n_steps=CONFIG.mc_steps,
        n_paths=20000,
        seed=12345,
    )
    vix = compute_vix_variance(
        option_df,
        S0=CONFIG.spot,
        r=CONFIG.rate,
        q=CONFIG.dividend_yield,
        T=CONFIG.variance_swap_maturity,
    )

    calibration.priced_df.to_csv(OUTPUT_DIR / "calibration_market_vs_model.csv", index=False)
    vix.replication_table.to_csv(OUTPUT_DIR / "vix_replication_table.csv", index=False)
    save_calibration_plot(calibration.priced_df, OUTPUT_DIR / "calibration_fit.png")
    save_sample_paths_plot(S_paths, CONFIG.variance_swap_maturity, OUTPUT_DIR / "mc_sample_paths.png")
    save_realised_variance_histogram(rv, OUTPUT_DIR / "mc_realised_variance_histogram.png")

    summary = {
        "data_validation": validation.as_dict(),
        "calibrated_heston_params": params.as_dict(),
        "calibration": {
            "success": calibration.success,
            "message": calibration.message,
            "objective_value_rmse": calibration.objective_value,
            "n_iterations": calibration.n_iterations,
            "diagnostics": calibration.diagnostics,
        },
        "approach_1_analytical_heston": analytical,
        "approach_2_monte_carlo": {
            "K_var": mc_result.K_var,
            "vol_strike": mc_result.K_var ** 0.5,
            "standard_error": mc_result.standard_error,
            "confidence_interval_95": mc_result.confidence_interval_95,
            "n_paths": mc_result.n_paths,
            "n_steps": mc_result.n_steps,
            "mean_terminal_price": mc_result.mean_terminal_price,
            "mean_terminal_variance": mc_result.mean_terminal_variance,
        },
        "approach_3_vix_style": {
            "K_var": vix.K_var,
            "vol_strike": vix.vol_strike,
            "forward": vix.forward,
            "K0": vix.K0,
            "n_options": vix.n_options,
        },
    }
    with open(OUTPUT_DIR / "summary_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
