"""Plot helpers kept separate from core calculations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_calibration_plot(priced_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    plt.figure(figsize=(8, 5))
    for T, block in priced_df.groupby("Maturity"):
        plt.plot(block["Strike"], block["Price"], marker="o", linestyle="", markersize=3, label=f"Market T={T:.3f}")
        plt.plot(block["Strike"], block["ModelPrice"], linewidth=1, label=f"Heston T={T:.3f}")
    plt.xlabel("Strike")
    plt.ylabel("Call price")
    plt.title("Heston calibration: market vs model call prices")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_sample_paths_plot(S_paths: np.ndarray, T: float, output_path: str | Path, n_paths: int = 30) -> Path:
    output_path = Path(output_path)
    n = min(n_paths, S_paths.shape[0])
    grid = np.linspace(0.0, T, S_paths.shape[1])
    plt.figure(figsize=(8, 5))
    for i in range(n):
        plt.plot(grid, S_paths[i], linewidth=0.8, alpha=0.75)
    plt.xlabel("Time in years")
    plt.ylabel("Stock price")
    plt.title("Sample Heston Monte Carlo paths")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_realised_variance_histogram(realised_variance: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    plt.figure(figsize=(8, 5))
    plt.hist(realised_variance, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Annualised realised variance")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo realised variance distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
