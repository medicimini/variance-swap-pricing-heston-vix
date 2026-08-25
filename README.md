## What this project does

1. Loads and validates raw option data.
2. Prices vanilla calls under Heston with the Carr-Madan FFT formula.
3. Calibrates Heston parameters to the full option surface.
4. Calculates the variance swap strike using:
   - the analytical Heston approximation,
   - Monte Carlo simulation under calibrated Heston dynamics,
   - the VIX-style model-free option replication formula.
5. Exports CSV diagnostics, plots, and a JSON summary in `outputs/`.

## Project structure

fe_variance_swap_project/
├── data/
│   └── option_data.xlsx
├── fe_variance_swap/
│   ├── calibration.py
│   ├── config.py
│   ├── data.py
│   ├── heston_fft.py
│   ├── monte_carlo.py
│   ├── plots.py
│   ├── variance_swap.py
│   └── vix.py
├── outputs/
│   └── optional summary/figures only
├── README.md
├── requirements.txt
└── run_assignment.py

## How to run

```bash
pip install -r requirements.txt
python run_assignment.py
```

The runner uses only NumPy, SciPy, Pandas and Matplotlib.

## Important implementation choices

- The raw workbook contains call prices only. For the VIX-style calculation, put prices below the forward are obtained by put-call parity.
- Heston calibration minimizes RMSE between market call prices and FFT model prices across all available maturities.
- Monte Carlo uses an Euler full-truncation scheme for the variance process and log-Euler stock updates so stock paths remain positive.
- The variance swap maturity is fixed at four months, `T = 4/12`, with `n = 84` daily monitoring steps.
