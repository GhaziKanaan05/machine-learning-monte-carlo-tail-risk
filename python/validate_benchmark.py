import math

import numpy as np
import pandas as pd


BENCHMARK_PATH = "data/benchmark_1000000.csv"
CONFIDENCE_LEVEL = 0.99


def tail_count(
    number_of_observations: int,
    confidence_level: float,
) -> int:
    raw_tail_count = (
        (1.0 - confidence_level)
        * number_of_observations
    )

    nearest_integer = round(raw_tail_count)

    if abs(
        raw_tail_count - nearest_integer
    ) < 1e-10 * max(
        1.0,
        abs(raw_tail_count),
    ):
        raw_tail_count = nearest_integer

    return max(
        1,
        math.ceil(raw_tail_count),
    )


df = pd.read_csv(BENCHMARK_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))

expected_columns = [
    "spot_log_return_0",
    "spot_log_return_1",
    "spot_log_return_2",
    "spot_log_return_3",
    "spot_log_return_4",
    "volatility_shock_0",
    "volatility_shock_1",
    "volatility_shock_2",
    "volatility_shock_3",
    "volatility_shock_4",
    "rate_shock",
    "pnl",
]

if list(df.columns) != expected_columns:
    raise ValueError(
        "Benchmark CSV columns are incorrect."
    )

if df.isna().any().any():
    raise ValueError(
        "Benchmark CSV contains missing values."
    )

pnl = df["pnl"].to_numpy()

losses = -pnl

losses.sort()

n_tail = tail_count(
    len(losses),
    CONFIDENCE_LEVEL,
)

var_99 = losses[-n_tail]

es_99 = np.mean(
    losses[-n_tail:]
)

print()
print("99% tail observations:", n_tail)
print("Mean 10-day P&L:", np.mean(pnl))
print("Worst 10-day P&L:", np.min(pnl))
print("Best 10-day P&L:", np.max(pnl))
print("99% VaR:", var_99)
print("99% Expected Shortfall:", es_99)