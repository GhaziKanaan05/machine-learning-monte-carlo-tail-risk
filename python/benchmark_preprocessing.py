import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


TRAINING_PATH = "data/training_pool_50000.csv"
BENCHMARK_PATH = "data/benchmark_1000000.csv"

FEATURE_COLUMNS = [
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
]


print("Loading data...")

training_df = pd.read_csv(
    TRAINING_PATH
)

benchmark_df = pd.read_csv(
    BENCHMARK_PATH
)

X_train = training_df[
    FEATURE_COLUMNS
].to_numpy()

X_benchmark = benchmark_df[
    FEATURE_COLUMNS
].to_numpy()


scaler = StandardScaler()

scaler.fit(
    X_train
)


# Warm-up
_ = scaler.transform(
    X_benchmark
)


times = []

print()
print("PREPROCESSING BENCHMARK")
print(
    "Benchmark observations:",
    len(X_benchmark),
)
print()


for repetition in range(5):

    start = time.perf_counter()

    X_scaled = scaler.transform(
        X_benchmark
    )

    finish = time.perf_counter()

    seconds = finish - start

    times.append(
        seconds
    )

    # Make result observable.
    checksum = float(
        np.sum(
            X_scaled[:, 0]
        )
    )

    print(
        f"Run {repetition + 1}: "
        f"{seconds:.6f} s "
        f"(checksum {checksum:.6f})"
    )


median_seconds = float(
    np.median(times)
)


print()
print(
    "Median preprocessing time:",
    median_seconds,
    "seconds",
)

print(
    "Throughput:",
    len(X_benchmark)
    / median_seconds,
    "scenarios/s",
)