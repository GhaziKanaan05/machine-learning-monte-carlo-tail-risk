import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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


def tail_risk(
    pnl_values: np.ndarray,
    confidence_level: float = 0.99,
):
    losses = -pnl_values

    number_of_observations = len(losses)

    tail_count = int(
        round(
            (1.0 - confidence_level)
            * number_of_observations
        )
    )

    tail_count = max(
        1,
        tail_count,
    )

    sorted_losses = np.sort(losses)

    var = sorted_losses[-tail_count]

    es = np.mean(
        sorted_losses[-tail_count:]
    )

    return var, es


print("Loading training data...")

training_df = pd.read_csv(
    TRAINING_PATH
)

print("Loading benchmark data...")

benchmark_df = pd.read_csv(
    BENCHMARK_PATH
)

X_train = training_df[
    FEATURE_COLUMNS
].to_numpy()

y_train = training_df[
    "pnl"
].to_numpy()

X_test = benchmark_df[
    FEATURE_COLUMNS
].to_numpy()

y_test = benchmark_df[
    "pnl"
].to_numpy()


print()
print("Training observations:", len(X_train))
print("Benchmark observations:", len(X_test))


model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=12345,
)


print()
print("Training model...")

training_start = time.perf_counter()

model.fit(
    X_train,
    y_train,
)

training_seconds = (
    time.perf_counter()
    - training_start
)


print("Running benchmark inference...")

inference_start = time.perf_counter()

predicted_pnl = model.predict(
    X_test
)

inference_seconds = (
    time.perf_counter()
    - inference_start
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_pnl,
    )
)

mae = mean_absolute_error(
    y_test,
    predicted_pnl,
)

r2 = r2_score(
    y_test,
    predicted_pnl,
)


true_var_99, true_es_99 = tail_risk(
    y_test,
    0.99,
)

predicted_var_99, predicted_es_99 = tail_risk(
    predicted_pnl,
    0.99,
)


print()
print("MODEL ACCURACY")
print("RMSE:", rmse)
print("MAE:", mae)
print("R^2:", r2)


print()
print("TAIL RISK")

print(
    "True 99% VaR:",
    true_var_99,
)

print(
    "Predicted 99% VaR:",
    predicted_var_99,
)

print(
    "VaR absolute error:",
    abs(
        predicted_var_99
        - true_var_99
    ),
)

print(
    "VaR percentage error:",
    100.0
    * abs(
        predicted_var_99
        - true_var_99
    )
    / true_var_99,
)


print()

print(
    "True 99% ES:",
    true_es_99,
)

print(
    "Predicted 99% ES:",
    predicted_es_99,
)

print(
    "ES absolute error:",
    abs(
        predicted_es_99
        - true_es_99
    ),
)

print(
    "ES percentage error:",
    100.0
    * abs(
        predicted_es_99
        - true_es_99
    )
    / true_es_99,
)

# ---------------------------------------------------------
# Tail diagnostics
# ---------------------------------------------------------

true_losses = -y_test
predicted_losses = -predicted_pnl


def worst_tail_indices(
    losses: np.ndarray,
    tail_fraction: float,
):
    count = int(
        round(
            tail_fraction
            * len(losses)
        )
    )

    count = max(
        1,
        count,
    )

    return np.argpartition(
        losses,
        -count,
    )[-count:]


true_worst_5_indices = worst_tail_indices(
    true_losses,
    0.05,
)

true_worst_1_indices = worst_tail_indices(
    true_losses,
    0.01,
)

predicted_worst_1_indices = worst_tail_indices(
    predicted_losses,
    0.01,
)


worst_5_rmse = np.sqrt(
    np.mean(
        (
            predicted_pnl[
                true_worst_5_indices
            ]
            - y_test[
                true_worst_5_indices
            ]
        ) ** 2
    )
)

worst_1_rmse = np.sqrt(
    np.mean(
        (
            predicted_pnl[
                true_worst_1_indices
            ]
            - y_test[
                true_worst_1_indices
            ]
        ) ** 2
    )
)


worst_1_loss_error = np.mean(
    predicted_losses[
        true_worst_1_indices
    ]
    - true_losses[
        true_worst_1_indices
    ]
)


tail_overlap_count = len(
    np.intersect1d(
        true_worst_1_indices,
        predicted_worst_1_indices,
    )
)

tail_overlap_percentage = (
    100.0
    * tail_overlap_count
    / len(true_worst_1_indices)
)


print()
print("TAIL DIAGNOSTICS")

print(
    "True worst 5% observations:",
    len(true_worst_5_indices),
)

print(
    "RMSE in true worst 5%:",
    worst_5_rmse,
)

print()

print(
    "True worst 1% observations:",
    len(true_worst_1_indices),
)

print(
    "RMSE in true worst 1%:",
    worst_1_rmse,
)

print(
    "Mean loss error in true worst 1%:",
    worst_1_loss_error,
)

print(
    "Worst 1% tail overlap:",
    tail_overlap_percentage,
    "%",
)

print()
print("RUNTIME")

print(
    "Training time:",
    training_seconds,
    "seconds",
)

print(
    "Inference time:",
    inference_seconds,
    "seconds",
)

print(
    "Inference scenarios per second:",
    len(X_test)
    / inference_seconds,
)