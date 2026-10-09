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


def tail_risk(pnl, confidence=0.99):
    losses = -pnl

    count = int(
        round(
            (1.0 - confidence)
            * len(losses)
        )
    )

    count = max(1, count)

    sorted_losses = np.sort(losses)

    var = sorted_losses[-count]

    es = np.mean(
        sorted_losses[-count:]
    )

    return var, es


def worst_tail_indices(
    losses,
    fraction,
):
    count = int(
        round(
            fraction * len(losses)
        )
    )

    count = max(1, count)

    return np.argpartition(
        losses,
        -count,
    )[-count:]


print("Loading data...")

train_df = pd.read_csv(
    TRAINING_PATH
)

test_df = pd.read_csv(
    BENCHMARK_PATH
)


X_train = train_df[
    FEATURE_COLUMNS
].to_numpy()

y_train = train_df[
    "pnl"
].to_numpy()

X_test = test_df[
    FEATURE_COLUMNS
].to_numpy()

y_test = test_df[
    "pnl"
].to_numpy()


# Tail weights chosen using validation data only
training_losses = -y_train

loss_95 = np.quantile(
    training_losses,
    0.95,
)

loss_99 = np.quantile(
    training_losses,
    0.99,
)

sample_weights = np.ones(
    len(y_train)
)

sample_weights[
    training_losses >= loss_95
] = 5.0

sample_weights[
    training_losses >= loss_99
] = 12.0


model = HistGradientBoostingRegressor(
    learning_rate=0.04,
    max_iter=500,
    max_leaf_nodes=63,
    min_samples_leaf=15,
    l2_regularization=0.5,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=12345,
)


print("Training frozen model...")

start = time.perf_counter()

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
)

training_seconds = (
    time.perf_counter() - start
)


print("Running 1m benchmark inference...")

start = time.perf_counter()

prediction = model.predict(
    X_test
)

inference_seconds = (
    time.perf_counter() - start
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction,
    )
)

mae = mean_absolute_error(
    y_test,
    prediction,
)

r2 = r2_score(
    y_test,
    prediction,
)


true_var, true_es = tail_risk(
    y_test
)

pred_var, pred_es = tail_risk(
    prediction
)


var_error_pct = (
    100.0
    * abs(pred_var - true_var)
    / true_var
)

es_error_pct = (
    100.0
    * abs(pred_es - true_es)
    / true_es
)


true_losses = -y_test
pred_losses = -prediction

worst_5 = worst_tail_indices(
    true_losses,
    0.05,
)

worst_1 = worst_tail_indices(
    true_losses,
    0.01,
)

pred_worst_1 = worst_tail_indices(
    pred_losses,
    0.01,
)


worst_5_rmse = np.sqrt(
    np.mean(
        (
            prediction[worst_5]
            - y_test[worst_5]
        ) ** 2
    )
)

worst_1_rmse = np.sqrt(
    np.mean(
        (
            prediction[worst_1]
            - y_test[worst_1]
        ) ** 2
    )
)

worst_1_bias = np.mean(
    pred_losses[worst_1]
    - true_losses[worst_1]
)


overlap = len(
    np.intersect1d(
        worst_1,
        pred_worst_1,
    )
)

tail_overlap = (
    100.0
    * overlap
    / len(worst_1)
)


print()
print("FROZEN 50K MODEL — FINAL BENCHMARK")
print()

print("RMSE:", rmse)
print("MAE:", mae)
print("R^2:", r2)

print()
print("True 99% VaR:", true_var)
print("Predicted 99% VaR:", pred_var)
print("VaR error:", var_error_pct, "%")

print()
print("True 99% ES:", true_es)
print("Predicted 99% ES:", pred_es)
print("ES error:", es_error_pct, "%")

print()
print("Worst 5% RMSE:", worst_5_rmse)
print("Worst 1% RMSE:", worst_1_rmse)
print(
    "Worst 1% loss bias:",
    worst_1_bias,
)

print(
    "Worst 1% tail overlap:",
    tail_overlap,
    "%",
)

print()
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
    "Inference scenarios/sec:",
    len(X_test)
    / inference_seconds,
)
