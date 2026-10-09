import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


TRAINING_PATH = "data/training_pool_50000.csv"
VALIDATION_PATH = "data/validation_50000.csv"


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


def tail_count(
    number_of_observations,
    confidence_level,
):
    raw_count = (
        (1.0 - confidence_level)
        * number_of_observations
    )

    nearest_integer = round(raw_count)

    if abs(
        raw_count - nearest_integer
    ) < 1e-10 * max(
        1.0,
        abs(raw_count),
    ):
        raw_count = nearest_integer

    return max(
        1,
        int(np.ceil(raw_count)),
    )


def tail_risk(
    pnl_values,
    confidence_level=0.99,
):
    losses = -pnl_values

    count = tail_count(
        len(losses),
        confidence_level,
    )

    sorted_losses = np.sort(
        losses
    )

    var = sorted_losses[
        -count
    ]

    es = np.mean(
        sorted_losses[
            -count:
        ]
    )

    return var, es


def worst_tail_indices(
    losses,
    fraction,
):
    count = max(
        1,
        int(
            round(
                fraction
                * len(losses)
            )
        ),
    )

    return np.argpartition(
        losses,
        -count,
    )[-count:]


print("Loading data...")

training_df = pd.read_csv(
    TRAINING_PATH
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)


X_train = training_df[
    FEATURE_COLUMNS
].to_numpy()

y_train = training_df[
    "pnl"
].to_numpy()

X_validation = validation_df[
    FEATURE_COLUMNS
].to_numpy()

y_validation = validation_df[
    "pnl"
].to_numpy()


print(
    "Training observations:",
    len(y_train),
)

print(
    "Validation observations:",
    len(y_validation),
)


# ---------------------------------------------------------
# Standardise inputs
# ---------------------------------------------------------

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
    X_train
)

X_validation_scaled = x_scaler.transform(
    X_validation
)


# ---------------------------------------------------------
# Standardise target
# ---------------------------------------------------------

y_mean = np.mean(
    y_train
)

y_std = np.std(
    y_train
)

y_train_scaled = (
    y_train - y_mean
) / y_std


# ---------------------------------------------------------
# Neural-network surrogate
#
# Architecture:
# 11 -> 64 -> 64 -> 32 -> 1
# ---------------------------------------------------------

model = MLPRegressor(
    hidden_layer_sizes=(
        64,
        64,
        32,
    ),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=12345,
)


print()
print("Training MLP...")

training_start = time.perf_counter()

model.fit(
    X_train_scaled,
    y_train_scaled,
)

training_seconds = (
    time.perf_counter()
    - training_start
)


print("Running validation inference...")

inference_start = time.perf_counter()

prediction_scaled = model.predict(
    X_validation_scaled
)

prediction = (
    prediction_scaled
    * y_std
    + y_mean
)

inference_seconds = (
    time.perf_counter()
    - inference_start
)


# ---------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        prediction,
    )
)

mae = mean_absolute_error(
    y_validation,
    prediction,
)

r2 = r2_score(
    y_validation,
    prediction,
)


# ---------------------------------------------------------
# Tail risk
# ---------------------------------------------------------

true_var, true_es = tail_risk(
    y_validation,
    0.99,
)

pred_var, pred_es = tail_risk(
    prediction,
    0.99,
)


var_error_pct = (
    100.0
    * abs(
        pred_var - true_var
    )
    / abs(true_var)
)

es_error_pct = (
    100.0
    * abs(
        pred_es - true_es
    )
    / abs(true_es)
)


# ---------------------------------------------------------
# Tail diagnostics
# ---------------------------------------------------------

true_losses = -y_validation
predicted_losses = -prediction


worst_5 = worst_tail_indices(
    true_losses,
    0.05,
)

worst_1 = worst_tail_indices(
    true_losses,
    0.01,
)

predicted_worst_1 = worst_tail_indices(
    predicted_losses,
    0.01,
)


worst_5_rmse = np.sqrt(
    np.mean(
        (
            prediction[worst_5]
            - y_validation[worst_5]
        ) ** 2
    )
)

worst_1_rmse = np.sqrt(
    np.mean(
        (
            prediction[worst_1]
            - y_validation[worst_1]
        ) ** 2
    )
)


worst_1_loss_bias = np.mean(
    predicted_losses[worst_1]
    - true_losses[worst_1]
)


overlap_count = len(
    np.intersect1d(
        worst_1,
        predicted_worst_1,
    )
)

tail_overlap_pct = (
    100.0
    * overlap_count
    / len(worst_1)
)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

print()
print("=" * 70)
print("MLP BASELINE — 50K TRAINING / 50K VALIDATION")
print("=" * 70)

print()
print("Iterations:", model.n_iter_)

print()
print("OVERALL ACCURACY")
print("RMSE:", rmse)
print("MAE:", mae)
print("R^2:", r2)

print()
print("TAIL RISK")

print(
    "True 99% VaR:",
    true_var,
)

print(
    "Predicted 99% VaR:",
    pred_var,
)

print(
    "VaR error:",
    var_error_pct,
    "%",
)

print()

print(
    "True 99% ES:",
    true_es,
)

print(
    "Predicted 99% ES:",
    pred_es,
)

print(
    "ES error:",
    es_error_pct,
    "%",
)

print()
print("TAIL DIAGNOSTICS")

print(
    "Worst 5% RMSE:",
    worst_5_rmse,
)

print(
    "Worst 1% RMSE:",
    worst_1_rmse,
)

print(
    "Worst 1% loss bias:",
    worst_1_loss_bias,
)

print(
    "Worst 1% tail overlap:",
    tail_overlap_pct,
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