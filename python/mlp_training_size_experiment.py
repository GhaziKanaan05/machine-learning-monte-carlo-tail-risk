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


TRAINING_PATH = (
    "data/training_replicates/"
    "training_replicate_1_250000.csv"
)

BENCHMARK_PATH = (
    "data/benchmark_1000000.csv"
)

TRAINING_SIZES = [
    2500,
    5000,
    10000,
    20000,
    50000,
    100000,
    250000,
]


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

    nearest_integer = round(
        raw_count
    )

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


def value_at_risk(
    pnl_values,
    confidence_level,
):
    losses = -pnl_values

    count = tail_count(
        len(losses),
        confidence_level,
    )

    sorted_losses = np.sort(
        losses
    )

    return sorted_losses[-count]


def expected_shortfall(
    pnl_values,
    confidence_level,
):
    losses = -pnl_values

    count = tail_count(
        len(losses),
        confidence_level,
    )

    sorted_losses = np.sort(
        losses
    )

    return np.mean(
        sorted_losses[-count:]
    )


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


def calculate_metrics(
    y_true,
    y_pred,
):
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    true_var_99 = value_at_risk(
        y_true,
        0.99,
    )

    predicted_var_99 = value_at_risk(
        y_pred,
        0.99,
    )

    true_es_99 = expected_shortfall(
        y_true,
        0.99,
    )

    predicted_es_99 = expected_shortfall(
        y_pred,
        0.99,
    )

    true_es_975 = expected_shortfall(
        y_true,
        0.975,
    )

    predicted_es_975 = expected_shortfall(
        y_pred,
        0.975,
    )

    var_99_error_pct = (
        100.0
        * abs(
            predicted_var_99
            - true_var_99
        )
        / abs(true_var_99)
    )

    es_99_error_pct = (
        100.0
        * abs(
            predicted_es_99
            - true_es_99
        )
        / abs(true_es_99)
    )

    es_975_error_pct = (
        100.0
        * abs(
            predicted_es_975
            - true_es_975
        )
        / abs(true_es_975)
    )

    true_losses = -y_true
    predicted_losses = -y_pred

    worst_5 = worst_tail_indices(
        true_losses,
        0.05,
    )

    worst_1 = worst_tail_indices(
        true_losses,
        0.01,
    )

    predicted_worst_1 = (
        worst_tail_indices(
            predicted_losses,
            0.01,
        )
    )

    worst_5_rmse = np.sqrt(
        np.mean(
            (
                y_pred[worst_5]
                - y_true[worst_5]
            ) ** 2
        )
    )

    worst_1_rmse = np.sqrt(
        np.mean(
            (
                y_pred[worst_1]
                - y_true[worst_1]
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

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "VaR_99_error_pct":
            var_99_error_pct,
        "ES_99_error_pct":
            es_99_error_pct,
        "ES_975_error_pct":
            es_975_error_pct,
        "Worst5_RMSE":
            worst_5_rmse,
        "Worst1_RMSE":
            worst_1_rmse,
        "Worst1_loss_bias":
            worst_1_loss_bias,
        "Tail_overlap_pct":
            tail_overlap_pct,
    }


print("Loading datasets...")

training_df = pd.read_csv(
    TRAINING_PATH
)

benchmark_df = pd.read_csv(
    BENCHMARK_PATH
)


X_pool = training_df[
    FEATURE_COLUMNS
].to_numpy()

y_pool = training_df[
    "pnl"
].to_numpy()

X_test_raw = benchmark_df[
    FEATURE_COLUMNS
].to_numpy()

y_test = benchmark_df[
    "pnl"
].to_numpy()


print(
    "Training pool:",
    len(y_pool),
)

print(
    "Benchmark:",
    len(y_test),
)


results = []


for training_size in TRAINING_SIZES:

    print()
    print("=" * 75)

    print(
        "Training size:",
        training_size,
    )

    print("=" * 75)


    X_train_raw = X_pool[
        :training_size
    ]

    y_train = y_pool[
        :training_size
    ]


    # -----------------------------------------------------
    # Fit input scaler using training subset only
    # -----------------------------------------------------

    x_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(
        X_train_raw
    )

    X_test = x_scaler.transform(
        X_test_raw
    )


    # -----------------------------------------------------
    # Fit target scaling using training subset only
    # -----------------------------------------------------

    y_mean = np.mean(
        y_train
    )

    y_std = np.std(
        y_train
    )

    y_train_scaled = (
        y_train - y_mean
    ) / y_std


    # -----------------------------------------------------
    # Frozen MLP
    # -----------------------------------------------------

    model = MLPRegressor(
        hidden_layer_sizes=(
            64,
            64,
            32,
        ),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=12345,
    )


    training_start = (
        time.perf_counter()
    )

    model.fit(
        X_train,
        y_train_scaled,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )


    inference_start = (
        time.perf_counter()
    )

    prediction_scaled = model.predict(
        X_test
    )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )


    prediction = (
        prediction_scaled
        * y_std
        + y_mean
    )


    metrics = calculate_metrics(
        y_test,
        prediction,
    )


    result = {
        "Training_size":
            training_size,
        "Iterations":
            model.n_iter_,
        "Training_seconds":
            training_seconds,
        "Inference_seconds":
            inference_seconds,
        **metrics,
    }

    results.append(
        result
    )


    print(
        "Iterations:",
        model.n_iter_,
    )

    print(
        "R^2:",
        metrics["R2"],
    )

    print(
        "RMSE:",
        metrics["RMSE"],
    )

    print(
        "99% VaR error:",
        metrics[
            "VaR_99_error_pct"
        ],
        "%",
    )

    print(
        "99% ES error:",
        metrics[
            "ES_99_error_pct"
        ],
        "%",
    )

    print(
        "97.5% ES error:",
        metrics[
            "ES_975_error_pct"
        ],
        "%",
    )

    print(
        "Worst 1% RMSE:",
        metrics[
            "Worst1_RMSE"
        ],
    )

    print(
        "Worst 1% loss bias:",
        metrics[
            "Worst1_loss_bias"
        ],
    )

    print(
        "Tail overlap:",
        metrics[
            "Tail_overlap_pct"
        ],
        "%",
    )

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


results_df = pd.DataFrame(
    results
)


columns = [
    "Training_size",
    "Iterations",
    "RMSE",
    "R2",
    "Worst5_RMSE",
    "Worst1_RMSE",
    "Worst1_loss_bias",
    "Tail_overlap_pct",
    "VaR_99_error_pct",
    "ES_99_error_pct",
    "ES_975_error_pct",
    "Training_seconds",
    "Inference_seconds",
]


print()
print()
print("=" * 130)
print("MLP TRAINING-SIZE EXPERIMENT")
print("=" * 130)


print(
    results_df[
        columns
    ].to_string(
        index=False
    )
)


output_path = (
    "data/"
    "mlp_training_size_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


print()
print(
    "Results saved to:",
    output_path,
)