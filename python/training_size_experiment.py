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

TRAINING_SIZES = [
    2500,
    5000,
    10000,
    20000,
    50000,
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


def tail_risk(
    pnl_values,
    confidence=0.99,
):
    losses = -pnl_values

    count = int(
        round(
            (1.0 - confidence)
            * len(losses)
        )
    )

    count = max(
        1,
        count,
    )

    sorted_losses = np.sort(
        losses
    )

    var = sorted_losses[-count]

    es = np.mean(
        sorted_losses[-count:]
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

    true_var, true_es = tail_risk(
        y_true
    )

    pred_var, pred_es = tail_risk(
        y_pred
    )

    var_error_pct = (
        100.0
        * abs(pred_var - true_var)
        / abs(true_var)
    )

    es_error_pct = (
        100.0
        * abs(pred_es - true_es)
        / abs(true_es)
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
        "VaR_true": true_var,
        "VaR_pred": pred_var,
        "VaR_error_pct": var_error_pct,
        "ES_true": true_es,
        "ES_pred": pred_es,
        "ES_error_pct": es_error_pct,
        "Worst5_RMSE": worst_5_rmse,
        "Worst1_RMSE": worst_1_rmse,
        "Worst1_loss_bias": worst_1_loss_bias,
        "Tail_overlap_pct": tail_overlap_pct,
    }


print("Loading datasets...")

training_df = pd.read_csv(
    TRAINING_PATH
)

benchmark_df = pd.read_csv(
    BENCHMARK_PATH
)


X_all = training_df[
    FEATURE_COLUMNS
].to_numpy()

y_all = training_df[
    "pnl"
].to_numpy()

X_test = benchmark_df[
    FEATURE_COLUMNS
].to_numpy()

y_test = benchmark_df[
    "pnl"
].to_numpy()


results = []


for training_size in TRAINING_SIZES:

    print()
    print("=" * 70)

    print(
        "Training size:",
        training_size,
    )

    print("=" * 70)


    # Nested training subsets:
    # 2.5k is contained in 5k,
    # 5k in 10k, etc.
    X_train = X_all[
        :training_size
    ]

    y_train = y_all[
        :training_size
    ]


    # Recalculate tail thresholds within
    # each training subset.
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
        training_size
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


    training_start = (
        time.perf_counter()
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )


    inference_start = (
        time.perf_counter()
    )

    prediction = model.predict(
        X_test
    )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )


    metrics = calculate_metrics(
        y_test,
        prediction,
    )


    result = {
        "Training_size": training_size,
        "Iterations": model.n_iter_,
        "Training_seconds": training_seconds,
        "Inference_seconds": inference_seconds,
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
        "RMSE:",
        metrics["RMSE"],
    )

    print(
        "R^2:",
        metrics["R2"],
    )

    print(
        "VaR error:",
        metrics["VaR_error_pct"],
        "%",
    )

    print(
        "ES error:",
        metrics["ES_error_pct"],
        "%",
    )

    print(
        "Worst 1% RMSE:",
        metrics["Worst1_RMSE"],
    )

    print(
        "Worst 1% loss bias:",
        metrics["Worst1_loss_bias"],
    )

    print(
        "Tail overlap:",
        metrics["Tail_overlap_pct"],
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
    "VaR_error_pct",
    "ES_error_pct",
    "Training_seconds",
    "Inference_seconds",
]


print()
print()
print("=" * 120)
print("TRAINING-SIZE EXPERIMENT")
print("=" * 120)

print(
    results_df[
        columns
    ].to_string(
        index=False
    )
)


output_path = (
    "data/training_size_results.csv"
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