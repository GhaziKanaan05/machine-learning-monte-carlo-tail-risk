import time

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


BENCHMARK_PATH = "data/benchmark_1000000.csv"

REPLICATE_PATHS = [
    "data/training_replicates/training_replicate_1_250000.csv",
    "data/training_replicates/training_replicate_2_250000.csv",
    "data/training_replicates/training_replicate_3_250000.csv",
    "data/training_replicates/training_replicate_4_250000.csv",
    "data/training_replicates/training_replicate_5_250000.csv",
]

REPLICATE_SEEDS = [
    54321,
    54322,
    54323,
    54324,
    54325,
]

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

    return sorted_losses[
        -count
    ]


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
        sorted_losses[
            -count:
        ]
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

    pred_var_99 = value_at_risk(
        y_pred,
        0.99,
    )

    true_es_99 = expected_shortfall(
        y_true,
        0.99,
    )

    pred_es_99 = expected_shortfall(
        y_pred,
        0.99,
    )

    true_es_975 = expected_shortfall(
        y_true,
        0.975,
    )

    pred_es_975 = expected_shortfall(
        y_pred,
        0.975,
    )

    var_99_error_pct = (
        100.0
        * abs(
            pred_var_99
            - true_var_99
        )
        / abs(true_var_99)
    )

    es_99_error_pct = (
        100.0
        * abs(
            pred_es_99
            - true_es_99
        )
        / abs(true_es_99)
    )

    es_975_error_pct = (
        100.0
        * abs(
            pred_es_975
            - true_es_975
        )
        / abs(true_es_975)
    )

    true_losses = -y_true
    pred_losses = -y_pred

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
            pred_losses,
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
        pred_losses[worst_1]
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

        "VaR_99_true": true_var_99,
        "VaR_99_pred": pred_var_99,
        "VaR_99_error_pct": var_99_error_pct,

        "ES_99_true": true_es_99,
        "ES_99_pred": pred_es_99,
        "ES_99_error_pct": es_99_error_pct,

        "ES_975_true": true_es_975,
        "ES_975_pred": pred_es_975,
        "ES_975_error_pct": es_975_error_pct,

        "Worst5_RMSE": worst_5_rmse,
        "Worst1_RMSE": worst_1_rmse,
        "Worst1_loss_bias": worst_1_loss_bias,
        "Tail_overlap_pct": tail_overlap_pct,
    }


print("Loading 1,000,000-scenario benchmark...")

benchmark_df = pd.read_csv(
    BENCHMARK_PATH
)

X_test = benchmark_df[
    FEATURE_COLUMNS
].to_numpy()

y_test = benchmark_df[
    "pnl"
].to_numpy()

print(
    "Benchmark observations:",
    len(y_test),
)


all_results = []


for replicate_index, replicate_path in enumerate(
    REPLICATE_PATHS
):

    replicate_number = (
        replicate_index + 1
    )

    replicate_seed = (
        REPLICATE_SEEDS[
            replicate_index
        ]
    )

    print()
    print("#" * 80)

    print(
        f"TRAINING REPLICATE "
        f"{replicate_number} "
        f"(seed {replicate_seed})"
    )

    print("#" * 80)

    training_df = pd.read_csv(
        replicate_path
    )

    X_pool = training_df[
        FEATURE_COLUMNS
    ].to_numpy()

    y_pool = training_df[
        "pnl"
    ].to_numpy()


    for training_size in TRAINING_SIZES:

        print()
        print(
            f"Replicate {replicate_number}, "
            f"training size {training_size}"
        )

        X_train = X_pool[
            :training_size
        ]

        y_train = y_pool[
            :training_size
        ]


        # Frozen tail-weighting methodology
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


        # Frozen model specification
        model = (
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=500,
                max_leaf_nodes=63,
                min_samples_leaf=15,
                l2_regularization=0.5,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=12345,
            )
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
            "Replicate": replicate_number,
            "Seed": replicate_seed,
            "Training_size": training_size,
            "Iterations": model.n_iter_,
            "Training_seconds": training_seconds,
            "Inference_seconds": inference_seconds,
            **metrics,
        }

        all_results.append(
            result
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
            "Tail overlap:",
            metrics[
                "Tail_overlap_pct"
            ],
            "%",
        )


results_df = pd.DataFrame(
    all_results
)


raw_output_path = (
    "data/"
    "robustness_training_size_raw.csv"
)

results_df.to_csv(
    raw_output_path,
    index=False,
)


# ---------------------------------------------------------
# Aggregate results across the 5 independent replicates
# ---------------------------------------------------------

summary_rows = []


for training_size in TRAINING_SIZES:

    subset = results_df[
        results_df[
            "Training_size"
        ] == training_size
    ]

    summary_row = {
        "Training_size": training_size,
    }


    metrics_to_summarize = [
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


    for metric in metrics_to_summarize:

        values = subset[
            metric
        ].to_numpy()

        summary_row[
            f"{metric}_median"
        ] = np.median(
            values
        )

        summary_row[
            f"{metric}_mean"
        ] = np.mean(
            values
        )

        summary_row[
            f"{metric}_std"
        ] = np.std(
            values,
            ddof=1,
        )

        summary_row[
            f"{metric}_q25"
        ] = np.quantile(
            values,
            0.25,
        )

        summary_row[
            f"{metric}_q75"
        ] = np.quantile(
            values,
            0.75,
        )


    summary_rows.append(
        summary_row
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_output_path = (
    "data/"
    "robustness_training_size_summary.csv"
)

summary_df.to_csv(
    summary_output_path,
    index=False,
)


print()
print()
print("=" * 120)
print("ROBUSTNESS SUMMARY — MEDIANS ACROSS 5 TRAINING REPLICATES")
print("=" * 120)


display_columns = [
    "Training_size",
    "VaR_99_error_pct_median",
    "ES_99_error_pct_median",
    "ES_975_error_pct_median",
    "Worst1_RMSE_median",
    "Worst1_loss_bias_median",
    "Tail_overlap_pct_median",
    "RMSE_median",
    "R2_median",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print()
print("Raw results saved to:")
print(raw_output_path)

print()
print("Summary results saved to:")
print(summary_output_path)