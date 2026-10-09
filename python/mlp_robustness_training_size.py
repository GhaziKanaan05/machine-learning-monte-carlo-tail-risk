import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


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

    sorted_losses = np.sort(losses)

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

    sorted_losses = np.sort(losses)

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

    var_error_pct = (
        100.0
        * abs(pred_var_99 - true_var_99)
        / abs(true_var_99)
    )

    es_99_error_pct = (
        100.0
        * abs(pred_es_99 - true_es_99)
        / abs(true_es_99)
    )

    es_975_error_pct = (
        100.0
        * abs(pred_es_975 - true_es_975)
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

    pred_worst_1 = worst_tail_indices(
        pred_losses,
        0.01,
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
            pred_worst_1,
        )
    )

    tail_overlap_pct = (
        100.0
        * overlap_count
        / len(worst_1)
    )

    return {
        "RMSE": rmse,
        "R2": r2,
        "VaR_99_error_pct":
            var_error_pct,
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


print(
    "Loading 1,000,000-scenario benchmark..."
)

benchmark_df = pd.read_csv(
    BENCHMARK_PATH
)

X_test_raw = benchmark_df[
    FEATURE_COLUMNS
].to_numpy()

y_test = benchmark_df[
    "pnl"
].to_numpy()


all_results = []


for replicate_index, path in enumerate(
    REPLICATE_PATHS
):

    replicate = replicate_index + 1
    seed = REPLICATE_SEEDS[
        replicate_index
    ]

    print()
    print("#" * 80)

    print(
        f"MLP TRAINING REPLICATE "
        f"{replicate} "
        f"(seed {seed})"
    )

    print("#" * 80)


    training_df = pd.read_csv(
        path
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
            f"Replicate {replicate}, "
            f"training size "
            f"{training_size}"
        )


        X_train_raw = X_pool[
            :training_size
        ]

        y_train = y_pool[
            :training_size
        ]


        # Input scaling fitted on
        # current training subset only.

        scaler = StandardScaler()

        X_train = (
            scaler.fit_transform(
                X_train_raw
            )
        )

        X_test = scaler.transform(
            X_test_raw
        )


        # Target scaling fitted on
        # current training subset only.

        y_mean = np.mean(
            y_train
        )

        y_std = np.std(
            y_train
        )

        y_train_scaled = (
            y_train - y_mean
        ) / y_std


        # Frozen MLP specification.

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


        start = time.perf_counter()

        model.fit(
            X_train,
            y_train_scaled,
        )

        training_seconds = (
            time.perf_counter()
            - start
        )


        start = time.perf_counter()

        prediction_scaled = (
            model.predict(
                X_test
            )
        )

        inference_seconds = (
            time.perf_counter()
            - start
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
            "Replicate":
                replicate,
            "Seed":
                seed,
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


raw_path = (
    "data/"
    "mlp_robustness_raw.csv"
)

results_df.to_csv(
    raw_path,
    index=False,
)


# ---------------------------------------------------------
# Aggregate across five independent training pools
# ---------------------------------------------------------

summary_rows = []


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


for training_size in TRAINING_SIZES:

    subset = results_df[
        results_df[
            "Training_size"
        ] == training_size
    ]

    row = {
        "Training_size":
            training_size,
    }


    for metric in metrics_to_summarize:

        values = subset[
            metric
        ].to_numpy()

        row[
            f"{metric}_median"
        ] = np.median(values)

        row[
            f"{metric}_mean"
        ] = np.mean(values)

        row[
            f"{metric}_std"
        ] = np.std(
            values,
            ddof=1,
        )

        row[
            f"{metric}_q25"
        ] = np.quantile(
            values,
            0.25,
        )

        row[
            f"{metric}_q75"
        ] = np.quantile(
            values,
            0.75,
        )


    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_path = (
    "data/"
    "mlp_robustness_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print()
print()
print("=" * 130)

print(
    "MLP ROBUSTNESS SUMMARY "
    "— MEDIANS ACROSS "
    "5 TRAINING REPLICATES"
)

print("=" * 130)


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
print(raw_path)

print()
print("Summary saved to:")
print(summary_path)