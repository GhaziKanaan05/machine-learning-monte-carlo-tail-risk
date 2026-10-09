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


WEIGHT_COMBINATIONS = [
    (2.5, 6.0),
    (3.0, 8.0),
    (3.0, 10.0),
    (4.0, 8.0),
    (4.0, 10.0),
    (4.0, 12.0),
    (5.0, 12.0),
    (5.0, 15.0),
    (6.0, 15.0),
    (6.0, 20.0),
]


def tail_risk(
    pnl_values,
    confidence_level=0.99,
):
    losses = -pnl_values

    raw_tail_count = (
        (1.0 - confidence_level)
        * len(losses)
    )

    nearest_integer = round(
        raw_tail_count
    )

    if abs(
        raw_tail_count
        - nearest_integer
    ) < 1e-10 * max(
        1.0,
        abs(raw_tail_count),
    ):
        raw_tail_count = nearest_integer

    tail_count = max(
        1,
        int(np.ceil(raw_tail_count)),
    )

    sorted_losses = np.sort(
        losses
    )

    var = sorted_losses[
        -tail_count
    ]

    es = np.mean(
        sorted_losses[
            -tail_count:
        ]
    )

    return var, es


def worst_tail_indices(
    losses,
    tail_fraction,
):
    count = max(
        1,
        int(
            round(
                tail_fraction
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
        y_true,
        0.99,
    )

    predicted_var, predicted_es = tail_risk(
        y_pred,
        0.99,
    )

    var_error_pct = (
        100.0
        * abs(
            predicted_var
            - true_var
        )
        / abs(true_var)
    )

    es_error_pct = (
        100.0
        * abs(
            predicted_es
            - true_es
        )
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

    predicted_worst_1 = worst_tail_indices(
        predicted_losses,
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
        predicted_losses[worst_1]
        - true_losses[worst_1]
    )

    overlap = len(
        np.intersect1d(
            worst_1,
            predicted_worst_1,
        )
    )

    tail_overlap_pct = (
        100.0
        * overlap
        / len(worst_1)
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "VaR_true": true_var,
        "VaR_pred": predicted_var,
        "VaR_error_pct": var_error_pct,
        "ES_true": true_es,
        "ES_pred": predicted_es,
        "ES_error_pct": es_error_pct,
        "Worst5_RMSE": worst_5_rmse,
        "Worst1_RMSE": worst_1_rmse,
        "Worst1_loss_bias": worst_1_loss_bias,
        "Tail_overlap_pct": tail_overlap_pct,
    }


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


training_losses = -y_train

loss_95 = np.quantile(
    training_losses,
    0.95,
)

loss_99 = np.quantile(
    training_losses,
    0.99,
)


print(
    "Training observations:",
    len(y_train),
)

print(
    "Validation observations:",
    len(y_validation),
)

print(
    "Training 95% loss threshold:",
    loss_95,
)

print(
    "Training 99% loss threshold:",
    loss_99,
)


results = []


for weight_95, weight_99 in WEIGHT_COMBINATIONS:

    sample_weights = np.ones(
        len(y_train)
    )

    sample_weights[
        training_losses >= loss_95
    ] = weight_95

    sample_weights[
        training_losses >= loss_99
    ] = weight_99


    print()
    print("=" * 70)

    print(
        f"Weights: worst 5% = {weight_95}, "
        f"worst 1% = {weight_99}"
    )

    print("=" * 70)


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


    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    training_seconds = (
        time.perf_counter()
        - start
    )


    prediction = model.predict(
        X_validation
    )


    metrics = calculate_metrics(
        y_validation,
        prediction,
    )


    metrics["Weight_95"] = weight_95
    metrics["Weight_99"] = weight_99
    metrics["Training_seconds"] = training_seconds

    results.append(
        metrics
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
        "Worst 5% RMSE:",
        metrics["Worst5_RMSE"],
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


results_df = pd.DataFrame(
    results
)


print()
print()
print("=" * 110)
print("TAIL-WEIGHT TUNING RESULTS")
print("=" * 110)


columns = [
    "Weight_95",
    "Weight_99",
    "RMSE",
    "R2",
    "Worst5_RMSE",
    "Worst1_RMSE",
    "Worst1_loss_bias",
    "Tail_overlap_pct",
    "VaR_error_pct",
    "ES_error_pct",
]


print(
    results_df[
        columns
    ].to_string(
        index=False
    )
)


print()
print(
    "Ranked by validation ES error:"
)


ranked = results_df.sort_values(
    "ES_error_pct"
)


print(
    ranked[
        columns
    ].to_string(
        index=False
    )
)