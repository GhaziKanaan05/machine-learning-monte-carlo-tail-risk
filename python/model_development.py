import time

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
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


def tail_risk(
    pnl_values: np.ndarray,
    confidence_level: float = 0.99,
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
    losses: np.ndarray,
    tail_fraction: float,
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
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

    var_percentage_error = (
        100.0
        * abs(
            predicted_var
            - true_var
        )
        / abs(true_var)
    )

    es_percentage_error = (
        100.0
        * abs(
            predicted_es
            - true_es
        )
        / abs(true_es)
    )

    true_losses = -y_true
    predicted_losses = -y_pred

    true_worst_5 = worst_tail_indices(
        true_losses,
        0.05,
    )

    true_worst_1 = worst_tail_indices(
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
                y_pred[true_worst_5]
                - y_true[true_worst_5]
            ) ** 2
        )
    )

    worst_1_rmse = np.sqrt(
        np.mean(
            (
                y_pred[true_worst_1]
                - y_true[true_worst_1]
            ) ** 2
        )
    )

    worst_1_mean_loss_error = np.mean(
        predicted_losses[
            true_worst_1
        ]
        - true_losses[
            true_worst_1
        ]
    )

    overlap_count = len(
        np.intersect1d(
            true_worst_1,
            predicted_worst_1,
        )
    )

    tail_overlap = (
        100.0
        * overlap_count
        / len(true_worst_1)
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "VaR_true": true_var,
        "VaR_pred": predicted_var,
        "VaR_error_pct": var_percentage_error,
        "ES_true": true_es,
        "ES_pred": predicted_es,
        "ES_error_pct": es_percentage_error,
        "Worst5_RMSE": worst_5_rmse,
        "Worst1_RMSE": worst_1_rmse,
        "Worst1_loss_bias": worst_1_mean_loss_error,
        "Tail_overlap_pct": tail_overlap,
    }


print("Loading training data...")

training_df = pd.read_csv(
    TRAINING_PATH
)

print("Loading validation data...")

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


print()
print(
    "Training observations:",
    len(X_train),
)

print(
    "Validation observations:",
    len(X_validation),
)


# ---------------------------------------------------------
# Tail-aware sample weights
# ---------------------------------------------------------

training_losses = -y_train

loss_95 = np.quantile(
    training_losses,
    0.95,
)

loss_99 = np.quantile(
    training_losses,
    0.99,
)


moderate_tail_weights = np.ones(
    len(y_train)
)

moderate_tail_weights[
    training_losses >= loss_95
] = 2.5

moderate_tail_weights[
    training_losses >= loss_99
] = 6.0


strong_tail_weights = np.ones(
    len(y_train)
)

strong_tail_weights[
    training_losses >= loss_95
] = 4.0

strong_tail_weights[
    training_losses >= loss_99
] = 12.0


# ---------------------------------------------------------
# Candidate models
# ---------------------------------------------------------

candidates = [
    {
        "name": "HGB baseline",
        "model": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=12345,
        ),
        "weights": None,
    },

    {
        "name": "HGB 63 leaves",
        "model": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=12345,
        ),
        "weights": None,
    },

    {
        "name": "HGB 127 leaves",
        "model": HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_iter=600,
            max_leaf_nodes=127,
            min_samples_leaf=10,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=12345,
        ),
        "weights": None,
    },

    {
        "name": "HGB tail weighted moderate",
        "model": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=12345,
        ),
        "weights": moderate_tail_weights,
    },

    {
        "name": "HGB tail weighted strong",
        "model": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=12345,
        ),
        "weights": strong_tail_weights,
    },

    {
        "name": "Extra Trees",
        "model": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=12345,
        ),
        "weights": None,
    },
]


results = []


# ---------------------------------------------------------
# Train and evaluate every candidate
# ---------------------------------------------------------

for candidate in candidates:

    name = candidate[
        "name"
    ]

    model = candidate[
        "model"
    ]

    sample_weight = candidate[
        "weights"
    ]

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    training_start = (
        time.perf_counter()
    )

    if sample_weight is None:
        model.fit(
            X_train,
            y_train,
        )

    else:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
        )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    inference_start = (
        time.perf_counter()
    )

    prediction = model.predict(
        X_validation
    )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    metrics = calculate_metrics(
        y_validation,
        prediction,
    )

    metrics[
        "Model"
    ] = name

    metrics[
        "Training_seconds"
    ] = training_seconds

    metrics[
        "Inference_seconds"
    ] = inference_seconds

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
        "99% VaR error:",
        metrics["VaR_error_pct"],
        "%",
    )

    print(
        "99% ES error:",
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


# ---------------------------------------------------------
# Summary table
# ---------------------------------------------------------

results_df = pd.DataFrame(
    results
)

columns_to_show = [
    "Model",
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

summary = results_df[
    columns_to_show
].copy()


print()
print()
print("=" * 100)
print("MODEL DEVELOPMENT SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False,
    )
)


print()
print(
    "Models ranked by validation 99% ES error:"
)

ranked = summary.sort_values(
    "ES_error_pct"
)

print(
    ranked[
        [
            "Model",
            "ES_error_pct",
            "VaR_error_pct",
            "Worst1_RMSE",
            "Worst1_loss_bias",
            "Tail_overlap_pct",
            "RMSE",
            "R2",
        ]
    ].to_string(
        index=False
    )
)