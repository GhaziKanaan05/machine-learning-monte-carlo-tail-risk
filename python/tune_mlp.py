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

    pred_var, pred_es = tail_risk(
        y_pred,
        0.99,
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
        "VaR_error_pct": var_error_pct,
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
# Controlled candidate set
# ---------------------------------------------------------

candidates = [
    {
        "name": "ReLU 64-64-32 baseline",
        "hidden_layers": (64, 64, 32),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate": 1e-3,
    },

    {
        "name": "ReLU 128-64-32",
        "hidden_layers": (128, 64, 32),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate": 1e-3,
    },

    {
        "name": "ReLU 128-128-64",
        "hidden_layers": (128, 128, 64),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate": 1e-3,
    },

    {
        "name": "Tanh 64-64-32",
        "hidden_layers": (64, 64, 32),
        "activation": "tanh",
        "alpha": 1e-4,
        "learning_rate": 1e-3,
    },

    {
        "name": "ReLU 64-64-32 lower LR",
        "hidden_layers": (64, 64, 32),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate": 5e-4,
    },

    {
        "name": "ReLU 64-64-32 stronger L2",
        "hidden_layers": (64, 64, 32),
        "activation": "relu",
        "alpha": 1e-3,
        "learning_rate": 1e-3,
    },
]


results = []


for candidate in candidates:

    print()
    print("=" * 75)
    print(candidate["name"])
    print("=" * 75)

    model = MLPRegressor(
        hidden_layer_sizes=
            candidate["hidden_layers"],
        activation=
            candidate["activation"],
        solver="adam",
        alpha=
            candidate["alpha"],
        batch_size=256,
        learning_rate_init=
            candidate["learning_rate"],
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=12345,
    )


    start = time.perf_counter()

    model.fit(
        X_train_scaled,
        y_train_scaled,
    )

    training_seconds = (
        time.perf_counter()
        - start
    )


    start = time.perf_counter()

    prediction_scaled = model.predict(
        X_validation_scaled
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
        y_validation,
        prediction,
    )


    result = {
        "Model": candidate["name"],
        "Iterations": model.n_iter_,
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


results_df = pd.DataFrame(
    results
)


columns = [
    "Model",
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
print("=" * 130)
print("MLP MODEL DEVELOPMENT SUMMARY")
print("=" * 130)

print(
    results_df[
        columns
    ].to_string(
        index=False
    )
)


print()
print(
    "Ranked by validation 99% ES error:"
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


output_path = (
    "data/mlp_tuning_results.csv"
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