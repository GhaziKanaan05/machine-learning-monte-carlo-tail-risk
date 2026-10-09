import time

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


TRAINING_PATH = "data/training_pool_50000.csv"
BENCHMARK_PATH = "data/benchmark_1000000.csv"

BOOTSTRAP_REPLICATES = 300
BOOTSTRAP_SEED = 24680

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


def tail_count(n, confidence):
    raw = (1.0 - confidence) * n
    nearest = round(raw)

    if abs(raw - nearest) < 1e-10 * max(
        1.0,
        abs(raw),
    ):
        raw = nearest

    return max(
        1,
        int(np.ceil(raw)),
    )


def var_es_from_losses(
    losses,
    confidence=0.99,
):
    n = len(losses)

    count = tail_count(
        n,
        confidence,
    )

    kth = n - count

    partitioned = np.partition(
        losses,
        kth,
    )

    tail = partitioned[kth:]

    var = partitioned[kth]
    es = np.mean(tail)

    return var, es


def percentile_interval(values):
    return (
        np.percentile(values, 2.5),
        np.percentile(values, 97.5),
    )


print("Loading datasets...")

training_df = pd.read_csv(
    TRAINING_PATH
)

benchmark_df = pd.read_csv(
    BENCHMARK_PATH
)


X_train_raw = training_df[
    FEATURE_COLUMNS
].to_numpy()

y_train = training_df[
    "pnl"
].to_numpy()

X_benchmark_raw = benchmark_df[
    FEATURE_COLUMNS
].to_numpy()

y_benchmark = benchmark_df[
    "pnl"
].to_numpy()


# ---------------------------------------------------------
# Reconstruct the frozen 50k MLP
# ---------------------------------------------------------

x_scaler = StandardScaler()

X_train = x_scaler.fit_transform(
    X_train_raw
)

X_benchmark = x_scaler.transform(
    X_benchmark_raw
)


y_mean = np.mean(y_train)
y_std = np.std(y_train)

y_train_scaled = (
    y_train - y_mean
) / y_std


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


print("Training frozen 50k MLP...")

model.fit(
    X_train,
    y_train_scaled,
)


print("Generating benchmark predictions...")

prediction_scaled = model.predict(
    X_benchmark
)

prediction = (
    prediction_scaled
    * y_std
    + y_mean
)


true_losses = -y_benchmark
ml_losses = -prediction


# ---------------------------------------------------------
# Point estimates
# ---------------------------------------------------------

true_var, true_es = var_es_from_losses(
    true_losses,
    0.99,
)

ml_var, ml_es = var_es_from_losses(
    ml_losses,
    0.99,
)


var_difference = (
    ml_var - true_var
)

es_difference = (
    ml_es - true_es
)

var_difference_pct = (
    100.0
    * var_difference
    / true_var
)

es_difference_pct = (
    100.0
    * es_difference
    / true_es
)


print()
print("POINT ESTIMATES")
print("True 99% VaR:", true_var)
print("ML 99% VaR:", ml_var)
print("Signed VaR difference:", var_difference)
print(
    "Signed VaR difference (%):",
    var_difference_pct,
)

print()

print("True 99% ES:", true_es)
print("ML 99% ES:", ml_es)
print("Signed ES difference:", es_difference)
print(
    "Signed ES difference (%):",
    es_difference_pct,
)


# ---------------------------------------------------------
# Paired non-parametric bootstrap
# ---------------------------------------------------------

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

n = len(true_losses)


true_var_boot = np.empty(
    BOOTSTRAP_REPLICATES
)

ml_var_boot = np.empty(
    BOOTSTRAP_REPLICATES
)

true_es_boot = np.empty(
    BOOTSTRAP_REPLICATES
)

ml_es_boot = np.empty(
    BOOTSTRAP_REPLICATES
)

var_difference_boot = np.empty(
    BOOTSTRAP_REPLICATES
)

es_difference_boot = np.empty(
    BOOTSTRAP_REPLICATES
)

var_difference_pct_boot = np.empty(
    BOOTSTRAP_REPLICATES
)

es_difference_pct_boot = np.empty(
    BOOTSTRAP_REPLICATES
)


print()
print(
    f"Running {BOOTSTRAP_REPLICATES} "
    "paired bootstrap replicates..."
)

start = time.perf_counter()


for b in range(
    BOOTSTRAP_REPLICATES
):

    indices = rng.integers(
        0,
        n,
        size=n,
        dtype=np.int32,
    )

    sampled_true_losses = (
        true_losses[indices]
    )

    sampled_ml_losses = (
        ml_losses[indices]
    )


    boot_true_var, boot_true_es = (
        var_es_from_losses(
            sampled_true_losses,
            0.99,
        )
    )

    boot_ml_var, boot_ml_es = (
        var_es_from_losses(
            sampled_ml_losses,
            0.99,
        )
    )


    boot_var_difference = (
        boot_ml_var
        - boot_true_var
    )

    boot_es_difference = (
        boot_ml_es
        - boot_true_es
    )


    true_var_boot[b] = (
        boot_true_var
    )

    ml_var_boot[b] = (
        boot_ml_var
    )

    true_es_boot[b] = (
        boot_true_es
    )

    ml_es_boot[b] = (
        boot_ml_es
    )

    var_difference_boot[b] = (
        boot_var_difference
    )

    es_difference_boot[b] = (
        boot_es_difference
    )

    var_difference_pct_boot[b] = (
        100.0
        * boot_var_difference
        / boot_true_var
    )

    es_difference_pct_boot[b] = (
        100.0
        * boot_es_difference
        / boot_true_es
    )


    if (
        (b + 1) % 25 == 0
        or b == 0
    ):
        print(
            f"Completed "
            f"{b + 1}/"
            f"{BOOTSTRAP_REPLICATES}"
        )


elapsed = (
    time.perf_counter()
    - start
)


# ---------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------

true_var_ci = percentile_interval(
    true_var_boot
)

ml_var_ci = percentile_interval(
    ml_var_boot
)

true_es_ci = percentile_interval(
    true_es_boot
)

ml_es_ci = percentile_interval(
    ml_es_boot
)

var_difference_ci = (
    percentile_interval(
        var_difference_boot
    )
)

es_difference_ci = (
    percentile_interval(
        es_difference_boot
    )
)

var_difference_pct_ci = (
    percentile_interval(
        var_difference_pct_boot
    )
)

es_difference_pct_ci = (
    percentile_interval(
        es_difference_pct_boot
    )
)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

print()
print("=" * 75)
print("PAIRED BOOTSTRAP UNCERTAINTY — FROZEN 50K MLP")
print("=" * 75)

print()
print("FULL-REVALUATION RISK ESTIMATES")

print(
    "99% VaR:",
    true_var,
)

print(
    "95% bootstrap CI:",
    true_var_ci,
)

print()

print(
    "99% ES:",
    true_es,
)

print(
    "95% bootstrap CI:",
    true_es_ci,
)


print()
print("ML RISK ESTIMATES")

print(
    "99% VaR:",
    ml_var,
)

print(
    "95% bootstrap CI:",
    ml_var_ci,
)

print()

print(
    "99% ES:",
    ml_es,
)

print(
    "95% bootstrap CI:",
    ml_es_ci,
)


print()
print("PAIRED ML - FULL DIFFERENCES")

print(
    "VaR difference:",
    var_difference,
)

print(
    "95% CI:",
    var_difference_ci,
)

print(
    "VaR difference (%):",
    var_difference_pct,
)

print(
    "95% CI (%):",
    var_difference_pct_ci,
)

print()

print(
    "ES difference:",
    es_difference,
)

print(
    "95% CI:",
    es_difference_ci,
)

print(
    "ES difference (%):",
    es_difference_pct,
)

print(
    "95% CI (%):",
    es_difference_pct_ci,
)


print()
print(
    "Bootstrap runtime:",
    elapsed,
    "seconds",
)


# ---------------------------------------------------------
# Save bootstrap draws
# ---------------------------------------------------------

results_df = pd.DataFrame({
    "true_var_99":
        true_var_boot,
    "ml_var_99":
        ml_var_boot,
    "true_es_99":
        true_es_boot,
    "ml_es_99":
        ml_es_boot,
    "var_difference":
        var_difference_boot,
    "es_difference":
        es_difference_boot,
    "var_difference_pct":
        var_difference_pct_boot,
    "es_difference_pct":
        es_difference_pct_boot,
})


output_path = (
    "data/bootstrap_uncertainty_50k.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print()
print(
    "Bootstrap draws saved to:",
    output_path,
)