import numpy as np
import pandas as pd


BENCHMARK_PATH = "data/benchmark_1000000.csv"

HORIZON_YEARS = 10.0 / 252.0

BASE_VOLS = np.array([
    0.18,
    0.22,
    0.25,
    0.30,
    0.35,
])

VOL_SHOCK_SCALES = np.array([
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
])

RATE_SHOCK_SCALE = 0.0025

VOL_FLOOR = 0.0001


spot_columns = [
    f"spot_log_return_{i}"
    for i in range(5)
]

vol_columns = [
    f"volatility_shock_{i}"
    for i in range(5)
]

rate_column = "rate_shock"


print("Loading benchmark...")

df = pd.read_csv(
    BENCHMARK_PATH
)


spots = df[
    spot_columns
].to_numpy()

vols = df[
    vol_columns
].to_numpy()

rates = df[
    rate_column
].to_numpy()


# ---------------------------------------------------------
# Means
# ---------------------------------------------------------

print()
print("=" * 70)
print("EMPIRICAL MEANS")
print("=" * 70)

for i in range(5):
    print(
        f"Spot return {i}: "
        f"{spots[:, i].mean():.8f}"
    )

for i in range(5):
    print(
        f"Vol shock {i}:  "
        f"{vols[:, i].mean():.8f}"
    )

print(
    f"Rate shock:   "
    f"{rates.mean():.8f}"
)


# ---------------------------------------------------------
# Standard deviations
# ---------------------------------------------------------

expected_spot_std = (
    BASE_VOLS
    * np.sqrt(HORIZON_YEARS)
)

print()
print("=" * 70)
print("STANDARD DEVIATIONS")
print("=" * 70)

for i in range(5):

    empirical = spots[
        :, i
    ].std(ddof=1)

    expected = expected_spot_std[i]

    print(
        f"Spot {i}: "
        f"empirical={empirical:.8f}, "
        f"expected={expected:.8f}, "
        f"ratio={empirical / expected:.4f}"
    )


for i in range(5):

    empirical = vols[
        :, i
    ].std(ddof=1)

    expected = VOL_SHOCK_SCALES[i]

    print(
        f"Vol {i}:  "
        f"empirical={empirical:.8f}, "
        f"expected={expected:.8f}, "
        f"ratio={empirical / expected:.4f}"
    )


rate_std = rates.std(
    ddof=1
)

print(
    f"Rate:   "
    f"empirical={rate_std:.8f}, "
    f"expected={RATE_SHOCK_SCALE:.8f}, "
    f"ratio={rate_std / RATE_SHOCK_SCALE:.4f}"
)


# ---------------------------------------------------------
# Correlations
# ---------------------------------------------------------

all_factors = np.column_stack([
    spots,
    vols,
    rates,
])

correlation = np.corrcoef(
    all_factors,
    rowvar=False,
)


spot_spot_values = []

for i in range(5):
    for j in range(i + 1, 5):
        spot_spot_values.append(
            correlation[i, j]
        )


vol_vol_values = []

for i in range(5):
    for j in range(i + 1, 5):
        vol_vol_values.append(
            correlation[
                5 + i,
                5 + j,
            ]
        )


spot_vol_values = []

for i in range(5):
    for j in range(5):
        spot_vol_values.append(
            correlation[
                i,
                5 + j,
            ]
        )


rate_correlations = []

for i in range(10):
    rate_correlations.append(
        correlation[i, 10]
    )


print()
print("=" * 70)
print("CORRELATION DIAGNOSTICS")
print("=" * 70)

print(
    "Mean spot-spot correlation:",
    np.mean(spot_spot_values),
)

print(
    "Target:",
    0.35,
)

print()

print(
    "Mean vol-vol correlation:",
    np.mean(vol_vol_values),
)

print(
    "Target:",
    0.50,
)

print()

print(
    "Mean spot-vol correlation:",
    np.mean(spot_vol_values),
)

print(
    "Target:",
    -0.30,
)

print()

print(
    "Mean linear rate correlation:",
    np.mean(rate_correlations),
)

print(
    "Maximum absolute rate correlation:",
    np.max(
        np.abs(rate_correlations)
    ),
)


# ---------------------------------------------------------
# Volatility-floor diagnostic
# ---------------------------------------------------------

shocked_vols_before_floor = (
    BASE_VOLS
    + vols
)

floor_hits = (
    shocked_vols_before_floor
    < VOL_FLOOR
)

total_floor_hits = int(
    floor_hits.sum()
)

total_vol_observations = (
    floor_hits.size
)

scenarios_with_floor_hit = int(
    np.any(
        floor_hits,
        axis=1,
    ).sum()
)


print()
print("=" * 70)
print("VOLATILITY FLOOR")
print("=" * 70)

print(
    "Individual volatility floor hits:",
    total_floor_hits,
)

print(
    "Out of volatility observations:",
    total_vol_observations,
)

print(
    "Individual floor-hit rate:",
    100.0
    * total_floor_hits
    / total_vol_observations,
    "%",
)

print()

print(
    "Scenarios with at least one floor hit:",
    scenarios_with_floor_hit,
)

print(
    "Scenario floor-hit rate:",
    100.0
    * scenarios_with_floor_hit
    / len(df),
    "%",
)


for i in range(5):

    hits = int(
        floor_hits[:, i].sum()
    )

    print(
        f"Underlying {i}: "
        f"{hits} hits "
        f"({100.0 * hits / len(df):.6f}%)"
    )