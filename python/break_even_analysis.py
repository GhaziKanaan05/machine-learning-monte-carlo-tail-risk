import math

import pandas as pd


FULL_REVALUATION_1M = 30.011147
SCENARIO_GENERATION_1M = 0.515085
PREPROCESSING_1M = 0.0635475


LABEL_GENERATION_TIMES = {
    2500: 0.089760,
    5000: 0.153740,
    10000: 0.304769,
    20000: 0.637250,
    50000: 1.511702,
    100000: 3.129988,
    250000: 7.641216,
}


summary = pd.read_csv(
    "data/mlp_robustness_summary.csv"
)


rows = []


for _, row in summary.iterrows():

    training_size = int(
        row["Training_size"]
    )

    label_time = (
        LABEL_GENERATION_TIMES[
            training_size
        ]
    )

    training_time = (
        row[
            "Training_seconds_median"
        ]
    )

    inference_time = (
        row[
            "Inference_seconds_median"
        ]
    )

    upfront_cost = (
        label_time
        + training_time
    )

    ml_recurring_cost = (
        SCENARIO_GENERATION_1M
        + PREPROCESSING_1M
        + inference_time
    )

    first_run_cost = (
        upfront_cost
        + ml_recurring_cost
    )

    recurring_saving = (
        FULL_REVALUATION_1M
        - ml_recurring_cost
    )

    recurring_speedup = (
        FULL_REVALUATION_1M
        / ml_recurring_cost
    )

    break_even_continuous = (
        upfront_cost
        / recurring_saving
    )

    break_even_integer = math.ceil(
        break_even_continuous
    )

    rows.append({
        "Training_size":
            training_size,

        "VaR_99_error_pct_median":
            row[
                "VaR_99_error_pct_median"
            ],

        "ES_99_error_pct_median":
            row[
                "ES_99_error_pct_median"
            ],

        "Tail_overlap_pct_median":
            row[
                "Tail_overlap_pct_median"
            ],

        "Label_generation_seconds":
            label_time,

        "Training_seconds":
            training_time,

        "Upfront_seconds":
            upfront_cost,

        "ML_recurring_1m_seconds":
            ml_recurring_cost,

        "First_run_total_seconds":
            first_run_cost,

        "Recurring_speedup":
            recurring_speedup,

        "Break_even_runs_continuous":
            break_even_continuous,

        "First_integer_break_even_run":
            break_even_integer,
    })


results = pd.DataFrame(
    rows
)


output_path = (
    "data/break_even_analysis.csv"
)

results.to_csv(
    output_path,
    index=False
)


print()
print("=" * 140)
print(
    "COMPUTATIONAL BREAK-EVEN ANALYSIS"
)
print("=" * 140)

print(
    results.to_string(
        index=False
    )
)

print()
print(
    "Full-revaluation 1m runtime:",
    FULL_REVALUATION_1M,
    "seconds",
)

print(
    "Results saved to:",
    output_path,
)