from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

HGB_PATH = (
    "data/robustness_training_size_summary.csv"
)

MLP_PATH = (
    "data/mlp_robustness_summary.csv"
)

BREAK_EVEN_PATH = (
    "data/break_even_analysis.csv"
)

BOOTSTRAP_PATH = (
    "data/bootstrap_uncertainty_50k.csv"
)

FIGURE_DIR = Path("figures")

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load results
# ============================================================

print("Loading final result files...")

hgb = pd.read_csv(
    HGB_PATH
)

mlp = pd.read_csv(
    MLP_PATH
)

tradeoff = pd.read_csv(
    BREAK_EVEN_PATH
)

bootstrap = pd.read_csv(
    BOOTSTRAP_PATH
)


training_sizes = (
    mlp["Training_size"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# Plot formatting
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def save_figure(
    figure,
    filename,
):
    png_path = (
        FIGURE_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{filename}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print(
        "Saved:",
        png_path,
    )

    print(
        "Saved:",
        pdf_path,
    )


def training_size_labels(values):

    labels = []

    for value in values:

        if value >= 1000:
            labels.append(
                f"{value / 1000:g}k"
            )

        else:
            labels.append(
                str(value)
            )

    return labels


# ============================================================
# FIGURE 1
#
# Training size vs median 99% ES error
# HGB vs MLP
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5.0)
)

ax.plot(
    hgb["Training_size"],
    hgb["ES_99_error_pct_median"],
    marker="o",
    linewidth=2,
    label="HGB",
)

ax.plot(
    mlp["Training_size"],
    mlp["ES_99_error_pct_median"],
    marker="o",
    linewidth=2,
    label="MLP",
)

ax.set_xscale(
    "log"
)

ax.set_xticks(
    training_sizes
)

ax.set_xticklabels(
    training_size_labels(
        training_sizes
    )
)

ax.set_xlabel(
    "Number of full-revaluation training labels"
)

ax.set_ylabel(
    "Median absolute 99% ES error (%)"
)

ax.set_title(
    "Tail-Risk Accuracy vs Training-Set Size"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "figure_1_es_error_vs_training_size",
)

plt.close(fig)


# ============================================================
# FIGURE 2
#
# Training size vs tail overlap
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5.0)
)

ax.plot(
    hgb["Training_size"],
    hgb["Tail_overlap_pct_median"],
    marker="o",
    linewidth=2,
    label="HGB",
)

ax.plot(
    mlp["Training_size"],
    mlp["Tail_overlap_pct_median"],
    marker="o",
    linewidth=2,
    label="MLP",
)

ax.set_xscale(
    "log"
)

ax.set_xticks(
    training_sizes
)

ax.set_xticklabels(
    training_size_labels(
        training_sizes
    )
)

ax.set_xlabel(
    "Number of full-revaluation training labels"
)

ax.set_ylabel(
    "Median true worst-1% tail overlap (%)"
)

ax.set_title(
    "Extreme-Tail Identification vs Training-Set Size"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "figure_2_tail_overlap_vs_training_size",
)

plt.close(fig)


# ============================================================
# FIGURE 3
#
# MLP accuracy-computation trade-off
#
# x = first-run total computational cost
# y = median 99% ES error
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.5, 5.0)
)

x = tradeoff[
    "First_run_total_seconds"
].to_numpy()

y = tradeoff[
    "ES_99_error_pct_median"
].to_numpy()

sizes = tradeoff[
    "Training_size"
].astype(int).to_numpy()


ax.plot(
    x,
    y,
    marker="o",
    linewidth=2,
)

ax.set_xscale(
    "log"
)


for x_value, y_value, size in zip(
    x,
    y,
    sizes,
):

    label = (
        f"{size / 1000:g}k"
    )

    ax.annotate(
        label,
        (
            x_value,
            y_value,
        ),
        xytext=(
            5,
            6,
        ),
        textcoords="offset points",
        fontsize=9,
    )


FULL_REVALUATION_TIME = (
    30.011147
)

ax.axvline(
    FULL_REVALUATION_TIME,
    linestyle="--",
    linewidth=1.5,
    label=(
        "1m full revaluation "
        "(30.01 s)"
    ),
)


ax.set_xlabel(
    "First 1m-scenario ML evaluation cost (seconds)"
)

ax.set_ylabel(
    "Median absolute 99% ES error (%)"
)

ax.set_title(
    "MLP Accuracy–Computation Trade-off"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "figure_3_accuracy_compute_tradeoff",
)

plt.close(fig)


# ============================================================
# FIGURE 4
#
# Paired bootstrap confidence intervals
# for frozen 50k MLP
# ============================================================

var_bootstrap = bootstrap[
    "var_difference_pct"
].to_numpy()

es_bootstrap = bootstrap[
    "es_difference_pct"
].to_numpy()


var_ci = np.percentile(
    var_bootstrap,
    [
        2.5,
        97.5,
    ],
)

es_ci = np.percentile(
    es_bootstrap,
    [
        2.5,
        97.5,
    ],
)


# Observed benchmark point estimates
# from frozen 50k MLP evaluation.

var_point = (
    1.7239158452753265
)

es_point = (
    -1.6484151501691569
)


points = np.array([
    var_point,
    es_point,
])

lower = np.array([
    var_point - var_ci[0],
    es_point - es_ci[0],
])

upper = np.array([
    var_ci[1] - var_point,
    es_ci[1] - es_point,
])


fig, ax = plt.subplots(
    figsize=(7.0, 5.0)
)

positions = np.array([
    0,
    1,
])

ax.errorbar(
    positions,
    points,
    yerr=np.vstack([
        lower,
        upper,
    ]),
    fmt="o",
    capsize=7,
    markersize=7,
    linewidth=2,
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.25,
)

ax.set_xticks(
    positions
)

ax.set_xticklabels([
    "99% VaR",
    "99% ES",
])

ax.set_ylabel(
    "ML − full-revaluation difference (%)"
)

ax.set_title(
    "Paired Bootstrap Uncertainty — Frozen 50k MLP"
)

ax.grid(
    True,
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

save_figure(
    fig,
    "figure_4_bootstrap_paired_difference",
)

plt.close(fig)


# ============================================================
# FINAL TABLE 1
#
# HGB vs MLP robustness comparison
# ============================================================

comparison = pd.DataFrame({
    "Training_size":
        mlp[
            "Training_size"
        ].astype(int),

    "HGB_VaR99_error_pct":
        hgb[
            "VaR_99_error_pct_median"
        ],

    "MLP_VaR99_error_pct":
        mlp[
            "VaR_99_error_pct_median"
        ],

    "HGB_ES99_error_pct":
        hgb[
            "ES_99_error_pct_median"
        ],

    "MLP_ES99_error_pct":
        mlp[
            "ES_99_error_pct_median"
        ],

    "HGB_tail_overlap_pct":
        hgb[
            "Tail_overlap_pct_median"
        ],

    "MLP_tail_overlap_pct":
        mlp[
            "Tail_overlap_pct_median"
        ],

    "HGB_R2":
        hgb[
            "R2_median"
        ],

    "MLP_R2":
        mlp[
            "R2_median"
        ],

    "HGB_worst1_RMSE":
        hgb[
            "Worst1_RMSE_median"
        ],

    "MLP_worst1_RMSE":
        mlp[
            "Worst1_RMSE_median"
        ],
})


comparison_path = (
    "data/final_model_comparison_table.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# FINAL TABLE 2
#
# MLP accuracy / computational trade-off
# ============================================================

tradeoff_columns = [
    "Training_size",
    "VaR_99_error_pct_median",
    "ES_99_error_pct_median",
    "Tail_overlap_pct_median",
    "Label_generation_seconds",
    "Training_seconds",
    "Upfront_seconds",
    "ML_recurring_1m_seconds",
    "First_run_total_seconds",
    "Recurring_speedup",
    "Break_even_runs_continuous",
    "First_integer_break_even_run",
]


final_tradeoff = tradeoff[
    tradeoff_columns
].copy()


tradeoff_path = (
    "data/final_mlp_tradeoff_table.csv"
)

final_tradeoff.to_csv(
    tradeoff_path,
    index=False,
)


# ============================================================
# FINAL TABLE 3
#
# Headline frozen 50k model results
# ============================================================

row_50k = tradeoff[
    tradeoff[
        "Training_size"
    ] == 50000
].iloc[0]


headline = pd.DataFrame({
    "Metric": [
        "Training labels",
        "Median 99% VaR absolute error (%)",
        "Median 99% ES absolute error (%)",
        "Median worst-1% tail overlap (%)",
        "Median MLP training time (s)",
        "Label-generation time (s)",
        "Recurring 1m ML cost (s)",
        "First-run total ML cost (s)",
        "Full-revaluation 1m cost (s)",
        "Recurring speedup",
        "Continuous break-even evaluations",
        "Observed VaR signed error (%)",
        "VaR paired-bootstrap CI lower (%)",
        "VaR paired-bootstrap CI upper (%)",
        "Observed ES signed error (%)",
        "ES paired-bootstrap CI lower (%)",
        "ES paired-bootstrap CI upper (%)",
    ],

    "Value": [
        50000,
        row_50k[
            "VaR_99_error_pct_median"
        ],
        row_50k[
            "ES_99_error_pct_median"
        ],
        row_50k[
            "Tail_overlap_pct_median"
        ],
        row_50k[
            "Training_seconds"
        ],
        row_50k[
            "Label_generation_seconds"
        ],
        row_50k[
            "ML_recurring_1m_seconds"
        ],
        row_50k[
            "First_run_total_seconds"
        ],
        FULL_REVALUATION_TIME,
        row_50k[
            "Recurring_speedup"
        ],
        row_50k[
            "Break_even_runs_continuous"
        ],
        var_point,
        var_ci[0],
        var_ci[1],
        es_point,
        es_ci[0],
        es_ci[1],
    ],
})


headline_path = (
    "data/final_headline_results.csv"
)

headline.to_csv(
    headline_path,
    index=False,
)


# ============================================================
# Terminal summaries
# ============================================================

print()
print("=" * 100)
print("FINAL HGB VS MLP COMPARISON")
print("=" * 100)

print(
    comparison.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("FINAL MLP COMPUTATIONAL TRADE-OFF")
print("=" * 100)

print(
    final_tradeoff.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("FROZEN 50K HEADLINE RESULTS")
print("=" * 100)

print(
    headline.to_string(
        index=False
    )
)


print()
print("Final tables saved to:")

print(
    comparison_path
)

print(
    tradeoff_path
)

print(
    headline_path
)

print()

print(
    "All figures saved in:",
    FIGURE_DIR
)