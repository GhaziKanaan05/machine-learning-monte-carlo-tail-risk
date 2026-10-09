# Can Machine Learning Reduce the Computational Cost of Monte Carlo Tail-Risk Estimation Without Sacrificing Accuracy?

A computational research project investigating whether machine-learning surrogate models can accelerate Monte Carlo portfolio tail-risk estimation while preserving the accuracy of Value-at-Risk (VaR) and Expected Shortfall (ES).

## Research question

**How much full-revaluation training data is required for a machine-learning surrogate to estimate extreme portfolio tail risk accurately, and when does the resulting computational saving outweigh the surrogate error?**

## Methodology

The experiment uses a synthetic nonlinear portfolio containing 500 European options across five equities, approximately delta-neutral at inception.

A C++20 engine is used for:

- correlated heavy-tailed scenario generation
- Black-Scholes option pricing
- full portfolio revaluation
- Monte Carlo P&L generation
- dataset export and runtime benchmarking

Python is used for:

- Histogram Gradient Boosting and MLP surrogate modelling
- training-size experiments
- tail-specific accuracy analysis
- robustness testing across independent training pools
- paired bootstrap uncertainty analysis
- computational break-even analysis

The market model contains 11 risk factors: five equity returns, five volatility shocks, and one interest-rate shock. Scenarios are generated using a correlated variance-standardised Student-t distribution with six degrees of freedom.

The reference benchmark contains 1,000,000 independently generated scenarios.

## Main result

Within this synthetic experiment, the 50,000-observation MLP offered the strongest practical accuracy-compute trade-off.

Across five independent training pools, the 50,000-observation model achieved median:

- **99% VaR absolute error:** 0.69%
- **99% Expected Shortfall absolute error:** 3.72%
- **True worst-1% tail overlap:** 89.37%

For a one-million-scenario evaluation, recurring ML computation took approximately 1.20 seconds compared with 30.01 seconds for direct full revaluation, corresponding to roughly a **25x compute-only speedup**.

The 250,000-observation MLP achieved the strongest overall predictive accuracy, but required substantially greater upfront training cost.

## Repository structure

cpp/
- include/ — C++ headers
- src/ — pricing, scenario generation and revaluation
- tests/ — automated tests

python/ — surrogate modelling and statistical analysis

figures/ — final research figures

CMakeLists.txt — C++20/CMake build configuration

## Scope

The portfolio and market model are synthetic rather than market-calibrated. The purpose of the project is to study the computational and statistical trade-off between full revaluation and machine-learning approximation under a controlled experimental design.

Reported speedups are compute-only and exclude file I/O and cross-language integration overhead.
