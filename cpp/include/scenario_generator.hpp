#pragma once

#include <random>
#include <vector>

#include "scenario.hpp"

class ScenarioGenerator {
public:
    ScenarioGenerator(
        const std::vector<double>& spot_volatilities,
        const std::vector<double>& volatility_shock_scales,
        double rate_shock_scale,
        double horizon_years,
        double degrees_of_freedom,
        double spot_correlation,
        double volatility_correlation,
        double spot_volatility_correlation,
        unsigned int seed
    );

    Scenario generate();

private:
    std::vector<double> spot_volatilities_;
    std::vector<double> volatility_shock_scales_;

    double rate_shock_scale_;
    double horizon_years_;
    double degrees_of_freedom_;

    double spot_correlation_;
    double volatility_correlation_;
    double spot_volatility_correlation_;

    std::mt19937 rng_;
};
