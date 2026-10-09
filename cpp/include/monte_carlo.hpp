#pragma once

#include <cstddef>
#include <vector>

#include "market_state.hpp"
#include "portfolio.hpp"
#include "scenario_generator.hpp"

std::vector<double> run_full_revaluation_monte_carlo(
    const Portfolio& portfolio,
    const MarketState& base_market,
    ScenarioGenerator& generator,
    double horizon_years,
    std::size_t number_of_scenarios
);