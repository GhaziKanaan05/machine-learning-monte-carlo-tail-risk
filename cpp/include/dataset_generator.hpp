#pragma once

#include <cstddef>
#include <vector>

#include "market_state.hpp"
#include "portfolio.hpp"
#include "scenario_generator.hpp"
#include "scenario_result.hpp"

std::vector<ScenarioResult> generate_scenario_dataset(
    const Portfolio& portfolio,
    const MarketState& base_market,
    ScenarioGenerator& generator,
    double horizon_years,
    std::size_t number_of_scenarios
);