#pragma once

#include <cstddef>
#include <string>

#include "market_state.hpp"
#include "portfolio.hpp"
#include "scenario_generator.hpp"

void export_scenario_dataset_csv(
    const Portfolio& portfolio,
    const MarketState& base_market,
    ScenarioGenerator& generator,
    double horizon_years,
    std::size_t number_of_scenarios,
    const std::string& file_path
);