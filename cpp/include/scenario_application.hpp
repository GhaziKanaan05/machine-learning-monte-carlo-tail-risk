#pragma once

#include "market_state.hpp"
#include "scenario.hpp"

MarketState apply_scenario(
    const MarketState& base_market,
    const Scenario& scenario
);