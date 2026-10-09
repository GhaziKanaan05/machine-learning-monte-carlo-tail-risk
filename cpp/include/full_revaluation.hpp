#pragma once

#include "market_state.hpp"
#include "portfolio.hpp"
#include "scenario.hpp"

double full_revaluation_pnl(
    const Portfolio& portfolio,
    const MarketState& base_market,
    const Scenario& scenario,
    double horizon_years
);