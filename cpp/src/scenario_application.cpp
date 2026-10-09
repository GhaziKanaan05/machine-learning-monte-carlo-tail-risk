#include "scenario_application.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

MarketState apply_scenario(
    const MarketState& base_market,
    const Scenario& scenario
) {
    if (scenario.spot_log_returns.size() != base_market.spots.size()) {
        throw std::invalid_argument(
            "Scenario spot dimension does not match market."
        );
    }

    if (scenario.volatility_shocks.size() !=
        base_market.volatilities.size()) {

        throw std::invalid_argument(
            "Scenario volatility dimension does not match market."
        );
    }

    MarketState shocked_market = base_market;

    for (std::size_t i = 0; i < base_market.spots.size(); ++i) {
        shocked_market.spots[i] =
            base_market.spots[i]
            * std::exp(scenario.spot_log_returns[i]);

        shocked_market.volatilities[i] =
            std::max(
                base_market.volatilities[i]
                + scenario.volatility_shocks[i],
                0.0001
            );
    }

    shocked_market.risk_free_rate =
        base_market.risk_free_rate
        + scenario.rate_shock;

    return shocked_market;
}