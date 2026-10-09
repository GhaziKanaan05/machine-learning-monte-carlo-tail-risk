#include "full_revaluation.hpp"
#include "scenario_application.hpp"

double full_revaluation_pnl(
    const Portfolio& portfolio,
    const MarketState& base_market,
    const Scenario& scenario,
    double horizon_years
) {
    double initial_value =
        portfolio.value(base_market, 0.0);

    MarketState shocked_market =
        apply_scenario(base_market, scenario);

    double horizon_value =
        portfolio.value(shocked_market, horizon_years);

    return horizon_value - initial_value;
}