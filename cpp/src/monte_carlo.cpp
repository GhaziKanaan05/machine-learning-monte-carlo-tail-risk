#include "monte_carlo.hpp"
#include "scenario_application.hpp"

std::vector<double> run_full_revaluation_monte_carlo(
    const Portfolio& portfolio,
    const MarketState& base_market,
    ScenarioGenerator& generator,
    double horizon_years,
    std::size_t number_of_scenarios
) {
    std::vector<double> pnl_values;
    pnl_values.reserve(number_of_scenarios);

    double initial_value =
        portfolio.value(
            base_market,
            0.0
        );

    for (std::size_t i = 0;
         i < number_of_scenarios;
         ++i) {

        Scenario scenario =
            generator.generate();

        MarketState shocked_market =
            apply_scenario(
                base_market,
                scenario
            );

        double horizon_value =
            portfolio.value(
                shocked_market,
                horizon_years
            );

        double pnl =
            horizon_value
            - initial_value;

        pnl_values.push_back(pnl);
    }

    return pnl_values;
}