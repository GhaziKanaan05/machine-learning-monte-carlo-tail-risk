#include "dataset_generator.hpp"
#include "scenario_application.hpp"

std::vector<ScenarioResult> generate_scenario_dataset(
    const Portfolio& portfolio,
    const MarketState& base_market,
    ScenarioGenerator& generator,
    double horizon_years,
    std::size_t number_of_scenarios
) {
    std::vector<ScenarioResult> results;
    results.reserve(number_of_scenarios);

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

        results.push_back({
            scenario,
            pnl
        });
    }

    return results;
}