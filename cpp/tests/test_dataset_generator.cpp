#include <iostream>
#include <vector>

#include "dataset_generator.hpp"

int main() {
    MarketState base_market{
        {100.0},
        {0.20},
        {0.00},
        0.05
    };

    Portfolio portfolio;

    portfolio.add_position({
        EuropeanOption(
            0,
            100.0,
            1.0,
            OptionType::Call
        ),
        1.0
    });

    double horizon_years =
        10.0 / 252.0;

    ScenarioGenerator generator_1(
        {0.20},
        {0.03},
        0.005,
        horizon_years,
        6.0,
        0.35,
        0.50,
        -0.30,
        12345
    );

    ScenarioGenerator generator_2(
        {0.20},
        {0.03},
        0.005,
        horizon_years,
        6.0,
        0.35,
        0.50,
        -0.30,
        12345
    );

    const std::size_t number_of_scenarios = 100;

    std::vector<ScenarioResult> results_1 =
        generate_scenario_dataset(
            portfolio,
            base_market,
            generator_1,
            horizon_years,
            number_of_scenarios
        );

    std::vector<ScenarioResult> results_2 =
        generate_scenario_dataset(
            portfolio,
            base_market,
            generator_2,
            horizon_years,
            number_of_scenarios
        );

    if (results_1.size() != number_of_scenarios) {
        std::cerr
            << "FAILED: Incorrect dataset size."
            << std::endl;

        return 1;
    }

    for (std::size_t i = 0;
         i < number_of_scenarios;
         ++i) {

        if (results_1[i].pnl != results_2[i].pnl) {
            std::cerr
                << "FAILED: P&L is not reproducible."
                << std::endl;

            return 1;
        }

        if (results_1[i].scenario.spot_log_returns
            != results_2[i].scenario.spot_log_returns) {

            std::cerr
                << "FAILED: Spot scenarios are not reproducible."
                << std::endl;

            return 1;
        }

        if (results_1[i].scenario.volatility_shocks
            != results_2[i].scenario.volatility_shocks) {

            std::cerr
                << "FAILED: Volatility scenarios are not reproducible."
                << std::endl;

            return 1;
        }

        if (results_1[i].scenario.rate_shock
            != results_2[i].scenario.rate_shock) {

            std::cerr
                << "FAILED: Rate scenarios are not reproducible."
                << std::endl;

            return 1;
        }
    }

    std::cout
        << "Dataset generator tests passed."
        << std::endl;

    return 0;
}