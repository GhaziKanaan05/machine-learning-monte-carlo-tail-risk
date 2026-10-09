#include <iostream>
#include <vector>

#include "monte_carlo.hpp"

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

    double horizon_years = 10.0 / 252.0;

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

    const std::size_t number_of_scenarios = 1000;

    std::vector<double> pnl_1 =
        run_full_revaluation_monte_carlo(
            portfolio,
            base_market,
            generator_1,
            horizon_years,
            number_of_scenarios
        );

    std::vector<double> pnl_2 =
        run_full_revaluation_monte_carlo(
            portfolio,
            base_market,
            generator_2,
            horizon_years,
            number_of_scenarios
        );

    if (pnl_1.size() != number_of_scenarios) {
        std::cerr << "FAILED: Incorrect Monte Carlo size"
                  << std::endl;
        return 1;
    }

    if (pnl_1 != pnl_2) {
        std::cerr << "FAILED: Monte Carlo is not reproducible"
                  << std::endl;
        return 1;
    }

    std::cout << "Monte Carlo tests passed."
              << std::endl;

    return 0;
}