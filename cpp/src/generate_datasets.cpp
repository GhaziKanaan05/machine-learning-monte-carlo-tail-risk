#include <filesystem>
#include <iostream>
#include <vector>

#include "dataset_exporter.hpp"
#include "market_state.hpp"
#include "portfolio.hpp"
#include "scenario_generator.hpp"

int main() {
    MarketState base_market{
        {100.0, 80.0, 120.0, 150.0, 60.0},
        {0.18, 0.22, 0.25, 0.30, 0.35},
        {0.01, 0.015, 0.005, 0.02, 0.00},
        0.04
    };

    std::vector<double> strike_multipliers{
        0.75, 0.80, 0.85, 0.90, 0.95,
        1.00, 1.05, 1.10, 1.15, 1.25
    };

    std::vector<double> maturities{
        0.25, 0.50, 0.75, 1.00, 1.50
    };

    Portfolio portfolio;

    for (std::size_t underlying = 0;
         underlying < base_market.spots.size();
         ++underlying) {

        for (std::size_t strike_index = 0;
             strike_index < strike_multipliers.size();
             ++strike_index) {

            double strike =
                base_market.spots[underlying]
                * strike_multipliers[strike_index];

            for (std::size_t maturity_index = 0;
                 maturity_index < maturities.size();
                 ++maturity_index) {

                double maturity =
                    maturities[maturity_index];

                double quantity =
                    ((underlying
                      + strike_index
                      + maturity_index) % 2 == 0)
                    ? 1.0
                    : -1.0;

                portfolio.add_position({
                    EuropeanOption(
                        underlying,
                        strike,
                        maturity,
                        OptionType::Call
                    ),
                    quantity
                });

                portfolio.add_position({
                    EuropeanOption(
                        underlying,
                        strike,
                        maturity,
                        OptionType::Put
                    ),
                    quantity
                });
            }
        }
    }

    std::vector<double> hedge_quantities =
        portfolio.delta_hedge_quantities(
            base_market
        );

    for (std::size_t underlying = 0;
         underlying < hedge_quantities.size();
         ++underlying) {

        portfolio.add_stock_position({
            underlying,
            hedge_quantities[underlying]
        });
    }

    const double horizon_years =
        10.0 / 252.0;

    const std::vector<double> volatility_shock_scales{
        0.020,
        0.025,
        0.030,
        0.035,
        0.040
    };

    const double rate_shock_scale = 0.0025;
    const double degrees_of_freedom = 6.0;
    const double spot_correlation = 0.35;
    const double volatility_correlation = 0.50;
    const double spot_volatility_correlation = -0.30;

    std::filesystem::create_directories(
        "data"
    );

    std::cout
        << "Generating 50,000-scenario training pool..."
        << std::endl;

    ScenarioGenerator training_generator(
        base_market.volatilities,
        volatility_shock_scales,
        rate_shock_scale,
        horizon_years,
        degrees_of_freedom,
        spot_correlation,
        volatility_correlation,
        spot_volatility_correlation,
        54321
    );

    export_scenario_dataset_csv(
        portfolio,
        base_market,
        training_generator,
        horizon_years,
        50000,
        "data/training_pool_50000.csv"
    );

    std::cout
        << "Training pool complete."
        << std::endl;

std::cout
    << "Generating 50,000-scenario validation set..."
    << std::endl;

ScenarioGenerator validation_generator(
    base_market.volatilities,
    volatility_shock_scales,
    rate_shock_scale,
    horizon_years,
    degrees_of_freedom,
    spot_correlation,
    volatility_correlation,
    spot_volatility_correlation,
    67890
);

export_scenario_dataset_csv(
    portfolio,
    base_market,
    validation_generator,
    horizon_years,
    50000,
    "data/validation_50000.csv"
);

std::cout
    << "Validation set complete."
    << std::endl;

    std::cout
        << "Generating 1,000,000-scenario benchmark set..."
        << std::endl;

    ScenarioGenerator benchmark_generator(
        base_market.volatilities,
        volatility_shock_scales,
        rate_shock_scale,
        horizon_years,
        degrees_of_freedom,
        spot_correlation,
        volatility_correlation,
        spot_volatility_correlation,
        12345
    );

    export_scenario_dataset_csv(
        portfolio,
        base_market,
        benchmark_generator,
        horizon_years,
        1000000,
        "data/benchmark_1000000.csv"
    );

    std::cout
        << "Benchmark set complete."
        << std::endl;

    std::cout
        << "Dataset generation finished."
        << std::endl;

    return 0;
}