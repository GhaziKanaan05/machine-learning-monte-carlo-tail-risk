#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "european_option.hpp"
#include "market_state.hpp"
#include "portfolio.hpp"
#include "scenario_application.hpp"
#include "scenario_generator.hpp"


struct TimingResult {
    double seconds;
    double checksum;
};


TimingResult run_benchmark(
    const Portfolio& portfolio,
    const MarketState& base_market,
    const std::vector<double>& volatility_shock_scales,
    double rate_shock_scale,
    double horizon_years,
    double degrees_of_freedom,
    double spot_correlation,
    double volatility_correlation,
    double spot_volatility_correlation,
    std::size_t number_of_scenarios,
    unsigned int seed
) {
    ScenarioGenerator generator(
        base_market.volatilities,
        volatility_shock_scales,
        rate_shock_scale,
        horizon_years,
        degrees_of_freedom,
        spot_correlation,
        volatility_correlation,
        spot_volatility_correlation,
        seed
    );

    // The initial portfolio value is the same for every scenario,
    // so calculate it once, exactly as in dataset_generator.cpp.
    const double initial_value =
        portfolio.value(
            base_market,
            0.0
        );

    double checksum = 0.0;

    const auto start =
        std::chrono::steady_clock::now();

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

        const double horizon_value =
            portfolio.value(
                shocked_market,
                horizon_years
            );

        const double pnl =
            horizon_value
            - initial_value;

        // Keeps the calculations observable to the compiler
        // without storing the full dataset.
        checksum += pnl;
    }

    const auto finish =
        std::chrono::steady_clock::now();

    const double seconds =
        std::chrono::duration<double>(
            finish - start
        ).count();

    return {
        seconds,
        checksum
    };
}


double median(
    std::vector<double> values
) {
    std::sort(
        values.begin(),
        values.end()
    );

    const std::size_t n =
        values.size();

    if (n % 2 == 1) {
        return values[n / 2];
    }

    return 0.5 * (
        values[n / 2 - 1]
        + values[n / 2]
    );
}


int main() {

    // -----------------------------------------------------
    // Baseline market
    // -----------------------------------------------------

    MarketState base_market{
        {100.0, 80.0, 120.0, 150.0, 60.0},
        {0.18, 0.22, 0.25, 0.30, 0.35},
        {0.01, 0.015, 0.005, 0.02, 0.00},
        0.04
    };


    // -----------------------------------------------------
    // Construct the exact 500-option portfolio used
    // throughout the project
    // -----------------------------------------------------

    const std::vector<double> strike_multipliers{
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.15,
        1.25
    };

    const std::vector<double> maturities{
        0.25,
        0.50,
        0.75,
        1.00,
        1.50
    };

    Portfolio portfolio;

    for (std::size_t underlying = 0;
         underlying < base_market.spots.size();
         ++underlying) {

        for (std::size_t strike_index = 0;
             strike_index < strike_multipliers.size();
             ++strike_index) {

            const double strike =
                base_market.spots[underlying]
                * strike_multipliers[
                    strike_index
                ];

            for (std::size_t maturity_index = 0;
                 maturity_index < maturities.size();
                 ++maturity_index) {

                const double maturity =
                    maturities[
                        maturity_index
                    ];

                const double quantity =
                    (
                        (
                            underlying
                            + strike_index
                            + maturity_index
                        ) % 2 == 0
                    )
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


    // -----------------------------------------------------
    // Delta hedge
    // -----------------------------------------------------

    const std::vector<double> hedge_quantities =
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


    // -----------------------------------------------------
    // Scenario model parameters
    // -----------------------------------------------------

    const double horizon_years =
        10.0 / 252.0;

    const std::vector<double>
        volatility_shock_scales{
            0.020,
            0.025,
            0.030,
            0.035,
            0.040
        };

    const double rate_shock_scale =
        0.0025;

    const double degrees_of_freedom =
        6.0;

    const double spot_correlation =
        0.35;

    const double volatility_correlation =
        0.50;

    const double
        spot_volatility_correlation =
            -0.30;


    // -----------------------------------------------------
    // Benchmark configuration
    // -----------------------------------------------------

    const std::vector<std::size_t>
        scenario_counts{
            2500,
            5000,
            10000,
            20000,
            50000,
            100000,
            250000,
            1000000
        };

    const std::size_t repetitions = 5;


    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "FULL-REVALUATION COMPUTE BENCHMARK\n"
        << "Scenario generation + portfolio repricing only\n"
        << "CSV I/O excluded\n"
        << "Result storage excluded\n"
        << "Timed repetitions per size: "
        << repetitions
        << "\n\n";


    // -----------------------------------------------------
    // Warm-up
    // -----------------------------------------------------

    std::cout
        << "Running 20,000-scenario warm-up..."
        << std::endl;

    const TimingResult warmup =
        run_benchmark(
            portfolio,
            base_market,
            volatility_shock_scales,
            rate_shock_scale,
            horizon_years,
            degrees_of_freedom,
            spot_correlation,
            volatility_correlation,
            spot_volatility_correlation,
            20000,
            99999
        );

    std::cout
        << "Warm-up complete: "
        << warmup.seconds
        << " seconds\n\n";


    // Grand checksum ensures benchmark results
    // remain observable.
    double grand_checksum = 0.0;


    // -----------------------------------------------------
    // Timed benchmarks
    // -----------------------------------------------------

    for (std::size_t count :
         scenario_counts) {

        std::vector<double> times;

        times.reserve(
            repetitions
        );

        std::cout
            << "========================================\n";

        std::cout
            << "Scenarios: "
            << count
            << "\n";

        for (std::size_t repetition = 0;
             repetition < repetitions;
             ++repetition) {

            const unsigned int seed =
                70000
                + static_cast<unsigned int>(
                    repetition
                );

            const TimingResult result =
                run_benchmark(
                    portfolio,
                    base_market,
                    volatility_shock_scales,
                    rate_shock_scale,
                    horizon_years,
                    degrees_of_freedom,
                    spot_correlation,
                    volatility_correlation,
                    spot_volatility_correlation,
                    count,
                    seed
                );

            times.push_back(
                result.seconds
            );

            grand_checksum +=
                result.checksum;

            std::cout
                << "Run "
                << repetition + 1
                << ": "
                << result.seconds
                << " s"
                << std::endl;
        }


        const double median_seconds =
            median(times);

        const double min_seconds =
            *std::min_element(
                times.begin(),
                times.end()
            );

        const double max_seconds =
            *std::max_element(
                times.begin(),
                times.end()
            );

        const double scenarios_per_second =
            static_cast<double>(count)
            / median_seconds;


        std::cout
            << "\nMedian: "
            << median_seconds
            << " s\n";

        std::cout
            << "Min:    "
            << min_seconds
            << " s\n";

        std::cout
            << "Max:    "
            << max_seconds
            << " s\n";

        std::cout
            << "Median throughput: "
            << scenarios_per_second
            << " scenarios/s\n\n";
    }


    std::cout
        << "Grand checksum: "
        << grand_checksum
        << "\n";

    return 0;
}