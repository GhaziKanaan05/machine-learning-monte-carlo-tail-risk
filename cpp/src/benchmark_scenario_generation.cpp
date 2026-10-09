#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "scenario_generator.hpp"


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

    const std::vector<double> spot_volatilities{
        0.18,
        0.22,
        0.25,
        0.30,
        0.35
    };

    const std::vector<double> volatility_shock_scales{
        0.020,
        0.025,
        0.030,
        0.035,
        0.040
    };

    const double horizon_years =
        10.0 / 252.0;

    const double rate_shock_scale =
        0.0025;

    const double degrees_of_freedom =
        6.0;

    const double spot_correlation =
        0.35;

    const double volatility_correlation =
        0.50;

    const double spot_volatility_correlation =
        -0.30;

    const std::size_t number_of_scenarios =
        1000000;

    const std::size_t repetitions =
        5;


    // Warm-up
    {
        ScenarioGenerator generator(
            spot_volatilities,
            volatility_shock_scales,
            rate_shock_scale,
            horizon_years,
            degrees_of_freedom,
            spot_correlation,
            volatility_correlation,
            spot_volatility_correlation,
            99999
        );

        double checksum = 0.0;

        for (std::size_t i = 0;
             i < 20000;
             ++i) {

            Scenario scenario =
                generator.generate();

            checksum +=
                scenario.rate_shock;

            checksum +=
                scenario.spot_log_returns[0];

            checksum +=
                scenario.volatility_shocks[0];
        }

        std::cout
            << "Warm-up checksum: "
            << checksum
            << "\n";
    }


    std::vector<double> times;

    double grand_checksum = 0.0;


    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "\nSCENARIO GENERATION BENCHMARK\n";

    std::cout
        << "Scenarios per run: "
        << number_of_scenarios
        << "\n\n";


    for (std::size_t repetition = 0;
         repetition < repetitions;
         ++repetition) {

        ScenarioGenerator generator(
            spot_volatilities,
            volatility_shock_scales,
            rate_shock_scale,
            horizon_years,
            degrees_of_freedom,
            spot_correlation,
            volatility_correlation,
            spot_volatility_correlation,
            80000
            + static_cast<unsigned int>(
                repetition
            )
        );

        double checksum = 0.0;

        const auto start =
            std::chrono::steady_clock::now();


        for (std::size_t i = 0;
             i < number_of_scenarios;
             ++i) {

            Scenario scenario =
                generator.generate();

            checksum +=
                scenario.rate_shock;

            checksum +=
                scenario.spot_log_returns[0];

            checksum +=
                scenario.volatility_shocks[0];
        }


        const auto finish =
            std::chrono::steady_clock::now();


        const double seconds =
            std::chrono::duration<double>(
                finish - start
            ).count();


        times.push_back(
            seconds
        );

        grand_checksum +=
            checksum;


        std::cout
            << "Run "
            << repetition + 1
            << ": "
            << seconds
            << " s\n";
    }


    const double median_seconds =
        median(times);


    std::cout
        << "\nMedian: "
        << median_seconds
        << " s\n";

    std::cout
        << "Median throughput: "
        << static_cast<double>(
            number_of_scenarios
        ) / median_seconds
        << " scenarios/s\n";

    std::cout
        << "Grand checksum: "
        << grand_checksum
        << "\n";


    return 0;
}