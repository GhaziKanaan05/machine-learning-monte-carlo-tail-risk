#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "dataset_exporter.hpp"

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

    ScenarioGenerator generator(
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

    const std::string file_path =
        "test_streamed_dataset.csv";

    const std::size_t number_of_scenarios = 10;

    export_scenario_dataset_csv(
        portfolio,
        base_market,
        generator,
        horizon_years,
        number_of_scenarios,
        file_path
    );

    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr
            << "FAILED: Dataset file was not created."
            << std::endl;

        return 1;
    }

    std::string line;

    std::getline(file, line);

    const std::string expected_header =
        "spot_log_return_0,"
        "volatility_shock_0,"
        "rate_shock,pnl";

    if (line != expected_header) {
        std::cerr
            << "FAILED: Incorrect CSV header."
            << std::endl;

        return 1;
    }

    std::size_t data_row_count = 0;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            ++data_row_count;
        }
    }

    if (data_row_count != number_of_scenarios) {
        std::cerr
            << "FAILED: Incorrect number of data rows."
            << std::endl;

        return 1;
    }

    file.close();

    std::remove(
        file_path.c_str()
    );

    std::cout
        << "Dataset exporter tests passed."
        << std::endl;

    return 0;
}