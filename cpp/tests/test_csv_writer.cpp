#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "csv_writer.hpp"

int main() {
    Scenario scenario_1{
        {0.01, -0.02},
        {0.03, -0.01},
        0.001
    };

    Scenario scenario_2{
        {-0.04, 0.05},
        {-0.02, 0.04},
        -0.002
    };

    std::vector<ScenarioResult> results{
        {scenario_1, -1.25},
        {scenario_2, 2.50}
    };

    const std::string file_path =
        "test_scenario_dataset.csv";

    write_scenario_dataset_csv(
        results,
        file_path
    );

    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr
            << "FAILED: CSV file was not created."
            << std::endl;

        return 1;
    }

    std::string header;
    std::getline(file, header);

    const std::string expected_header =
        "spot_log_return_0,"
        "spot_log_return_1,"
        "volatility_shock_0,"
        "volatility_shock_1,"
        "rate_shock,pnl";

    if (header != expected_header) {
        std::cerr
            << "FAILED: Incorrect CSV header."
            << std::endl;

        return 1;
    }

    std::string first_data_row;
    std::getline(file, first_data_row);

    if (first_data_row.empty()) {
        std::cerr
            << "FAILED: First data row is empty."
            << std::endl;

        return 1;
    }

    std::string second_data_row;
    std::getline(file, second_data_row);

    if (second_data_row.empty()) {
        std::cerr
            << "FAILED: Second data row is empty."
            << std::endl;

        return 1;
    }

    file.close();

    std::remove(
        file_path.c_str()
    );

    std::cout
        << "CSV writer tests passed."
        << std::endl;

    return 0;
}