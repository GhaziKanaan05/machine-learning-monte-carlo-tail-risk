#include "csv_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

void write_scenario_dataset_csv(
    const std::vector<ScenarioResult>& results,
    const std::string& file_path
) {
    if (results.empty()) {
        throw std::invalid_argument(
            "Cannot write an empty scenario dataset."
        );
    }

    const std::size_t number_of_spot_factors =
        results.front().scenario.spot_log_returns.size();

    const std::size_t number_of_volatility_factors =
        results.front().scenario.volatility_shocks.size();

    std::ofstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open output CSV file."
        );
    }

    file << std::setprecision(17);

    for (std::size_t i = 0;
         i < number_of_spot_factors;
         ++i) {

        file << "spot_log_return_" << i << ",";
    }

    for (std::size_t i = 0;
         i < number_of_volatility_factors;
         ++i) {

        file << "volatility_shock_" << i << ",";
    }

    file << "rate_shock,pnl\n";

    for (const ScenarioResult& result : results) {

        if (result.scenario.spot_log_returns.size()
            != number_of_spot_factors) {

            throw std::runtime_error(
                "Inconsistent spot factor dimension."
            );
        }

        if (result.scenario.volatility_shocks.size()
            != number_of_volatility_factors) {

            throw std::runtime_error(
                "Inconsistent volatility factor dimension."
            );
        }

        for (double value :
             result.scenario.spot_log_returns) {

            file << value << ",";
        }

        for (double value :
             result.scenario.volatility_shocks) {

            file << value << ",";
        }

        file << result.scenario.rate_shock
             << ","
             << result.pnl
             << "\n";
    }
}