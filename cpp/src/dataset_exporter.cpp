#include "dataset_exporter.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "scenario_application.hpp"

void export_scenario_dataset_csv(
    const Portfolio& portfolio,
    const MarketState& base_market,
    ScenarioGenerator& generator,
    double horizon_years,
    std::size_t number_of_scenarios,
    const std::string& file_path
) {
    std::ofstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open output CSV file."
        );
    }

    file << std::setprecision(17);

    const std::size_t number_of_equities =
        base_market.spots.size();

    for (std::size_t i = 0;
         i < number_of_equities;
         ++i) {

        file << "spot_log_return_" << i << ",";
    }

    for (std::size_t i = 0;
         i < number_of_equities;
         ++i) {

        file << "volatility_shock_" << i << ",";
    }

    file << "rate_shock,pnl\n";

    double initial_value =
        portfolio.value(
            base_market,
            0.0
        );

    for (std::size_t scenario_index = 0;
         scenario_index < number_of_scenarios;
         ++scenario_index) {

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

        for (double value :
             scenario.spot_log_returns) {

            file << value << ",";
        }

        for (double value :
             scenario.volatility_shocks) {

            file << value << ",";
        }

        file << scenario.rate_shock
             << ","
             << pnl
             << "\n";
    }
}