#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "market_state.hpp"
#include "monte_carlo.hpp"
#include "portfolio.hpp"
#include "risk_metrics.hpp"
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
    std::size_t option_count = 0;

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

                option_count += 2;
            }
        }
    }

    std::vector<double> hedge_quantities =
        portfolio.delta_hedge_quantities(base_market);

    for (std::size_t underlying = 0;
         underlying < hedge_quantities.size();
         ++underlying) {

        portfolio.add_stock_position({
            underlying,
            hedge_quantities[underlying]
        });
    }

    std::vector<double> net_deltas =
        portfolio.net_deltas(base_market);

    double horizon_years = 10.0 / 252.0;

    double initial_value =
        portfolio.value(base_market, 0.0);

    std::cout << "Number of equities: "
              << base_market.spots.size()
              << std::endl;

    std::cout << "Number of option positions: "
              << option_count
              << std::endl;

    std::cout << "\nNet portfolio deltas:" << std::endl;

    for (std::size_t i = 0;
         i < net_deltas.size();
         ++i) {

        std::cout << "Equity "
                  << i
                  << ": "
                  << net_deltas[i]
                  << std::endl;
    }

    std::cout << "\nInitial portfolio value: "
              << initial_value
              << std::endl;

    ScenarioGenerator generator(
        base_market.volatilities,
        {0.020, 0.025, 0.030, 0.035, 0.040},
        0.0025,
        horizon_years,
        6.0,
        0.35,
        0.50,
        -0.30,
        12345
    );

    const std::size_t number_of_scenarios = 1000000;

    auto start_time =
    std::chrono::high_resolution_clock::now();

std::vector<double> pnl_values =
    run_full_revaluation_monte_carlo(
        portfolio,
        base_market,
        generator,
        horizon_years,
        number_of_scenarios
    );

auto end_time =
    std::chrono::high_resolution_clock::now();

double monte_carlo_seconds =
    std::chrono::duration<double>(
        end_time - start_time
    ).count();

    double mean_pnl =
    std::accumulate(
        pnl_values.begin(),
        pnl_values.end(),
        0.0
    ) / pnl_values.size();

double minimum_pnl =
    *std::min_element(
        pnl_values.begin(),
        pnl_values.end()
    );

double maximum_pnl =
    *std::max_element(
        pnl_values.begin(),
        pnl_values.end()
    );

double var_99 =
    value_at_risk(
        pnl_values,
        0.99
    );

double es_99 =
    expected_shortfall(
        pnl_values,
        0.99
    );

std::cout << "\nMonte Carlo scenarios: "
          << pnl_values.size()
          << std::endl;

std::cout << "Full-revaluation runtime: "
          << monte_carlo_seconds
          << " seconds"
          << std::endl;

std::cout << "Scenarios per second: "
          << static_cast<double>(pnl_values.size())
             / monte_carlo_seconds
          << std::endl;

std::cout << "Mean 10-day P&L: "
          << mean_pnl
          << std::endl;

std::cout << "Worst 10-day P&L: "
          << minimum_pnl
          << std::endl;

std::cout << "Best 10-day P&L: "
          << maximum_pnl
          << std::endl;

std::cout << "99% VaR: "
          << var_99
          << std::endl;

std::cout << "99% Expected Shortfall: "
          << es_99
          << std::endl;

return 0;
}