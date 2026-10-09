#include <cmath>
#include <iostream>

#include "black_scholes.hpp"
#include "full_revaluation.hpp"

bool approximately_equal(
    double a,
    double b,
    double tolerance = 1e-10
) {
    return std::abs(a - b) < tolerance;
}

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

    Scenario scenario{
        {std::log(1.10)},
        {0.01},
        0.005
    };

    double horizon_years = 10.0 / 252.0;

    double actual_pnl =
        full_revaluation_pnl(
            portfolio,
            base_market,
            scenario,
            horizon_years
        );

    double initial_value =
        black_scholes_call(
            100.0,
            100.0,
            0.05,
            0.00,
            0.20,
            1.0
        );

    double horizon_value =
        black_scholes_call(
            110.0,
            100.0,
            0.055,
            0.00,
            0.21,
            1.0 - horizon_years
        );

    double expected_pnl =
        horizon_value - initial_value;

    if (!approximately_equal(
            actual_pnl,
            expected_pnl)) {

        std::cerr
            << "FAILED: Full revaluation P&L"
            << std::endl;

        return 1;
    }

    std::cout
        << "Full revaluation test passed."
        << std::endl;

    return 0;
}