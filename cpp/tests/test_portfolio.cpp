#include <iostream>
#include <cmath>
#include <vector>
#include "black_scholes.hpp"

#include "portfolio.hpp"

bool approximately_equal(double a, double b, double tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

int main() {
    MarketState market{
        {100.0, 120.0},
        {0.20, 0.30},
        {0.00, 0.01},
        0.05
    };

    Portfolio portfolio;

    portfolio.add_position({
        EuropeanOption(0, 100.0, 1.0, OptionType::Call),
        2.0
    });

    portfolio.add_position({
        EuropeanOption(1, 100.0, 1.0, OptionType::Call),
        -1.0
    });

    std::vector<double> deltas = portfolio.option_deltas(market);

    double expected_delta_0 =
        2.0 * black_scholes_call_delta(
            100.0, 100.0, 0.05, 0.00, 0.20, 1.0
        );

    double expected_delta_1 =
        -1.0 * black_scholes_call_delta(
            120.0, 100.0, 0.05, 0.01, 0.30, 1.0
        );

    if (!approximately_equal(deltas[0], expected_delta_0)) {
        std::cerr << "FAILED: Delta aggregation for underlying 0" << std::endl;
        return 1;
    }

    if (!approximately_equal(deltas[1], expected_delta_1)) {
        std::cerr << "FAILED: Delta aggregation for underlying 1" << std::endl;
        return 1;
    }

std::vector<double> hedges =
    portfolio.delta_hedge_quantities(market);

if (!approximately_equal(hedges[0], -expected_delta_0)) {
    std::cerr << "FAILED: Hedge quantity for underlying 0" << std::endl;
    return 1;
}

if (!approximately_equal(hedges[1], -expected_delta_1)) {
    std::cerr << "FAILED: Hedge quantity for underlying 1" << std::endl;
    return 1;
}

    std::cout << "Portfolio delta test passed." << std::endl;
    return 0;
}