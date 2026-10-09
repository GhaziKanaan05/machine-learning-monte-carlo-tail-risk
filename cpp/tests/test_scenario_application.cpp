#include <cmath>
#include <iostream>

#include "scenario_application.hpp"

bool approximately_equal(
    double a,
    double b,
    double tolerance = 1e-10
) {
    return std::abs(a - b) < tolerance;
}

int main() {
    MarketState base_market{
        {100.0, 80.0},
        {0.20, 0.30},
        {0.01, 0.02},
        0.04
    };

    Scenario scenario{
        {0.10, -0.05},
        {0.02, -0.04},
        0.01
    };

    MarketState shocked_market =
        apply_scenario(base_market, scenario);

    if (!approximately_equal(
            shocked_market.spots[0],
            100.0 * std::exp(0.10))) {

        std::cerr << "FAILED: Spot 0 shock" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            shocked_market.spots[1],
            80.0 * std::exp(-0.05))) {

        std::cerr << "FAILED: Spot 1 shock" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            shocked_market.volatilities[0],
            0.22)) {

        std::cerr << "FAILED: Volatility 0 shock" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            shocked_market.volatilities[1],
            0.26)) {

        std::cerr << "FAILED: Volatility 1 shock" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            shocked_market.risk_free_rate,
            0.05)) {

        std::cerr << "FAILED: Rate shock" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            shocked_market.dividend_yields[0],
            0.01)) {

        std::cerr << "FAILED: Dividend yield changed" << std::endl;
        return 1;
    }

    std::cout << "Scenario application tests passed."
              << std::endl;

    return 0;
}