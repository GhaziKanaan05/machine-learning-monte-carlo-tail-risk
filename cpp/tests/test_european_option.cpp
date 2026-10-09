#include <iostream>
#include <cmath>

#include "european_option.hpp"

bool approximately_equal(double a, double b, double tolerance = 1e-5) {
    return std::abs(a - b) < tolerance;
}

int main() {
    MarketState market{
        {100.0, 120.0},       // spots
        {0.20, 0.30},         // volatilities
        {0.00, 0.01},         // dividend yields
        0.05                  // risk-free rate
    };

    EuropeanOption call_1(
        0,
        100.0,
        1.0,
        OptionType::Call
    );

    EuropeanOption call_2(
        1,
        100.0,
        1.0,
        OptionType::Call
    );

    double price_1 = call_1.price(market);
    double price_2 = call_2.price(market);

    if (!approximately_equal(price_1, 10.45058)) {
        std::cerr << "FAILED: First underlying price test" << std::endl;
        return 1;
    }

    if (!approximately_equal(price_2, 27.90374)) {
        std::cerr << "FAILED: Second underlying price test" << std::endl;
        return 1;
    }

    std::cout << "EuropeanOption tests passed." << std::endl;
    return 0;
}