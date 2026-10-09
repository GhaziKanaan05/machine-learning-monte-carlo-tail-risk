#include <cmath>
#include <iostream>
#include <vector>

#include "risk_metrics.hpp"

bool approximately_equal(
    double a,
    double b,
    double tolerance = 1e-10
) {
    return std::abs(a - b) < tolerance;
}

int main() {
    std::vector<double> pnl_values{
        -10.0,
        -8.0,
        -3.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0
    };

    double var_80 =
        value_at_risk(
            pnl_values,
            0.80
        );

    double es_80 =
        expected_shortfall(
            pnl_values,
            0.80
        );

    if (!approximately_equal(var_80, 8.0)) {
        std::cerr
            << "FAILED: VaR calculation"
            << std::endl;

        return 1;
    }

    if (!approximately_equal(es_80, 9.0)) {
        std::cerr
            << "FAILED: Expected Shortfall calculation"
            << std::endl;

        return 1;
    }

    std::vector<double> large_pnl_values;

    large_pnl_values.reserve(10000);

    for (int loss = 1;
         loss <= 10000;
         ++loss) {

        large_pnl_values.push_back(
            -static_cast<double>(loss)
        );
    }

    double var_99 =
        value_at_risk(
            large_pnl_values,
            0.99
        );

    double es_99 =
        expected_shortfall(
            large_pnl_values,
            0.99
        );

    if (!approximately_equal(var_99, 9901.0)) {
        std::cerr
            << "FAILED: 99% VaR tail-count calculation"
            << std::endl;

        return 1;
    }

    if (!approximately_equal(es_99, 9950.5)) {
        std::cerr
            << "FAILED: 99% ES tail-count calculation"
            << std::endl;

        return 1;
    }

    std::cout
        << "Risk metric tests passed."
        << std::endl;

    return 0;
}