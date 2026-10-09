#include <cmath>
#include <iostream>
#include <vector>

#include "scenario_generator.hpp"

double mean(const std::vector<double>& x) {
    double total = 0.0;

    for (double value : x) {
        total += value;
    }

    return total / x.size();
}

double standard_deviation(const std::vector<double>& x) {
    double m = mean(x);
    double total = 0.0;

    for (double value : x) {
        double difference = value - m;
        total += difference * difference;
    }

    return std::sqrt(total / (x.size() - 1));
}

double correlation(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    double mean_x = mean(x);
    double mean_y = mean(y);

    double covariance = 0.0;
    double variance_x = 0.0;
    double variance_y = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;

        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }

    return covariance /
        std::sqrt(variance_x * variance_y);
}

bool approximately_equal(
    double value,
    double target,
    double tolerance
) {
    return std::abs(value - target) < tolerance;
}

int main() {
    double horizon_years = 10.0 / 252.0;

    ScenarioGenerator generator(
        {0.20, 0.25},
        {0.03, 0.04},
        0.005,
        horizon_years,
        6.0,
        0.35,
        0.50,
        -0.30,
        12345
    );

    const std::size_t number_of_scenarios = 100000;

    std::vector<double> spot_0;
    std::vector<double> spot_1;
    std::vector<double> vol_0;
    std::vector<double> vol_1;
    std::vector<double> rates;

    spot_0.reserve(number_of_scenarios);
    spot_1.reserve(number_of_scenarios);
    vol_0.reserve(number_of_scenarios);
    vol_1.reserve(number_of_scenarios);
    rates.reserve(number_of_scenarios);

    for (std::size_t i = 0; i < number_of_scenarios; ++i) {
        Scenario scenario = generator.generate();

        if (scenario.spot_log_returns.size() != 2 ||
            scenario.volatility_shocks.size() != 2) {

            std::cerr << "FAILED: Scenario dimensions" << std::endl;
            return 1;
        }

        spot_0.push_back(scenario.spot_log_returns[0]);
        spot_1.push_back(scenario.spot_log_returns[1]);

        vol_0.push_back(scenario.volatility_shocks[0]);
        vol_1.push_back(scenario.volatility_shocks[1]);

        rates.push_back(scenario.rate_shock);
    }

    double expected_spot_0_std =
        0.20 * std::sqrt(horizon_years);

    if (!approximately_equal(
            standard_deviation(spot_0),
            expected_spot_0_std,
            0.002)) {

        std::cerr << "FAILED: Spot scale" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            standard_deviation(vol_0),
            0.03,
            0.002)) {

        std::cerr << "FAILED: Volatility shock scale" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            standard_deviation(rates),
            0.005,
            0.0005)) {

        std::cerr << "FAILED: Rate shock scale" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            correlation(spot_0, spot_1),
            0.35,
            0.03)) {

        std::cerr << "FAILED: Spot correlation" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            correlation(vol_0, vol_1),
            0.50,
            0.03)) {

        std::cerr << "FAILED: Volatility correlation" << std::endl;
        return 1;
    }

    if (!approximately_equal(
            correlation(spot_0, vol_0),
            -0.30,
            0.03)) {

        std::cerr << "FAILED: Spot-volatility correlation" << std::endl;
        return 1;
    }

    std::cout << "Scenario generator statistical tests passed."
              << std::endl;

    return 0;
}
