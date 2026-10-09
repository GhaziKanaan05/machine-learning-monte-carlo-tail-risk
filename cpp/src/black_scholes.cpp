#include "black_scholes.hpp"
#include <cmath>

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double black_scholes_call(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
) {
    double d1 = (
        std::log(S / K)
        + (r - q + 0.5 * sigma * sigma) * T
    ) / (sigma * std::sqrt(T));

    double d2 = d1 - sigma * std::sqrt(T);

    return S * std::exp(-q * T) * normal_cdf(d1)
         - K * std::exp(-r * T) * normal_cdf(d2);
}

double black_scholes_put(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
) {
    double d1 = (
        std::log(S / K)
        + (r - q + 0.5 * sigma * sigma) * T
    ) / (sigma * std::sqrt(T));

    double d2 = d1 - sigma * std::sqrt(T);

    return K * std::exp(-r * T) * normal_cdf(-d2)
         - S * std::exp(-q * T) * normal_cdf(-d1);
}

double black_scholes_call_delta(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
) {
    double d1 = (
        std::log(S / K)
        + (r - q + 0.5 * sigma * sigma) * T
    ) / (sigma * std::sqrt(T));

    return std::exp(-q * T) * normal_cdf(d1);
}

double black_scholes_put_delta(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
) {
    double d1 = (
        std::log(S / K)
        + (r - q + 0.5 * sigma * sigma) * T
    ) / (sigma * std::sqrt(T));

    return std::exp(-q * T) * (normal_cdf(d1) - 1.0);
}