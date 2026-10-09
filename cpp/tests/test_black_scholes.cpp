#include <iostream>
#include <cmath>
#include "black_scholes.hpp"

bool approximately_equal(double a, double b, double tolerance = 1e-5) {
    return std::abs(a - b) < tolerance;
}

int main() {
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.00;
    double sigma = 0.20;
    double T = 1.0;

    double call = black_scholes_call(S, K, r, q, sigma, T);
    double put = black_scholes_put(S, K, r, q, sigma, T);

    if (!approximately_equal(call, 10.45058)) {
        std::cerr << "FAILED: Call price test" << std::endl;
        return 1;
    }

    if (!approximately_equal(put, 5.57353)) {
        std::cerr << "FAILED: Put price test" << std::endl;
        return 1;
    }

    double lhs = call - put;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);

    if (!approximately_equal(lhs, rhs)) {
        std::cerr << "FAILED: Put-call parity test" << std::endl;
        return 1;
    }

double call_delta = black_scholes_call_delta(
    S, K, r, q, sigma, T
);

double put_delta = black_scholes_put_delta(
    S, K, r, q, sigma, T
);

if (!approximately_equal(call_delta, 0.63683)) {
    std::cerr << "FAILED: Call delta test" << std::endl;
    return 1;
}

if (!approximately_equal(put_delta, -0.36317)) {
    std::cerr << "FAILED: Put delta test" << std::endl;
    return 1;
}

    std::cout << "All Black-Scholes tests passed." << std::endl;
    return 0;
}