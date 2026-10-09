#pragma once

double normal_cdf(double x);

double black_scholes_call(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
);

double black_scholes_put(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
);

double black_scholes_call_delta(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
);

double black_scholes_put_delta(
    double S,
    double K,
    double r,
    double q,
    double sigma,
    double T
);