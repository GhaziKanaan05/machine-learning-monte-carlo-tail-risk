#pragma once

#include <vector>

struct MarketState {
    std::vector<double> spots;
    std::vector<double> volatilities;
    std::vector<double> dividend_yields;
    double risk_free_rate;
};