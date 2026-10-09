#pragma once

#include <vector>

struct Scenario {
    std::vector<double> spot_log_returns;
    std::vector<double> volatility_shocks;
    double rate_shock;
};
