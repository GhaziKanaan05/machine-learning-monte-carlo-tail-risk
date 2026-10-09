#pragma once

#include <vector>

double value_at_risk(
    const std::vector<double>& pnl_values,
    double confidence_level
);

double expected_shortfall(
    const std::vector<double>& pnl_values,
    double confidence_level
);