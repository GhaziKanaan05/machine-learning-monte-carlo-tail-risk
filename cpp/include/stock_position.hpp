#pragma once

#include <cstddef>

struct StockPosition {
    std::size_t underlying_index;
    double quantity;
};