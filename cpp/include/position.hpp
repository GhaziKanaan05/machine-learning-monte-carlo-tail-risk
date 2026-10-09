#pragma once

#include "european_option.hpp"

struct Position {
    EuropeanOption option;
    double quantity;
};