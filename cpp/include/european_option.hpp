#pragma once

#include <cstddef>
#include "market_state.hpp"

enum class OptionType {
    Call,
    Put
};

class EuropeanOption {
public:
    EuropeanOption(
        std::size_t underlying_index,
        double strike,
        double maturity,
        OptionType type
    );

    double price(
        const MarketState& market,
        double time_elapsed = 0.0
    ) const;

    double delta(const MarketState& market) const;

    std::size_t underlying_index() const;

private:
    std::size_t underlying_index_;
    double strike_;
    double maturity_;
    OptionType type_;
};