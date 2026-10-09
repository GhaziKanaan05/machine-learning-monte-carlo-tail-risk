#pragma once

#include <vector>

#include "position.hpp"
#include "stock_position.hpp"
#include "market_state.hpp"

class Portfolio {
public:
    void add_position(const Position& position);
    void add_stock_position(const StockPosition& position);

    double value(
        const MarketState& market,
        double time_elapsed = 0.0
    ) const;

    std::vector<double> option_deltas(const MarketState& market) const;
    std::vector<double> delta_hedge_quantities(const MarketState& market) const;
    std::vector<double> net_deltas(const MarketState& market) const;

private:
    std::vector<Position> positions_;
    std::vector<StockPosition> stock_positions_;
};