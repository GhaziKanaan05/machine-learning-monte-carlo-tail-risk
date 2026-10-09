#include "portfolio.hpp"

void Portfolio::add_position(const Position& position) {
    positions_.push_back(position);
}

void Portfolio::add_stock_position(const StockPosition& position) {
    stock_positions_.push_back(position);
}

double Portfolio::value(
    const MarketState& market,
    double time_elapsed
) const {
    double total = 0.0;

    for (const Position& position : positions_) {
        total += position.quantity
               * position.option.price(market, time_elapsed);
    }

    for (const StockPosition& position : stock_positions_) {
        total += position.quantity
               * market.spots[position.underlying_index];
    }

    return total;
}

std::vector<double> Portfolio::option_deltas(
    const MarketState& market
) const {
    std::vector<double> deltas(market.spots.size(), 0.0);

    for (const Position& position : positions_) {
        std::size_t index = position.option.underlying_index();

        deltas[index] +=
            position.quantity * position.option.delta(market);
    }

    return deltas;
}

std::vector<double> Portfolio::delta_hedge_quantities(
    const MarketState& market
) const {
    std::vector<double> deltas = option_deltas(market);

    for (double& delta : deltas) {
        delta = -delta;
    }

    return deltas;
}

std::vector<double> Portfolio::net_deltas(
    const MarketState& market
) const {
    std::vector<double> deltas = option_deltas(market);

    for (const StockPosition& position : stock_positions_) {
        deltas[position.underlying_index] += position.quantity;
    }

    return deltas;
}