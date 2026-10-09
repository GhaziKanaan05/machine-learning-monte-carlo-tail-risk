#include "european_option.hpp"
#include "black_scholes.hpp"

EuropeanOption::EuropeanOption(
    std::size_t underlying_index,
    double strike,
    double maturity,
    OptionType type
)
    : underlying_index_(underlying_index),
      strike_(strike),
      maturity_(maturity),
      type_(type)
{
}

double EuropeanOption::price(
    const MarketState& market,
    double time_elapsed
) const {
    double spot = market.spots[underlying_index_];
    double volatility = market.volatilities[underlying_index_];
    double dividend_yield = market.dividend_yields[underlying_index_];

    double remaining_maturity = maturity_ - time_elapsed;

    if (remaining_maturity <= 0.0) {
        if (type_ == OptionType::Call) {
            return (spot > strike_) ? (spot - strike_) : 0.0;
        }

        return (strike_ > spot) ? (strike_ - spot) : 0.0;
    }

    if (type_ == OptionType::Call) {
        return black_scholes_call(
            spot,
            strike_,
            market.risk_free_rate,
            dividend_yield,
            volatility,
            remaining_maturity
        );
    }

    return black_scholes_put(
        spot,
        strike_,
        market.risk_free_rate,
        dividend_yield,
        volatility,
        remaining_maturity
    );
}

double EuropeanOption::delta(const MarketState& market) const {
    double spot = market.spots[underlying_index_];
    double volatility = market.volatilities[underlying_index_];
    double dividend_yield = market.dividend_yields[underlying_index_];

    if (type_ == OptionType::Call) {
        return black_scholes_call_delta(
            spot,
            strike_,
            market.risk_free_rate,
            dividend_yield,
            volatility,
            maturity_
        );
    }

    return black_scholes_put_delta(
        spot,
        strike_,
        market.risk_free_rate,
        dividend_yield,
        volatility,
        maturity_
    );
}

std::size_t EuropeanOption::underlying_index() const {
    return underlying_index_;
}