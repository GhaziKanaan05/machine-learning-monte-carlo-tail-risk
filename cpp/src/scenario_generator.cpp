#include "scenario_generator.hpp"

#include <cmath>
#include <stdexcept>

ScenarioGenerator::ScenarioGenerator(
    const std::vector<double>& spot_volatilities,
    const std::vector<double>& volatility_shock_scales,
    double rate_shock_scale,
    double horizon_years,
    double degrees_of_freedom,
    double spot_correlation,
    double volatility_correlation,
    double spot_volatility_correlation,
    unsigned int seed
)
    : spot_volatilities_(spot_volatilities),
      volatility_shock_scales_(volatility_shock_scales),
      rate_shock_scale_(rate_shock_scale),
      horizon_years_(horizon_years),
      degrees_of_freedom_(degrees_of_freedom),
      spot_correlation_(spot_correlation),
      volatility_correlation_(volatility_correlation),
      spot_volatility_correlation_(spot_volatility_correlation),
      rng_(seed)
{
    if (spot_volatilities_.size() != volatility_shock_scales_.size()) {
        throw std::invalid_argument(
            "Spot and volatility vectors must have the same size."
        );
    }

    if (degrees_of_freedom_ <= 2.0) {
        throw std::invalid_argument(
            "Degrees of freedom must be greater than 2."
        );
    }

    if (spot_correlation_ <= 0.0 || spot_correlation_ >= 1.0) {
        throw std::invalid_argument(
            "Spot correlation must be between 0 and 1."
        );
    }

    if (volatility_correlation_ < 0.0 ||
        volatility_correlation_ >= 1.0) {
        throw std::invalid_argument(
            "Volatility correlation must be between 0 and 1."
        );
    }

    double spot_vol_loading =
        spot_volatility_correlation_
        / std::sqrt(spot_correlation_);

    if (spot_vol_loading * spot_vol_loading >
        volatility_correlation_) {
        throw std::invalid_argument(
            "Correlation parameters are not compatible."
        );
    }
}

Scenario ScenarioGenerator::generate() {
    std::normal_distribution<double> normal(0.0, 1.0);

    std::chi_squared_distribution<double> chi_squared(
        degrees_of_freedom_
    );

    double spot_common_factor = normal(rng_);
    double volatility_common_factor = normal(rng_);
    double rate_factor = normal(rng_);

    double spot_common_loading =
        std::sqrt(spot_correlation_);

    double spot_idiosyncratic_loading =
        std::sqrt(1.0 - spot_correlation_);

    double spot_vol_loading =
        spot_volatility_correlation_
        / spot_common_loading;

    double volatility_extra_loading =
        std::sqrt(
            volatility_correlation_
            - spot_vol_loading * spot_vol_loading
        );

    double volatility_idiosyncratic_loading =
        std::sqrt(1.0 - volatility_correlation_);

    double chi_squared_draw = chi_squared(rng_);

    double student_scale =
        std::sqrt(
            (degrees_of_freedom_ - 2.0)
            / chi_squared_draw
        );

    Scenario scenario;

    scenario.spot_log_returns.reserve(
        spot_volatilities_.size()
    );

    scenario.volatility_shocks.reserve(
        volatility_shock_scales_.size()
    );

    for (std::size_t i = 0;
         i < spot_volatilities_.size();
         ++i) {

        double spot_idiosyncratic = normal(rng_);
        double volatility_idiosyncratic = normal(rng_);

        double standardized_spot =
            spot_common_loading * spot_common_factor
            + spot_idiosyncratic_loading
                * spot_idiosyncratic;

        double standardized_volatility =
            spot_vol_loading * spot_common_factor
            + volatility_extra_loading
                * volatility_common_factor
            + volatility_idiosyncratic_loading
                * volatility_idiosyncratic;

        double spot_return =
            spot_volatilities_[i]
            * std::sqrt(horizon_years_)
            * standardized_spot
            * student_scale;

        double volatility_shock =
            volatility_shock_scales_[i]
            * standardized_volatility
            * student_scale;

        scenario.spot_log_returns.push_back(
            spot_return
        );

        scenario.volatility_shocks.push_back(
            volatility_shock
        );
    }

    scenario.rate_shock =
        rate_shock_scale_
        * rate_factor
        * student_scale;

    return scenario;
}
