#include "risk_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

std::vector<double> sorted_losses(
    const std::vector<double>& pnl_values
) {
    if (pnl_values.empty()) {
        throw std::invalid_argument(
            "P&L vector must not be empty."
        );
    }

    std::vector<double> losses;
    losses.reserve(pnl_values.size());

    for (double pnl : pnl_values) {
        losses.push_back(-pnl);
    }

    std::sort(
        losses.begin(),
        losses.end()
    );

    return losses;
}

std::size_t tail_count(
    std::size_t number_of_observations,
    double confidence_level
) {
    if (confidence_level <= 0.0 ||
        confidence_level >= 1.0) {

        throw std::invalid_argument(
            "Confidence level must be between 0 and 1."
        );
    }

    double raw_tail_count =
        (1.0 - confidence_level)
        * static_cast<double>(
            number_of_observations
        );

    double nearest_integer =
        std::round(raw_tail_count);

    if (std::abs(
            raw_tail_count
            - nearest_integer
        )
        < 1e-10
          * std::max(
                1.0,
                std::abs(raw_tail_count)
            )) {

        raw_tail_count =
            nearest_integer;
    }

    std::size_t count =
        static_cast<std::size_t>(
            std::ceil(raw_tail_count)
        );

    if (count < 1) {
        count = 1;
    }

    return count;
}

}

double value_at_risk(
    const std::vector<double>& pnl_values,
    double confidence_level
) {
    std::vector<double> losses =
        sorted_losses(pnl_values);

    std::size_t count =
        tail_count(
            losses.size(),
            confidence_level
        );

    std::size_t index =
        losses.size() - count;

    return losses[index];
}

double expected_shortfall(
    const std::vector<double>& pnl_values,
    double confidence_level
) {
    std::vector<double> losses =
        sorted_losses(pnl_values);

    std::size_t count =
        tail_count(
            losses.size(),
            confidence_level
        );

    std::size_t start_index =
        losses.size() - count;

    double total_tail_loss = 0.0;

    for (std::size_t i = start_index;
         i < losses.size();
         ++i) {

        total_tail_loss += losses[i];
    }

    return total_tail_loss
        / static_cast<double>(count);
}