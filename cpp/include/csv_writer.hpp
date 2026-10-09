#pragma once

#include <string>
#include <vector>

#include "scenario_result.hpp"

void write_scenario_dataset_csv(
    const std::vector<ScenarioResult>& results,
    const std::string& file_path
);