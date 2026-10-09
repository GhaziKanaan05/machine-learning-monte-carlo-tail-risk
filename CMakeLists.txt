cmake_minimum_required(VERSION 3.20)

project(TailRiskSurrogate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tail_risk
    cpp/src/main.cpp
    cpp/src/black_scholes.cpp
    cpp/src/european_option.cpp
    cpp/src/portfolio.cpp
    cpp/src/scenario_generator.cpp
    cpp/src/scenario_application.cpp
    cpp/src/full_revaluation.cpp
    cpp/src/monte_carlo.cpp
    cpp/src/risk_metrics.cpp
    cpp/src/dataset_generator.cpp
    cpp/src/csv_writer.cpp
    cpp/src/dataset_exporter.cpp
)

target_include_directories(tail_risk PRIVATE
    cpp/include
)

enable_testing()

add_executable(test_black_scholes
    cpp/tests/test_black_scholes.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(test_black_scholes PRIVATE
    cpp/include
)

add_test(
    NAME BlackScholesTests
    COMMAND test_black_scholes
)

add_executable(test_european_option
    cpp/tests/test_european_option.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(test_european_option PRIVATE
    cpp/include
)

add_test(
    NAME EuropeanOptionTests
    COMMAND test_european_option
)

add_executable(test_portfolio
    cpp/tests/test_portfolio.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(test_portfolio PRIVATE
    cpp/include
)

add_test(
    NAME PortfolioTests
    COMMAND test_portfolio
)

add_executable(test_scenario_generator
    cpp/tests/test_scenario_generator.cpp
    cpp/src/scenario_generator.cpp
)

target_include_directories(test_scenario_generator PRIVATE
    cpp/include
)

add_test(
    NAME ScenarioGeneratorTests
    COMMAND test_scenario_generator
)

add_executable(test_scenario_application
    cpp/tests/test_scenario_application.cpp
    cpp/src/scenario_application.cpp
)

target_include_directories(test_scenario_application PRIVATE
    cpp/include
)

add_test(
    NAME ScenarioApplicationTests
    COMMAND test_scenario_application
)

add_executable(test_full_revaluation
    cpp/tests/test_full_revaluation.cpp
    cpp/src/full_revaluation.cpp
    cpp/src/scenario_application.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(test_full_revaluation PRIVATE
    cpp/include
)

add_test(
    NAME FullRevaluationTests
    COMMAND test_full_revaluation
)

add_executable(test_monte_carlo
    cpp/tests/test_monte_carlo.cpp
    cpp/src/monte_carlo.cpp
    cpp/src/full_revaluation.cpp
    cpp/src/scenario_application.cpp
    cpp/src/scenario_generator.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(test_monte_carlo PRIVATE
    cpp/include
)

add_test(
    NAME MonteCarloTests
    COMMAND test_monte_carlo
)

add_executable(test_risk_metrics
    cpp/tests/test_risk_metrics.cpp
    cpp/src/risk_metrics.cpp
)

target_include_directories(test_risk_metrics PRIVATE
    cpp/include
)

add_test(
    NAME RiskMetricsTests
    COMMAND test_risk_metrics
)

add_executable(test_dataset_generator
    cpp/tests/test_dataset_generator.cpp
    cpp/src/dataset_generator.cpp
    cpp/src/scenario_application.cpp
    cpp/src/scenario_generator.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(test_dataset_generator PRIVATE
    cpp/include
)

add_test(
    NAME DatasetGeneratorTests
    COMMAND test_dataset_generator
)

add_executable(test_csv_writer
    cpp/tests/test_csv_writer.cpp
    cpp/src/csv_writer.cpp
)

target_include_directories(test_csv_writer PRIVATE
    cpp/include
)

add_test(
    NAME CsvWriterTests
    COMMAND test_csv_writer
)

add_executable(test_dataset_exporter
    cpp/tests/test_dataset_exporter.cpp
    cpp/src/dataset_exporter.cpp
    cpp/src/scenario_application.cpp
    cpp/src/scenario_generator.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(test_dataset_exporter PRIVATE
    cpp/include
)

add_test(
    NAME DatasetExporterTests
    COMMAND test_dataset_exporter
)

add_executable(generate_datasets
    cpp/src/generate_datasets.cpp
    cpp/src/dataset_exporter.cpp
    cpp/src/scenario_application.cpp
    cpp/src/scenario_generator.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(generate_datasets PRIVATE
    cpp/include
)

add_executable(generate_training_replicates
    cpp/src/generate_training_replicates.cpp
    cpp/src/dataset_exporter.cpp
    cpp/src/scenario_application.cpp
    cpp/src/scenario_generator.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(generate_training_replicates PRIVATE
    cpp/include
)

add_executable(benchmark_full_revaluation
    cpp/src/benchmark_full_revaluation.cpp
    cpp/src/scenario_application.cpp
    cpp/src/scenario_generator.cpp
    cpp/src/portfolio.cpp
    cpp/src/european_option.cpp
    cpp/src/black_scholes.cpp
)

target_include_directories(benchmark_full_revaluation PRIVATE
    cpp/include
)

add_executable(benchmark_scenario_generation
    cpp/src/benchmark_scenario_generation.cpp
    cpp/src/scenario_generator.cpp
)

target_include_directories(benchmark_scenario_generation PRIVATE
    cpp/include
)