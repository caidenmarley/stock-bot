#include "model/metrics.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-9;

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

void expectNear(double actual, double expected, const std::string& msg, double tol = kTol) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

void test_cash_baseline_daily_behavior() {
    const std::vector<double> returns = {0.01, -0.02, 0.03};
    const auto positions = metrics::cashBaselinePositions(returns.size());
    const auto pnl = metrics::calcDailyPnLAndTurnover(positions, returns, 0.001);

    expectTrue(positions.size() == returns.size(), "cash positions size mismatch");
    for (std::size_t i = 0; i < returns.size(); ++i) {
        expectNear(positions[i], 0.0, "cash position should be flat");
        expectNear(pnl.grossReturn[i], 0.0, "cash gross return should be zero");
        expectNear(pnl.turnover[i], 0.0, "cash turnover should be zero");
        expectNear(pnl.netReturn[i], 0.0, "cash net return should be zero");
    }

    expectNear(metrics::sharpe(pnl.netReturn, 252), 0.0, "cash sharpe should be neutral");
}

void test_buy_and_hold_positions() {
    const auto positions = metrics::buyAndHoldBaselinePositions(5);
    expectTrue(positions.size() == 5, "buy_and_hold size mismatch");
    for (double p : positions) {
        expectNear(p, 1.0, "buy_and_hold should be fully invested");
    }
}

void test_transaction_costs_use_existing_pnl_path() {
    const std::vector<double> returns = {0.01, -0.02, 0.03};
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};

    const auto summaries = metrics::evaluateStandardBenchmarks(returns, params, 123u);
    expectTrue(summaries.size() == 4, "expected four benchmark summaries");

    // buy_and_hold: positions [1,1,1], turnover [1,0,0], costs [0.001,0,0]
    // net: [0.009, -0.02, 0.03], cumulative = 0.019
    double buyHoldCum = 0.0;
    bool found = false;
    for (const auto& s : summaries) {
        if (s.name == "buy_and_hold") {
            buyHoldCum = s.cumulativeNetReturn;
            found = true;
            break;
        }
    }
    expectTrue(found, "buy_and_hold summary missing");
    expectNear(buyHoldCum, 0.019, "buy_and_hold cumulative net should include entry cost");
}

void test_random_baseline_reproducible_for_seed() {
    const std::size_t n = 32;
    const auto a = metrics::randomNoSkillBaselinePositions(n, 42u);
    const auto b = metrics::randomNoSkillBaselinePositions(n, 42u);
    const auto c = metrics::randomNoSkillBaselinePositions(n, 7u);

    expectTrue(a == b, "random baseline should be reproducible for same seed");
    expectTrue(a != c, "random baseline should differ for different seed (high probability)");
}

void test_momentum_baseline_no_current_day_leakage() {
    const std::vector<double> returns = {0.5, -0.7, 0.2, -0.1};
    const auto pos = metrics::previousReturnMomentumPositions(returns);

    expectTrue(pos.size() == returns.size(), "momentum positions size mismatch");
    expectNear(pos[0], 0.0, "position[0] must be flat when no previous return exists");

    for (std::size_t t = 1; t < returns.size(); ++t) {
        const double expected = returns[t - 1] > 0.0 ? 1.0 : 0.0;
        expectNear(pos[t], expected, "momentum position must use previous return only");
    }
}

void test_empty_and_tiny_input_safe() {
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};

    const auto empty = metrics::evaluateStandardBenchmarks({}, params, 1u);
    expectTrue(empty.size() == 4, "empty input should still return all benchmark summaries");
    for (const auto& s : empty) {
        expectNear(s.sharpeNet, 0.0, "empty sharpe should be zero");
        expectNear(s.avgTurnover, 0.0, "empty turnover should be zero");
        expectNear(s.cumulativeNetReturn, 0.0, "empty cumulative return should be zero");
    }

    const std::vector<double> tiny = {0.02};
    const auto one = metrics::evaluateStandardBenchmarks(tiny, params, 5u);
    expectTrue(one.size() == 4, "tiny input should return all benchmark summaries");
}

void test_benchmark_names_stable() {
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};
    const std::vector<double> returns = {0.01, -0.02};

    const auto summaries = metrics::evaluateStandardBenchmarks(returns, params, 99u);
    expectTrue(summaries.size() == 4, "expected four benchmark summaries");

    expectTrue(summaries[0].name == "cash", "benchmark name mismatch: cash");
    expectTrue(summaries[1].name == "buy_and_hold", "benchmark name mismatch: buy_and_hold");
    expectTrue(summaries[2].name == "random_noskill", "benchmark name mismatch: random_noskill");
    expectTrue(summaries[3].name == "prev_return_momentum", "benchmark name mismatch: prev_return_momentum");
}

void test_model_and_benchmark_rows_have_stable_order_and_schema() {
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};
    const std::vector<double> preds = {0.02, -0.01, 0.03};
    const std::vector<double> returns = {0.01, -0.02, 0.03};

    const auto rows = metrics::evaluateModelAndBenchmarks(preds, returns, params, 7u);
    expectTrue(rows.size() == 5, "expected model plus four benchmark rows");

    expectTrue(rows[0].name == "model", "row 0 should be model");
    expectTrue(rows[1].name == "cash", "row 1 should be cash");
    expectTrue(rows[2].name == "buy_and_hold", "row 2 should be buy_and_hold");
    expectTrue(rows[3].name == "random_noskill", "row 3 should be random_noskill");
    expectTrue(rows[4].name == "prev_return_momentum", "row 4 should be prev_return_momentum");

    for (const auto& row : rows) {
        expectTrue(row.numObservations == returns.size(), "numObservations schema mismatch");
    }
}

void test_model_row_uses_threshold_position_logic() {
    metrics::ProfitAndLossParams params{0.001, 0.001, 252};
    const std::vector<double> preds = {0.001, 0.0011, -0.1};
    const std::vector<double> returns = {0.01, 0.02, 0.03};

    const auto rows = metrics::evaluateModelAndBenchmarks(preds, returns, params, 11u);
    expectTrue(rows.size() == 5, "expected model plus four benchmark rows");

    // threshold rule is strict > threshold, so positions => [0,1,0]
    // turnover => [0,1,1], avg = 2/3
    // net => [0, 0.019, -0.001], cumulative = 0.018
    expectNear(rows[0].avgTurnover, 2.0 / 3.0, "model avgTurnover should follow threshold position logic");
    expectNear(rows[0].cumulativeNetReturn, 0.018, "model cumulative net should use same pnl path");
}

void test_benchmark_rows_match_standard_helper() {
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};
    const std::vector<double> preds = {0.2, -0.1, 0.3, -0.4};
    const std::vector<double> returns = {0.01, -0.02, 0.03, -0.04};

    const auto combo = metrics::evaluateModelAndBenchmarks(preds, returns, params, 123u);
    const auto base = metrics::evaluateStandardBenchmarks(returns, params, 123u);

    expectTrue(combo.size() == 5, "combo row count mismatch");
    expectTrue(base.size() == 4, "base row count mismatch");

    for (std::size_t i = 0; i < base.size(); ++i) {
        const auto& a = combo[i + 1];
        const auto& b = base[i];
        expectTrue(a.name == b.name, "benchmark row name mismatch");
        expectNear(a.sharpeNet, b.sharpeNet, "benchmark sharpe mismatch");
        expectNear(a.avgTurnover, b.avgTurnover, "benchmark turnover mismatch");
        expectNear(a.cumulativeNetReturn, b.cumulativeNetReturn, "benchmark cumulative net mismatch");
        expectTrue(a.numObservations == b.numObservations, "benchmark numObservations mismatch");
    }
}

void test_model_and_benchmark_random_reproducibility() {
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};
    const std::vector<double> preds = {0.1, -0.2, 0.3, -0.4, 0.5};
    const std::vector<double> returns = {0.01, -0.02, 0.03, -0.04, 0.05};

    const auto a = metrics::evaluateModelAndBenchmarks(preds, returns, params, 77u);
    const auto b = metrics::evaluateModelAndBenchmarks(preds, returns, params, 77u);

    expectTrue(a.size() == b.size(), "comparison size mismatch");
    for (std::size_t i = 0; i < a.size(); ++i) {
        expectTrue(a[i].name == b[i].name, "comparison name mismatch");
        expectNear(a[i].sharpeNet, b[i].sharpeNet, "comparison sharpe mismatch");
        expectNear(a[i].avgTurnover, b[i].avgTurnover, "comparison turnover mismatch");
        expectNear(a[i].cumulativeNetReturn, b[i].cumulativeNetReturn, "comparison cumulative net mismatch");
        expectTrue(a[i].numObservations == b[i].numObservations, "comparison observations mismatch");
    }
}

void test_csv_header_is_exactly_stable() {
    expectTrue(
        metrics::benchmarkComparisonCsvHeader() ==
            "strategy,sharpe_net,avg_turnover,cumulative_net_return,num_observations",
        "CSV header mismatch"
    );
}

void test_csv_row_order_is_preserved() {
    const std::vector<metrics::BenchmarkSummary> rows = {
        {"model", 1.0, 0.1, 0.2, 3},
        {"cash", 0.0, 0.0, 0.0, 3},
        {"buy_and_hold", 2.0, 1.0, 0.3, 3},
    };

    const auto csvRows = metrics::benchmarkComparisonToCsvRows(rows);
    expectTrue(csvRows.size() == rows.size(), "csv row count mismatch");
    expectTrue(csvRows[0].rfind("model,", 0) == 0, "row order mismatch for model");
    expectTrue(csvRows[1].rfind("cash,", 0) == 0, "row order mismatch for cash");
    expectTrue(csvRows[2].rfind("buy_and_hold,", 0) == 0, "row order mismatch for buy_and_hold");
}

void test_csv_emits_all_expected_fields_with_deterministic_format() {
    const std::vector<metrics::BenchmarkSummary> rows = {
        {"model", 1.23456789, 0.25, -0.5, 42},
    };

    const auto csvRows = metrics::benchmarkComparisonToCsvRows(rows);
    expectTrue(csvRows.size() == 1, "expected one csv row");
    expectTrue(
        csvRows[0] == "model,1.234568,0.250000,-0.500000,42",
        "deterministic CSV formatting mismatch"
    );
}

void test_csv_empty_input_produces_no_data_rows() {
    const auto csvRows = metrics::benchmarkComparisonToCsvRows({});
    expectTrue(csvRows.empty(), "empty input should produce no csv data rows");
}

void test_csv_serializer_does_not_mutate_input() {
    std::vector<metrics::BenchmarkSummary> rows = {
        {"model", 1.0, 0.2, 0.3, 4},
        {"cash", 0.0, 0.0, 0.0, 4},
    };
    const auto before = rows;

    const auto csvRows = metrics::benchmarkComparisonToCsvRows(rows);
    expectTrue(csvRows.size() == 2, "expected two csv rows");
    expectTrue(rows.size() == before.size(), "input row count mutated");
    for (std::size_t i = 0; i < rows.size(); ++i) {
        expectTrue(rows[i].name == before[i].name, "input name mutated");
        expectNear(rows[i].sharpeNet, before[i].sharpeNet, "input sharpe mutated");
        expectNear(rows[i].avgTurnover, before[i].avgTurnover, "input turnover mutated");
        expectNear(rows[i].cumulativeNetReturn, before[i].cumulativeNetReturn, "input cumulative return mutated");
        expectTrue(rows[i].numObservations == before[i].numObservations, "input observations mutated");
    }
}

void test_comparison_rows_serialize_directly() {
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};
    const std::vector<double> preds = {0.2, -0.1, 0.3, -0.4};
    const std::vector<double> returns = {0.01, -0.02, 0.03, -0.04};

    const auto rows = metrics::evaluateModelAndBenchmarks(preds, returns, params, 123u);
    const auto csvRows = metrics::benchmarkComparisonToCsvRows(rows);

    expectTrue(rows.size() == 5, "expected five comparison rows");
    expectTrue(csvRows.size() == rows.size(), "csv row count should match comparison rows");
    expectTrue(csvRows[0].rfind("model,", 0) == 0, "serialized model row missing or out of order");
    expectTrue(csvRows[1].rfind("cash,", 0) == 0, "serialized cash row missing or out of order");
    expectTrue(csvRows[2].rfind("buy_and_hold,", 0) == 0, "serialized buy_and_hold row missing or out of order");
    expectTrue(csvRows[3].rfind("random_noskill,", 0) == 0, "serialized random_noskill row missing or out of order");
    expectTrue(csvRows[4].rfind("prev_return_momentum,", 0) == 0, "serialized prev_return_momentum row missing or out of order");
}

void test_csv_block_starts_with_exact_header() {
    const std::string block = metrics::benchmarkComparisonToCsvBlock({});
    const std::string expectedPrefix =
        "strategy,sharpe_net,avg_turnover,cumulative_net_return,num_observations\n";
    expectTrue(block == expectedPrefix, "CSV block should contain exact header and newline for empty input");
}

void test_csv_block_contains_header_and_all_rows_in_order() {
    const std::vector<metrics::BenchmarkSummary> rows = {
        {"model", 1.0, 0.1, 0.2, 3},
        {"cash", 0.0, 0.0, 0.0, 3},
        {"buy_and_hold", 2.0, 1.0, 0.3, 3},
    };

    const std::string block = metrics::benchmarkComparisonToCsvBlock(rows);
    const std::string expected =
        "strategy,sharpe_net,avg_turnover,cumulative_net_return,num_observations\n"
        "model,1.000000,0.100000,0.200000,3\n"
        "cash,0.000000,0.000000,0.000000,3\n"
        "buy_and_hold,2.000000,1.000000,0.300000,3\n";
    expectTrue(block == expected, "CSV block header/rows/order mismatch");
}

void test_csv_block_newline_convention_is_deterministic() {
    const std::vector<metrics::BenchmarkSummary> rows = {
        {"model", 1.0, 0.1, 0.2, 3},
    };
    const std::string block = metrics::benchmarkComparisonToCsvBlock(rows);

    expectTrue(!block.empty(), "CSV block should not be empty");
    expectTrue(block.back() == '\n', "CSV block should end with newline");
    expectTrue(block.find("\r") == std::string::npos, "CSV block should use '\\n' and not contain carriage returns");
}

void test_csv_block_accepts_model_and_benchmark_output_directly() {
    metrics::ProfitAndLossParams params{0.0, 0.001, 252};
    const std::vector<double> preds = {0.2, -0.1, 0.3, -0.4};
    const std::vector<double> returns = {0.01, -0.02, 0.03, -0.04};

    const auto rows = metrics::evaluateModelAndBenchmarks(preds, returns, params, 123u);
    const std::string block = metrics::benchmarkComparisonToCsvBlock(rows);

    expectTrue(block.rfind("strategy,sharpe_net,avg_turnover,cumulative_net_return,num_observations\n", 0) == 0,
               "CSV block should start with fixed header");
    expectTrue(block.find("\nmodel,") != std::string::npos, "CSV block should contain model row");
    expectTrue(block.find("\ncash,") != std::string::npos, "CSV block should contain cash row");
    expectTrue(block.find("\nbuy_and_hold,") != std::string::npos, "CSV block should contain buy_and_hold row");
    expectTrue(block.find("\nrandom_noskill,") != std::string::npos, "CSV block should contain random_noskill row");
    expectTrue(block.find("\nprev_return_momentum,") != std::string::npos, "CSV block should contain prev_return_momentum row");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"cash baseline daily behavior", test_cash_baseline_daily_behavior},
        {"buy and hold baseline positions", test_buy_and_hold_positions},
        {"transaction costs through pnl path", test_transaction_costs_use_existing_pnl_path},
        {"random baseline reproducibility", test_random_baseline_reproducible_for_seed},
        {"momentum baseline no leakage", test_momentum_baseline_no_current_day_leakage},
        {"empty and tiny input safety", test_empty_and_tiny_input_safe},
        {"benchmark names stable", test_benchmark_names_stable},
        {"model+benchmark stable row order", test_model_and_benchmark_rows_have_stable_order_and_schema},
        {"model row threshold logic", test_model_row_uses_threshold_position_logic},
        {"benchmark rows match standard helper", test_benchmark_rows_match_standard_helper},
        {"model+benchmark random reproducibility", test_model_and_benchmark_random_reproducibility},
        {"csv header stable", test_csv_header_is_exactly_stable},
        {"csv row order preserved", test_csv_row_order_is_preserved},
        {"csv emits deterministic fields", test_csv_emits_all_expected_fields_with_deterministic_format},
        {"csv empty input", test_csv_empty_input_produces_no_data_rows},
        {"csv serializer non mutating", test_csv_serializer_does_not_mutate_input},
        {"comparison rows serialize directly", test_comparison_rows_serialize_directly},
        {"csv block exact header", test_csv_block_starts_with_exact_header},
        {"csv block contains all rows", test_csv_block_contains_header_and_all_rows_in_order},
        {"csv block newline convention", test_csv_block_newline_convention_is_deterministic},
        {"csv block accepts comparison rows", test_csv_block_accepts_model_and_benchmark_output_directly},
    };

    std::size_t passed = 0;
    for (const auto& [name, fn] : tests) {
        try {
            fn();
            ++passed;
            std::cout << "[PASS] " << name << "\n";
        } catch (const std::exception& ex) {
            std::cout << "[FAIL] " << name << " -> " << ex.what() << "\n";
        }
    }

    std::cout << "\nBenchmark metrics tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
