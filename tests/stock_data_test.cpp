#include "inputs/parser.h"
#include "inputs/rolling_window_scaler.h"
#include "inputs/stock_data.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kTol = 1e-9;

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

void expectNear(double actual, double expected, const std::string& msg) {
    if (std::abs(actual - expected) > kTol) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

PriceData makePrice(double close, uint64_t volume) {
    PriceData p{};
    p.open = close - 0.5;
    p.high = close + 0.5;
    p.low = close - 1.0;
    p.close = close;
    p.adjClose = close;
    p.volume = volume;
    p.date = "";
    return p;
}

PriceData makeCustomPrice(double open, double high, double low, double close, double adjClose, uint64_t volume) {
    PriceData p{};
    p.open = open;
    p.high = high;
    p.low = low;
    p.close = close;
    p.adjClose = adjClose;
    p.volume = volume;
    p.date = "";
    return p;
}

std::vector<PriceData> makeSyntheticData(const std::vector<double>& closes) {
    std::vector<PriceData> out;
    out.reserve(closes.size());
    for (std::size_t i = 0; i < closes.size(); ++i) {
        out.push_back(makePrice(closes[i], static_cast<uint64_t>(1000 + i * 10)));
    }
    return out;
}

std::vector<double> expectedTargets(const std::vector<double>& closes, int sequenceLength) {
    const int numWindows = static_cast<int>(closes.size()) - sequenceLength;
    std::vector<double> targets;
    targets.reserve(static_cast<std::size_t>(numWindows));
    for (int seq = 0; seq < numWindows; ++seq) {
        const int lastDay = seq + sequenceLength - 1;
        const double closeT = closes[static_cast<std::size_t>(lastDay)];
        const double closeTp1 = closes[static_cast<std::size_t>(lastDay + 1)];
        targets.push_back((closeTp1 - closeT) / closeT);
    }
    return targets;
}

StockData makeStockData(const std::vector<double>& closes, int sequenceLength, int batchSize,
                        int numFeatures = stock_features::kFeatureCount) {
    const auto raw = makeSyntheticData(closes);
    RollingWindowScaler scaler(/*windowSize=*/3, static_cast<std::size_t>(numFeatures));
    return StockData(raw, numFeatures, sequenceLength, batchSize, scaler);
}

void test_constructor_rejects_too_small_data() {
    const int sequenceLength = 3;
    const int batchSize = 2;
    const auto raw = makeSyntheticData({10.0, 11.0, 12.0}); // sequenceLength only
    RollingWindowScaler scaler(3, 6);

    bool threw = false;
    try {
        StockData data(raw, 6, sequenceLength, batchSize, scaler);
        (void)data;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expectTrue(threw, "constructor should reject datasets with fewer than sequenceLength + 1 rows");
}

void test_num_windows_calculation() {
    const int sequenceLength = 3;
    StockData data = makeStockData({10, 12, 15, 19, 24, 30, 37, 45}, sequenceLength, 2);
    expectTrue(data.getNumWindows() == 5, "numWindows should be numDays - sequenceLength");
}

void test_next_batch_size_and_targets() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    const int sequenceLength = 3;
    const int batchSize = 2;

    StockData data = makeStockData(closes, sequenceLength, batchSize);
    const auto expected = expectedTargets(closes, sequenceLength);

    auto [inputs, targets] = data.nextBatch();
    expectTrue(inputs.dimension(0) == 2, "first batch should have size 2");
    expectTrue(targets.size() == 2, "first target batch should have size 2");
    expectNear(targets(0), expected[0], "first batch target[0] mismatch");
    expectNear(targets(1), expected[1], "first batch target[1] mismatch");
}

void test_partial_final_batch() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45}; // numWindows=5
    StockData data = makeStockData(closes, /*sequenceLength=*/3, /*batchSize=*/2);

    auto b1 = data.nextBatch();
    auto b2 = data.nextBatch();
    auto b3 = data.nextBatch();

    (void)b1;
    (void)b2;
    expectTrue(b3.first.dimension(0) == 1, "final partial batch should contain remaining 1 window");
    expectTrue(b3.second.size() == 1, "final partial target batch should contain 1 element");
}

void test_reset_restarts_batching() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    StockData data = makeStockData(closes, /*sequenceLength=*/3, /*batchSize=*/2);
    const auto expected = expectedTargets(closes, /*sequenceLength=*/3);

    auto first = data.nextBatch();
    expectNear(first.second(0), expected[0], "pre-reset first target[0] mismatch");
    expectNear(first.second(1), expected[1], "pre-reset first target[1] mismatch");

    // Consume another batch then reset
    (void)data.nextBatch();
    data.reset();

    auto afterReset = data.nextBatch();
    expectNear(afterReset.second(0), expected[0], "after reset first target[0] should repeat");
    expectNear(afterReset.second(1), expected[1], "after reset first target[1] should repeat");
}

void test_next_batch_throws_after_consumed() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    StockData data = makeStockData(closes, /*sequenceLength=*/3, /*batchSize=*/2);

    (void)data.nextBatch();
    (void)data.nextBatch();
    (void)data.nextBatch();

    bool threw = false;
    try {
        (void)data.nextBatch();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expectTrue(threw, "nextBatch() should throw after all batches are consumed");
}

void test_target_alignment_formula() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    const int sequenceLength = 3;
    StockData data = makeStockData(closes, sequenceLength, /*batchSize=*/5);
    const auto expected = expectedTargets(closes, sequenceLength);

    auto [inputs, targets] = data.nextBatch();
    (void)inputs;

    expectTrue(static_cast<int>(targets.size()) == static_cast<int>(expected.size()), "target size mismatch");
    for (int i = 0; i < targets.size(); ++i) {
        expectNear(targets(i), expected[static_cast<std::size_t>(i)], "target alignment formula mismatch at index " + std::to_string(i));
    }
}

void test_target_alignment_formula_six_feature_mode() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    const int sequenceLength = 3;
    StockData data = makeStockData(closes, sequenceLength, /*batchSize=*/2, /*numFeatures=*/6);
    const auto expected = expectedTargets(closes, sequenceLength);

    std::vector<double> actual;
    actual.reserve(expected.size());
    while (data.hasAnotherBatch()) {
        auto [inputs, targets] = data.nextBatch();
        (void)inputs;
        for (int i = 0; i < targets.size(); ++i) {
            actual.push_back(targets(i));
        }
    }

    expectTrue(actual.size() == expected.size(), "6-feature target count mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        expectNear(actual[i], expected[i], "6-feature target alignment mismatch at index " + std::to_string(i));
    }
}

void test_target_alignment_formula_thirteen_feature_mode() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    const int sequenceLength = 3;
    StockData data = makeStockData(closes, sequenceLength, /*batchSize=*/2, stock_features::kFeatureCount);
    const auto expected = expectedTargets(closes, sequenceLength);

    std::vector<double> actual;
    actual.reserve(expected.size());
    while (data.hasAnotherBatch()) {
        auto [inputs, targets] = data.nextBatch();
        (void)inputs;
        for (int i = 0; i < targets.size(); ++i) {
            actual.push_back(targets(i));
        }
    }

    expectTrue(actual.size() == expected.size(), "13-feature target count mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        expectNear(actual[i], expected[i], "13-feature target alignment mismatch at index " + std::to_string(i));
    }
}

void test_next_batch_shuffled_respects_order() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    const int sequenceLength = 3;
    StockData data = makeStockData(closes, sequenceLength, /*batchSize=*/5);
    const auto expected = expectedTargets(closes, sequenceLength);

    // Choose non-trivial order and pull a middle slice.
    const std::vector<int> order = {4, 1, 3, 0, 2};
    const int batchStart = 1;
    const int currentBatchSize = 3;

    auto [inputs, targets] = data.nextBatchShuffled(order, batchStart, currentBatchSize);
    (void)inputs;

    expectTrue(targets.size() == currentBatchSize, "shuffled target batch size mismatch");
    expectNear(targets(0), expected[1], "shuffled target[0] mismatch");
    expectNear(targets(1), expected[3], "shuffled target[1] mismatch");
    expectNear(targets(2), expected[0], "shuffled target[2] mismatch");
}

void test_input_tensor_shape() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    const int sequenceLength = 3;
    const int batchSize = 2;
    StockData data = makeStockData(closes, sequenceLength, batchSize);

    auto [inputs, targets] = data.nextBatch();
    (void)targets;

    expectTrue(inputs.dimension(0) == batchSize, "tensor dim0 should be currentBatch");
    expectTrue(inputs.dimension(1) == sequenceLength, "tensor dim1 should be sequenceLength");
    expectTrue(inputs.dimension(2) == stock_features::kFeatureCount, "tensor dim2 should be numFeatures");
}

void test_engineered_feature_count_constant_and_raw_vector_size() {
    expectTrue(stock_features::kFeatureCount == 13, "engineered feature count should be 13");

    const auto raw = makeSyntheticData({10.0, 11.0, 12.0, 13.0});
    const auto featureRow = stock_features::buildRawFeatureVector(raw, 2);
    expectTrue(static_cast<int>(featureRow.size()) == stock_features::kFeatureCount,
               "raw feature row size should match engineered feature count");
}

void test_original_ohlcv_features_preserved_in_first_six_scaled_slots() {
    const std::vector<double> closes = {10, 12, 15, 19, 24, 30, 37, 45};
    const int sequenceLength = 3;
    const int batchSize = 2;

    StockData legacy = makeStockData(closes, sequenceLength, batchSize, /*numFeatures=*/6);
    StockData engineered = makeStockData(closes, sequenceLength, batchSize, stock_features::kFeatureCount);

    auto [legacyInputs, legacyTargets] = legacy.nextBatch();
    auto [engInputs, engTargets] = engineered.nextBatch();

    expectTrue(legacyTargets.size() == engTargets.size(), "legacy and engineered target sizes should match");
    for (int i = 0; i < legacyTargets.size(); ++i) {
        expectNear(legacyTargets(i), engTargets(i), "targets should remain identical when adding engineered features");
    }

    for (int b = 0; b < legacyInputs.dimension(0); ++b) {
        for (int t = 0; t < legacyInputs.dimension(1); ++t) {
            for (int f = 0; f < 6; ++f) {
                expectNear(legacyInputs(b, t, f), engInputs(b, t, f),
                           "first six scaled features should preserve OHLCV behavior");
            }
        }
    }
}

void test_engineered_features_expected_values_on_tiny_dataset() {
    const std::vector<PriceData> raw = {
        makeCustomPrice(10.0, 11.0, 9.0, 10.0, 10.0, 100),
        makeCustomPrice(11.0, 12.0, 10.0, 11.0, 11.0, 110),
        makeCustomPrice(12.0, 13.0, 11.0, 12.0, 12.0, 121),
        makeCustomPrice(13.0, 15.0, 12.0, 14.0, 14.0, 133),
    };

    const auto row = stock_features::buildRawFeatureVector(raw, 3);
    expectTrue(static_cast<int>(row.size()) == 13, "expected 13 engineered features");

    expectNear(row[0], 13.0, "open feature mismatch");
    expectNear(row[1], 15.0, "high feature mismatch");
    expectNear(row[2], 12.0, "low feature mismatch");
    expectNear(row[3], 14.0, "close feature mismatch");
    expectNear(row[4], 14.0, "adj close feature mismatch");
    expectNear(row[5], 133.0, "volume feature mismatch");

    expectNear(row[6], (14.0 - 12.0) / 12.0, "close-to-close return mismatch");
    expectNear(row[7], (14.0 - 13.0) / 13.0, "open-to-close return mismatch");
    expectNear(row[8], (15.0 - 12.0) / 14.0, "range relative-to-close mismatch");
    expectNear(row[9], (14.0 - 12.0) / (15.0 - 12.0), "close position in range mismatch");
    expectNear(row[10], (133.0 - 121.0) / 121.0, "volume change mismatch");
    expectNear(row[11], (14.0 - 10.0) / 10.0, "3-day momentum mismatch");

    const double r1 = (11.0 - 10.0) / 10.0;
    const double r2 = (12.0 - 11.0) / 11.0;
    const double r3 = (14.0 - 12.0) / 12.0;
    const double mean = (r1 + r2 + r3) / 3.0;
    const double variance = ((r1 * r1) + (r2 * r2) + (r3 * r3)) / 3.0 - (mean * mean);
    expectNear(row[12], std::sqrt(variance), "rolling volatility mismatch");
}

void test_future_row_perturbation_does_not_change_earlier_engineered_feature_rows() {
    const std::vector<PriceData> base = {
        makeCustomPrice(10.0, 11.0, 9.0, 10.0, 10.0, 100),
        makeCustomPrice(11.0, 12.0, 10.0, 11.0, 11.0, 110),
        makeCustomPrice(12.0, 13.0, 11.0, 12.0, 12.0, 120),
        makeCustomPrice(13.0, 14.0, 12.0, 13.0, 13.0, 130),
        makeCustomPrice(14.0, 15.0, 13.0, 14.0, 14.0, 140),
    };

    auto perturbed = base;
    perturbed[4] = makeCustomPrice(1400.0, 1500.0, 1300.0, 1400.0, 1400.0, 1400000);

    const auto rowBase = stock_features::buildRawFeatureVector(base, 2);
    const auto rowPerturbed = stock_features::buildRawFeatureVector(perturbed, 2);

    expectTrue(rowBase.size() == rowPerturbed.size(), "feature-row size mismatch under perturbation");
    for (std::size_t i = 0; i < rowBase.size(); ++i) {
        expectNear(rowBase[i], rowPerturbed[i], "future-row perturbation changed earlier engineered feature");
    }
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"constructor rejects insufficient rows", test_constructor_rejects_too_small_data},
        {"numWindows calculation", test_num_windows_calculation},
        {"nextBatch size and targets", test_next_batch_size_and_targets},
        {"partial final batch", test_partial_final_batch},
        {"reset restarts batching", test_reset_restarts_batching},
        {"nextBatch throws after consumed", test_next_batch_throws_after_consumed},
        {"target alignment formula", test_target_alignment_formula},
        {"target alignment formula in 6-feature mode", test_target_alignment_formula_six_feature_mode},
        {"target alignment formula in 13-feature mode", test_target_alignment_formula_thirteen_feature_mode},
        {"nextBatchShuffled order", test_next_batch_shuffled_respects_order},
        {"input tensor shape", test_input_tensor_shape},
        {"engineered feature count", test_engineered_feature_count_constant_and_raw_vector_size},
        {"ohlcv preservation in first six features", test_original_ohlcv_features_preserved_in_first_six_scaled_slots},
        {"engineered feature expected values", test_engineered_features_expected_values_on_tiny_dataset},
        {"future perturbation no leakage", test_future_row_perturbation_does_not_change_earlier_engineered_feature_rows},
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

    std::cout << "\nStockData tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
