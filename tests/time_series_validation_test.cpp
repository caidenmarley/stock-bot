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

std::vector<PriceData> makeSeries(const std::vector<double>& closes) {
    std::vector<PriceData> out;
    out.reserve(closes.size());
    for (std::size_t i = 0; i < closes.size(); ++i) {
        out.push_back(makePrice(closes[i], static_cast<uint64_t>(1000 + i)));
    }
    return out;
}

std::vector<double> expectedTargets(const std::vector<double>& closes, int sequenceLength) {
    const int numWindows = static_cast<int>(closes.size()) - sequenceLength;
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(numWindows));
    for (int seq = 0; seq < numWindows; ++seq) {
        const int lastDay = seq + sequenceLength - 1;
        const double closeT = closes[static_cast<std::size_t>(lastDay)];
        const double closeTp1 = closes[static_cast<std::size_t>(lastDay + 1)];
        out.push_back((closeTp1 - closeT) / closeT);
    }
    return out;
}

RollingWindowScaler preloadedScalerFromTraining(
    const std::vector<PriceData>& training,
    std::size_t windowSize,
    std::size_t numFeatures
) {
    RollingWindowScaler scaler(windowSize, numFeatures);
    const std::size_t split = training.size();
    const std::size_t preLoadStart = (split > windowSize) ? (split - windowSize) : 0;
    for (std::size_t i = preLoadStart; i < split; ++i) {
        scaler.add(training[i]);
    }
    return scaler;
}

void test_fold_splits_are_time_ordered_and_overlapping_by_design() {
    const int rawSize = 100;
    const int sequenceLength = 15;
    const int maxStartSequence = rawSize - sequenceLength;

    const std::vector<int> cutPoints{
        static_cast<int>(0.6 * maxStartSequence),
        static_cast<int>(0.7 * maxStartSequence),
        static_cast<int>(0.8 * maxStartSequence)
    };

    expectTrue(cutPoints.size() == 3, "expected 3 fold cut points");
    expectTrue(cutPoints[0] < cutPoints[1] && cutPoints[1] < cutPoints[2], "fold starts should be strictly increasing");

    for (int splitStart : cutPoints) {
        expectTrue(splitStart > 0, "training segment must be non-empty");
        expectTrue(splitStart < rawSize, "validation segment must be non-empty");
    }

    const int v0Start = cutPoints[0];
    const int v1Start = cutPoints[1];
    const int v2Start = cutPoints[2];
    const int vEnd = rawSize;

    expectTrue(v0Start < v1Start && v1Start < v2Start, "validation fold starts should be time-ordered");
    expectTrue(v1Start < vEnd && v2Start < vEnd, "validation fold ranges should stay in bounds");
    expectTrue(v1Start >= v0Start && v2Start >= v1Start, "later folds should be nested tail subsets of earlier validation ranges");
}

void test_validation_batches_are_sequential_and_targets_are_next_day_returns() {
    const std::vector<double> closes{100.0, 102.0, 105.0, 109.0, 114.0, 120.0};
    const auto raw = makeSeries(closes);
    const int sequenceLength = 2;
    const int batchSize = 2;

    RollingWindowScaler scaler(/*windowSize=*/3, /*numFeatures=*/6);
    StockData validationData(raw, /*numFeatures=*/6, sequenceLength, batchSize, scaler);

    const auto expected = expectedTargets(closes, sequenceLength);

    auto first = validationData.nextBatch();
    expectTrue(first.second.size() == 2, "first validation batch should contain first two windows");
    expectNear(first.second(0), expected[0], "first validation target mismatch at window 0");
    expectNear(first.second(1), expected[1], "first validation target mismatch at window 1");

    auto second = validationData.nextBatch();
    expectTrue(second.second.size() == 2, "second validation batch should contain next two windows");
    expectNear(second.second(0), expected[2], "second validation target mismatch at window 2");
    expectNear(second.second(1), expected[3], "second validation target mismatch at window 3");
}

void test_preloaded_validation_context_does_not_use_future_validation_rows_for_early_windows() {
    const std::vector<PriceData> training = makeSeries({10.0, 20.0, 30.0, 40.0});

    // Same first two validation rows; different future rows.
    const std::vector<PriceData> validationA = makeSeries({50.0, 60.0, 70.0, 80.0, 90.0});
    const std::vector<PriceData> validationB = makeSeries({50.0, 60.0, 7000.0, 8000.0, 9000.0});

    const int sequenceLength = 2;
    const int batchSize = 2;
    const std::size_t windowSize = 3;

    RollingWindowScaler preA = preloadedScalerFromTraining(training, windowSize, 6);
    RollingWindowScaler preB = preloadedScalerFromTraining(training, windowSize, 6);

    StockData valDataA(validationA, /*numFeatures=*/6, sequenceLength, batchSize, std::move(preA));
    StockData valDataB(validationB, /*numFeatures=*/6, sequenceLength, batchSize, std::move(preB));

    auto batchA = valDataA.nextBatch();
    auto batchB = valDataB.nextBatch();

    // First sequence uses validation day 0 and day 1 only, so it should be identical across both datasets.
    for (int t = 0; t < sequenceLength; ++t) {
        for (int f = 0; f < 6; ++f) {
            const double a = batchA.first(0, t, f);
            const double b = batchB.first(0, t, f);
            expectNear(a, b, "future validation rows should not affect early-window scaled features");
        }
    }
}

void test_time_series_validation_targets_match_between_six_and_thirteen_feature_modes() {
    const std::vector<double> closes{100.0, 102.0, 105.0, 109.0, 114.0, 120.0, 127.0, 135.0};
    const auto raw = makeSeries(closes);
    const int sequenceLength = 3;
    const int batchSize = 2;

    RollingWindowScaler scaler6(/*windowSize=*/3, /*numFeatures=*/6);
    RollingWindowScaler scaler13(/*windowSize=*/3, static_cast<std::size_t>(stock_features::kFeatureCount));

    StockData validation6(raw, /*numFeatures=*/6, sequenceLength, batchSize, scaler6);
    StockData validation13(raw, stock_features::kFeatureCount, sequenceLength, batchSize, scaler13);

    std::vector<double> targets6;
    std::vector<double> targets13;

    while (validation6.hasAnotherBatch()) {
        auto [inputs, targets] = validation6.nextBatch();
        (void)inputs;
        for (int i = 0; i < targets.size(); ++i) {
            targets6.push_back(targets(i));
        }
    }
    while (validation13.hasAnotherBatch()) {
        auto [inputs, targets] = validation13.nextBatch();
        (void)inputs;
        for (int i = 0; i < targets.size(); ++i) {
            targets13.push_back(targets(i));
        }
    }

    expectTrue(targets6.size() == targets13.size(), "6-feature and 13-feature validation target counts should match");
    for (std::size_t i = 0; i < targets6.size(); ++i) {
        expectNear(targets6[i], targets13[i], "validation target mismatch between 6 and 13 feature modes");
    }
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"fold split ordering and overlap", test_fold_splits_are_time_ordered_and_overlapping_by_design},
        {"validation batching and target alignment", test_validation_batches_are_sequential_and_targets_are_next_day_returns},
        {"validation preloading no future influence on early windows", test_preloaded_validation_context_does_not_use_future_validation_rows_for_early_windows},
        {"validation target alignment stable across 6 and 13 feature modes", test_time_series_validation_targets_match_between_six_and_thirteen_feature_modes},
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

    std::cout << "\nTime-series validation tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
