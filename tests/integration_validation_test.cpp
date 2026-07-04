#include "inputs/parser.h"
#include "inputs/rolling_window_scaler.h"
#include "inputs/stock_data.h"
#include "model/dense.h"
#include "model/huber_loss_function.h"
#include "model/lstm.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kTol = 1e-9;
constexpr double kDeterminismTol = 1e-12;

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

PriceData makePrice(int dayIndex, double close, uint64_t volume) {
    PriceData p{};
    p.open = close - 0.15;
    p.high = close + 0.25;
    p.low = close - 0.30;
    p.close = close;
    p.adjClose = close;
    p.volume = volume;
    p.date = "";
    (void)dayIndex;
    return p;
}

std::vector<PriceData> makeChronologicalSeries(int startDay, int count) {
    std::vector<PriceData> out;
    out.reserve(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        const int day = startDay + i;
        const double base = 100.0 + 1.25 * static_cast<double>(day);
        const double wiggle = 0.08 * std::sin(0.3 * static_cast<double>(day));
        const double close = base + wiggle;
        out.push_back(makePrice(day, close, static_cast<uint64_t>(1000 + day * 7)));
    }

    return out;
}

RollingWindowScaler preloadedValidationScaler(const std::vector<PriceData>& training,
                                              std::size_t windowSize,
                                              std::size_t numFeatures) {
    RollingWindowScaler scaler(windowSize, numFeatures);

    const std::size_t split = training.size();
    const std::size_t preLoadStart = (split > windowSize) ? (split - windowSize) : 0;

    for (std::size_t i = preLoadStart; i < split; ++i) {
        scaler.add(training[i]);
    }

    return scaler;
}

Eigen::VectorXd deterministicLstmParams(int count) {
    Eigen::VectorXd v(count);
    for (int i = 0; i < count; ++i) {
        v(i) = 0.012 * std::sin(0.27 * static_cast<double>(i + 1))
             + 0.003 * std::cos(0.11 * static_cast<double>(i + 1));
    }
    return v;
}

void setDeterministicDense(Dense& dense) {
    for (int i = 0; i < dense.W.size(); ++i) {
        dense.W(i) = 0.02 * std::cos(0.21 * static_cast<double>(i + 1));
    }
    dense.b = -0.01;
}

Eigen::VectorXd predictBatch(const Eigen::TensorMap<const Eigen::Tensor<double, 3, Eigen::RowMajor>>& batchInputs,
                             LSTMCell& lstm,
                             const Dense& dense,
                             int sequenceLength,
                             int numFeatures) {
    const int batchSize = static_cast<int>(batchInputs.dimension(0));
    Eigen::VectorXd preds(batchSize);

    for (int b = 0; b < batchSize; ++b) {
        lstm.reset();

        for (int t = 0; t < sequenceLength; ++t) {
            Eigen::VectorXd x(numFeatures);
            for (int f = 0; f < numFeatures; ++f) {
                x(f) = batchInputs(b, t, f);
            }
            (void)lstm.forwardPass(x);
        }

        preds(b) = dense.forward(lstm.getHiddenState());
    }

    return preds;
}

void test_end_to_end_validation_integration_path() {
    constexpr int numFeatures = 6;
    constexpr int sequenceLength = 3;
    constexpr int batchSize = 2;
    constexpr int hiddenSize = 5;
    constexpr std::size_t windowSize = 4;

    const int totalDays = 18;
    const int splitIndex = 10;

    const auto full = makeChronologicalSeries(/*startDay=*/0, /*count=*/totalDays);

    const std::vector<PriceData> training(full.begin(), full.begin() + splitIndex);
    const std::vector<PriceData> validation(full.begin() + splitIndex, full.end());

    expectTrue(!training.empty() && !validation.empty(), "training and validation splits must be non-empty");
    expectTrue(static_cast<int>(training.size()) == splitIndex, "training split size mismatch");
    expectTrue(training.back().close < validation.front().close,
               "validation rows must occur strictly after training rows in this chronological synthetic set");

    // Future validation rows should not alter earlier validation windows when the first validation rows are unchanged.
    std::vector<PriceData> validationAlt = validation;
    for (std::size_t i = static_cast<std::size_t>(sequenceLength); i < validationAlt.size(); ++i) {
        validationAlt[i].open += 1500.0;
        validationAlt[i].high += 1500.0;
        validationAlt[i].low += 1500.0;
        validationAlt[i].close += 1500.0;
        validationAlt[i].adjClose += 1500.0;
        validationAlt[i].volume += 900000;
    }

    RollingWindowScaler trainScaler(windowSize, numFeatures);
    RollingWindowScaler validationScalerA = preloadedValidationScaler(training, windowSize, numFeatures);
    RollingWindowScaler validationScalerB = preloadedValidationScaler(training, windowSize, numFeatures);

    StockData trainData(training, numFeatures, sequenceLength, batchSize, std::move(trainScaler));
    StockData validationDataA(validation, numFeatures, sequenceLength, batchSize, std::move(validationScalerA));
    StockData validationDataB(validationAlt, numFeatures, sequenceLength, batchSize, std::move(validationScalerB));

    expectTrue(trainData.getNumWindows() > 0, "training StockData should produce at least one window");
    expectTrue(validationDataA.getNumWindows() > 0, "validation StockData should produce at least one window");

    auto trainBatch = trainData.nextBatch();
    auto valBatchA = validationDataA.nextBatch();
    auto valBatchB = validationDataB.nextBatch();

    expectTrue(trainBatch.first.dimension(0) == batchSize, "train batch dimension(0) should match batchSize");
    expectTrue(trainBatch.first.dimension(1) == sequenceLength, "train batch dimension(1) should match sequenceLength");
    expectTrue(trainBatch.first.dimension(2) == numFeatures, "train batch dimension(2) should match numFeatures");
    expectTrue(trainBatch.second.size() == trainBatch.first.dimension(0), "train target size must match train batch size");

    expectTrue(valBatchA.first.dimension(1) == sequenceLength, "validation batch sequenceLength mismatch");
    expectTrue(valBatchA.first.dimension(2) == numFeatures, "validation batch numFeatures mismatch");
    expectTrue(valBatchA.second.size() == valBatchA.first.dimension(0), "validation target size must match validation batch size");

    for (int t = 0; t < sequenceLength; ++t) {
        for (int f = 0; f < numFeatures; ++f) {
            const double a = valBatchA.first(0, t, f);
            const double b = valBatchB.first(0, t, f);
            expectNear(a, b, kTol, "future validation rows should not influence first validation window scaling");
        }
    }

    LSTMCell lstm(numFeatures, hiddenSize, sequenceLength);
    const int paramCount = static_cast<int>(lstm.getParametersVector().size());
    lstm.setParametersVector(deterministicLstmParams(paramCount));

    Dense dense(hiddenSize);
    setDeterministicDense(dense);

    Eigen::VectorXd predsA = predictBatch(valBatchA.first, lstm, dense, sequenceLength, numFeatures);
    expectTrue(predsA.size() == valBatchA.second.size(), "prediction vector size must match validation target size");

    HuberLossFunction huber(1.0);
    const double loss = huber.forward(predsA, valBatchA.second);
    expectTrue(std::isfinite(loss), "Huber loss must be finite for integration path batch");

    // Deterministic forward check: same parameters + same inputs + reset-per-sequence path should match exactly.
    Eigen::VectorXd predsB = predictBatch(valBatchA.first, lstm, dense, sequenceLength, numFeatures);
    expectTrue(predsA.size() == predsB.size(), "deterministic prediction comparison size mismatch");
    for (int i = 0; i < predsA.size(); ++i) {
        expectNear(predsA(i), predsB(i), kDeterminismTol,
                   "repeated deterministic forward pass should reproduce the same prediction");
    }
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"end-to-end numerical integration validation path", test_end_to_end_validation_integration_path},
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

    std::cout << "\nIntegration validation tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
