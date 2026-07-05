#include "inputs/parser.h"
#include "inputs/rolling_window_scaler.h"
#include "inputs/stock_data.h"
#include "model/ada_belief.h"
#include "model/dense.h"
#include "model/huber_loss_function.h"
#include "model/lstm.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kTol = 1e-12;

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

void expectVectorNear(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol, const std::string& msg) {
    expectTrue(a.size() == b.size(), msg + " size mismatch");
    for (int i = 0; i < a.size(); ++i) {
        expectNear(a(i), b(i), tol, msg + " index=" + std::to_string(i));
    }
}

void expectRowVectorNear(const Eigen::RowVectorXd& a, const Eigen::RowVectorXd& b, double tol, const std::string& msg) {
    expectTrue(a.size() == b.size(), msg + " size mismatch");
    for (int i = 0; i < a.size(); ++i) {
        expectNear(a(i), b(i), tol, msg + " index=" + std::to_string(i));
    }
}

PriceData makePrice(int dayIndex) {
    const double base = 80.0 + 0.9 * static_cast<double>(dayIndex);
    const double wiggle = 0.05 * std::sin(0.41 * static_cast<double>(dayIndex));
    const double close = base + wiggle;

    PriceData p{};
    p.open = close - 0.10;
    p.high = close + 0.20;
    p.low = close - 0.35;
    p.close = close;
    p.adjClose = close;
    p.volume = static_cast<uint64_t>(1200 + dayIndex * 11);
    p.date = "";
    return p;
}

std::vector<PriceData> makeChronologicalSeries(int startDay, int count) {
    std::vector<PriceData> out;
    out.reserve(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        out.push_back(makePrice(startDay + i));
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

struct DeterminismRunResult {
    Eigen::VectorXd predsBefore;
    double lossBefore;
    Eigen::VectorXd predsAfter;
    double lossAfter;
    Eigen::VectorXd lstmParamsBefore;
    Eigen::VectorXd lstmParamsAfter;
    Eigen::RowVectorXd denseWBefore;
    Eigen::RowVectorXd denseWAfter;
    double denseBBefore;
    double denseBAfter;
};

DeterminismRunResult runOneDeterministicPath(uint32_t lstmSeed) {
    constexpr int numFeatures = 6;
    constexpr int sequenceLength = 4;
    constexpr int batchSize = 3;
    constexpr int hiddenSize = 6;
    constexpr std::size_t windowSize = 5;
    constexpr double learningRate = 1e-3;

    const auto full = makeChronologicalSeries(0, 22);
    const int split = 13;

    const std::vector<PriceData> training(full.begin(), full.begin() + split);
    const std::vector<PriceData> validation(full.begin() + split, full.end());

    RollingWindowScaler trainScaler(windowSize, numFeatures);
    RollingWindowScaler valScaler = preloadedValidationScaler(training, windowSize, numFeatures);

    StockData trainingData(training, numFeatures, sequenceLength, batchSize, std::move(trainScaler));
    StockData validationData(validation, numFeatures, sequenceLength, batchSize, std::move(valScaler));

    expectTrue(trainingData.getNumWindows() > 0, "training windows should be non-empty");
    expectTrue(validationData.getNumWindows() > 0, "validation windows should be non-empty");

    std::vector<int> order(trainingData.getNumWindows());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        order[i] = i;
    }
    std::reverse(order.begin(), order.end());

    const int shuffledBatchSize = std::min(batchSize, static_cast<int>(order.size()));
    auto shuffledA = trainingData.nextBatchShuffled(order, 0, shuffledBatchSize);
    auto shuffledB = trainingData.nextBatchShuffled(order, 0, shuffledBatchSize);
    for (int i = 0; i < shuffledBatchSize; ++i) {
        expectNear(shuffledA.second(i), shuffledB.second(i), kTol, "nextBatchShuffled should be deterministic for a fixed order");
    }

    auto validationBatch = validationData.nextBatch();
    const int currentBatch = static_cast<int>(validationBatch.second.size());

    LSTMCell::setGlobalInitSeed(lstmSeed);
    LSTMCell lstm(numFeatures, hiddenSize, sequenceLength);
    Dense dense(hiddenSize, /*initSeed=*/424242u);

    HuberLossFunction huber(1.0);

    DeterminismRunResult out{};
    out.lstmParamsBefore = lstm.getParametersVector();
    out.denseWBefore = dense.W;
    out.denseBBefore = dense.b;

    out.predsBefore = Eigen::VectorXd::Zero(currentBatch);
    std::vector<Eigen::VectorXd> hiddenPerSample;
    hiddenPerSample.reserve(static_cast<std::size_t>(currentBatch));

    for (int i = 0; i < currentBatch; ++i) {
        lstm.reset();
        Eigen::VectorXd hidden;

        for (int t = 0; t < sequenceLength; ++t) {
            Eigen::VectorXd x(numFeatures);
            for (int f = 0; f < numFeatures; ++f) {
                x(f) = validationBatch.first(i, t, f);
            }
            hidden = lstm.forwardPass(x);
        }

        hiddenPerSample.push_back(hidden);
        out.predsBefore(i) = dense.forward(hidden);
    }

    out.lossBefore = huber.forward(out.predsBefore, validationBatch.second);
    expectTrue(std::isfinite(out.lossBefore), "loss before update should be finite");

    Eigen::VectorXd dLdy = huber.backward();

    lstm.zeroGrad();
    dense.zeroGrad();

    for (int i = 0; i < currentBatch; ++i) {
        dense.backward(hiddenPerSample[static_cast<std::size_t>(i)], dLdy(i));

        lstm.reset();
        for (int t = 0; t < sequenceLength; ++t) {
            Eigen::VectorXd x(numFeatures);
            for (int f = 0; f < numFeatures; ++f) {
                x(f) = validationBatch.first(i, t, f);
            }
            (void)lstm.forwardPass(x);
        }

        Eigen::VectorXd dLdh = dense.W.transpose() * dLdy(i);
        Eigen::VectorXd dLdc = Eigen::VectorXd::Zero(hiddenSize);
        for (int t = sequenceLength; t > 0; --t) {
            std::tie(dLdh, dLdc) = lstm.backwardPass(dLdh, dLdc);
        }
    }

    Eigen::VectorXd lstmParams = lstm.getParametersVector();
    const Eigen::VectorXd lstmGrads = lstm.getGradientsVector();

    AdaBelief optimiser(static_cast<std::size_t>(lstmParams.size()), learningRate);
    optimiser.update(lstmParams, lstmGrads);
    lstm.setParametersVector(lstmParams);

    dense.W -= learningRate * dense.dW;
    dense.b -= learningRate * dense.db;

    out.lstmParamsAfter = lstm.getParametersVector();
    out.denseWAfter = dense.W;
    out.denseBAfter = dense.b;

    out.predsAfter = Eigen::VectorXd::Zero(currentBatch);
    for (int i = 0; i < currentBatch; ++i) {
        lstm.reset();
        Eigen::VectorXd hidden;

        for (int t = 0; t < sequenceLength; ++t) {
            Eigen::VectorXd x(numFeatures);
            for (int f = 0; f < numFeatures; ++f) {
                x(f) = validationBatch.first(i, t, f);
            }
            hidden = lstm.forwardPass(x);
        }

        out.predsAfter(i) = dense.forward(hidden);
    }

    out.lossAfter = huber.forward(out.predsAfter, validationBatch.second);
    expectTrue(std::isfinite(out.lossAfter), "loss after update should be finite");

    return out;
}

void test_same_seed_repeated_path_matches_before_and_after_tiny_update() {
    const DeterminismRunResult runA = runOneDeterministicPath(2026u);
    const DeterminismRunResult runB = runOneDeterministicPath(2026u);

    expectVectorNear(runA.lstmParamsBefore, runB.lstmParamsBefore, kTol,
                     "same-seed repeated run LSTM initial params should match");
    expectRowVectorNear(runA.denseWBefore, runB.denseWBefore, kTol,
                        "same controlled Dense initial weights should match");
    expectNear(runA.denseBBefore, runB.denseBBefore, kTol,
               "same controlled Dense initial bias should match");

    expectVectorNear(runA.predsBefore, runB.predsBefore, kTol,
                     "same-seed repeated run predictions before update should match");
    expectNear(runA.lossBefore, runB.lossBefore, kTol,
               "same-seed repeated run loss before update should match");

    expectVectorNear(runA.lstmParamsAfter, runB.lstmParamsAfter, kTol,
                     "same-seed repeated run LSTM params after tiny update should match");
    expectRowVectorNear(runA.denseWAfter, runB.denseWAfter, kTol,
                        "same controlled Dense weights after tiny update should match");
    expectNear(runA.denseBAfter, runB.denseBAfter, kTol,
               "same controlled Dense bias after tiny update should match");

    expectVectorNear(runA.predsAfter, runB.predsAfter, kTol,
                     "same-seed repeated run predictions after update should match");
    expectNear(runA.lossAfter, runB.lossAfter, kTol,
               "same-seed repeated run loss after update should match");
}

void test_different_seed_changes_lstm_controlled_path_outputs() {
    const DeterminismRunResult runA = runOneDeterministicPath(101u);
    const DeterminismRunResult runB = runOneDeterministicPath(202u);

    double maxParamDiff = 0.0;
    for (int i = 0; i < runA.lstmParamsBefore.size(); ++i) {
        const double d = std::abs(runA.lstmParamsBefore(i) - runB.lstmParamsBefore(i));
        if (d > maxParamDiff) {
            maxParamDiff = d;
        }
    }

    expectTrue(maxParamDiff > kTol,
               "different LSTM seeds should change initial LSTM parameter vectors in controlled path");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"same-seed repeated controlled path matches before and after tiny update", test_same_seed_repeated_path_matches_before_and_after_tiny_update},
        {"different LSTM seeds alter controlled path initialization", test_different_seed_changes_lstm_controlled_path_outputs},
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

    std::cout << "\nEnd-to-end determinism audit tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
