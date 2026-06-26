#include "model/lstm.h"

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

double maxAbsDiff(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("vector size mismatch in maxAbsDiff");
    }

    double m = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        const double d = std::abs(a(i) - b(i));
        if (d > m) {
            m = d;
        }
    }
    return m;
}

Eigen::VectorXd runSequence(LSTMCell& model, const std::vector<Eigen::VectorXd>& sequence) {
    model.reset();
    Eigen::VectorXd h;
    for (const auto& x : sequence) {
        h = model.forwardPass(x);
    }
    return h;
}

void test_same_seed_same_dimensions_identical_lstm_parameters() {
    const int numFeatures = 3;
    const int hiddenSize = 5;
    const int sequenceLength = 4;

    LSTMCell::setGlobalInitSeed(123u);
    LSTMCell modelA(numFeatures, hiddenSize, sequenceLength);
    const Eigen::VectorXd paramsA = modelA.getParametersVector();

    LSTMCell::setGlobalInitSeed(123u);
    LSTMCell modelB(numFeatures, hiddenSize, sequenceLength);
    const Eigen::VectorXd paramsB = modelB.getParametersVector();

    const double diff = maxAbsDiff(paramsA, paramsB);
    expectTrue(diff <= kTol, "same seed should produce identical LSTM parameter vector");
}

void test_different_seed_same_dimensions_different_lstm_parameters() {
    const int numFeatures = 3;
    const int hiddenSize = 5;
    const int sequenceLength = 4;

    LSTMCell::setGlobalInitSeed(111u);
    LSTMCell modelA(numFeatures, hiddenSize, sequenceLength);
    const Eigen::VectorXd paramsA = modelA.getParametersVector();

    LSTMCell::setGlobalInitSeed(222u);
    LSTMCell modelB(numFeatures, hiddenSize, sequenceLength);
    const Eigen::VectorXd paramsB = modelB.getParametersVector();

    const double diff = maxAbsDiff(paramsA, paramsB);
    expectTrue(diff > kTol, "different seeds should produce a different LSTM parameter vector");
}

void test_same_parameters_and_input_sequence_identical_forward_outputs() {
    const int numFeatures = 3;
    const int hiddenSize = 5;
    const int sequenceLength = 4;

    LSTMCell::setGlobalInitSeed(77u);
    LSTMCell modelA(numFeatures, hiddenSize, sequenceLength);
    const Eigen::VectorXd sharedParams = modelA.getParametersVector();

    LSTMCell::setGlobalInitSeed(999u);
    LSTMCell modelB(numFeatures, hiddenSize, sequenceLength);
    modelB.setParametersVector(sharedParams);

    std::vector<Eigen::VectorXd> seq;
    seq.reserve(static_cast<std::size_t>(sequenceLength));

    Eigen::VectorXd x0(numFeatures); x0 << 0.10, -0.20, 0.05;
    Eigen::VectorXd x1(numFeatures); x1 << -0.30, 0.25, 0.15;
    Eigen::VectorXd x2(numFeatures); x2 << 0.40, -0.10, -0.35;
    Eigen::VectorXd x3(numFeatures); x3 << 0.20, 0.30, -0.05;

    seq.push_back(x0);
    seq.push_back(x1);
    seq.push_back(x2);
    seq.push_back(x3);

    const Eigen::VectorXd outA1 = runSequence(modelA, seq);
    const Eigen::VectorXd outA2 = runSequence(modelA, seq);
    const Eigen::VectorXd outB1 = runSequence(modelB, seq);

    expectTrue(maxAbsDiff(outA1, outA2) <= kTol, "same model + same input sequence should be deterministic after reset");
    expectTrue(maxAbsDiff(outA1, outB1) <= kTol, "same parameters + same input sequence should match across models");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"same seed -> identical LSTM params", test_same_seed_same_dimensions_identical_lstm_parameters},
        {"different seed -> different LSTM params", test_different_seed_same_dimensions_different_lstm_parameters},
        {"same params + same inputs -> identical forwards", test_same_parameters_and_input_sequence_identical_forward_outputs},
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

    std::cout << "\nReproducibility tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
