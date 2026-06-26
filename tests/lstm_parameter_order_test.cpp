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

void expectNear(double actual, double expected, const std::string& msg, double tol = kTol) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

LSTMCell makeSmallLSTM() {
    return LSTMCell(/*inputSize=*/2, /*hiddenSize=*/3, /*sequenceLength=*/2);
}

void test_parameter_count_formula() {
    LSTMCell lstm = makeSmallLSTM();

    const int inputSize = 2;
    const int hiddenSize = 3;

    const std::size_t expected = static_cast<std::size_t>(4 * hiddenSize * (hiddenSize + inputSize) + 4 * hiddenSize);

    const auto& params = lstm.getParametersVector();
    expectTrue(static_cast<std::size_t>(params.size()) == expected, "getParametersVector size should match expected count");
    expectTrue(lstm.getParameterCount() == expected, "getParameterCount should match expected formula");
}

void test_set_get_roundtrip_deterministic_vector() {
    LSTMCell lstm = makeSmallLSTM();
    const int n = lstm.getParametersVector().size();

    Eigen::VectorXd deterministic(n);
    for (int i = 0; i < n; ++i) {
        deterministic(i) = 0.001 * static_cast<double>(i + 1);
    }

    lstm.setParametersVector(deterministic);
    const auto& roundTrip = lstm.getParametersVector();

    expectTrue(roundTrip.size() == deterministic.size(), "round-trip vector size mismatch");
    for (int i = 0; i < n; ++i) {
        expectNear(roundTrip(i), deterministic(i), "set/get round-trip mismatch at index " + std::to_string(i));
    }
}

void test_single_index_mutation_affects_single_index() {
    LSTMCell lstm = makeSmallLSTM();
    const int paramCount = lstm.getParametersVector().size();

    Eigen::VectorXd params(paramCount);
    for (int i = 0; i < paramCount; ++i) {
        params(i) = 0.001 * static_cast<double>(i + 1);
    }

    lstm.setParametersVector(params);

    Eigen::VectorXd mutated = params;
    const int idx = paramCount / 2;
    mutated(idx) += 0.123456;

    lstm.setParametersVector(mutated);
    const auto& after = lstm.getParametersVector();

    expectTrue(after.size() == mutated.size(), "single-index mutation size mismatch");
    for (int i = 0; i < after.size(); ++i) {
        expectNear(after(i), mutated(i), "single-index mutation round-trip mismatch at index " + std::to_string(i));
    }
}

void test_gradient_vector_length_matches_params() {
    LSTMCell lstm = makeSmallLSTM();
    const auto& params = lstm.getParametersVector();
    const auto& grads = lstm.getGradientsVector();

    expectTrue(grads.size() == params.size(), "gradient vector length should match parameter vector length");
}

void test_zero_grad_produces_all_zeros() {
    LSTMCell lstm = makeSmallLSTM();
    lstm.zeroGrad();
    const auto& grads = lstm.getGradientsVector();

    for (int i = 0; i < grads.size(); ++i) {
        expectNear(grads(i), 0.0, "gradient should be zero after zeroGrad at index " + std::to_string(i));
    }
}

void test_tiny_forward_backward_gradients_finite() {
    LSTMCell lstm = makeSmallLSTM();
    lstm.reset();
    lstm.zeroGrad();

    Eigen::VectorXd x(2);
    x << 0.1, -0.2;

    (void)lstm.forwardPass(x);

    Eigen::VectorXd deltaH = Eigen::VectorXd::Constant(3, 0.01);
    Eigen::VectorXd deltaC = Eigen::VectorXd::Zero(3);
    (void)lstm.backwardPass(deltaH, deltaC);

    const auto& params = lstm.getParametersVector();
    const auto& grads = lstm.getGradientsVector();

    expectTrue(grads.size() == params.size(), "finite-gradient check size mismatch");
    for (int i = 0; i < grads.size(); ++i) {
        expectTrue(std::isfinite(grads(i)), "non-finite gradient detected at index " + std::to_string(i));
    }
}

} // namespace

int main() {
    // Observed flat ordering from implementation (get/set/gradient vectors):
    // Wf, Uf, bf, Wi, Ui, bi, Wc, Uc, bc, Wo, Uo, bo
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"parameter count formula", test_parameter_count_formula},
        {"set/get deterministic round-trip", test_set_get_roundtrip_deterministic_vector},
        {"single-index mutation safety", test_single_index_mutation_affects_single_index},
        {"gradient vector length matches params", test_gradient_vector_length_matches_params},
        {"zeroGrad returns all zeros", test_zero_grad_produces_all_zeros},
        {"tiny forward/backward finite gradients", test_tiny_forward_backward_gradients_finite},
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

    std::cout << "\nLSTM parameter-order tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
