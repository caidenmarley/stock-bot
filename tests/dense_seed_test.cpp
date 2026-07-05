#include "model/dense.h"

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

double maxAbsDiff(const Eigen::RowVectorXd& a, const Eigen::RowVectorXd& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("size mismatch in maxAbsDiff");
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

void test_same_seed_same_dimensions_identical_weights_and_bias() {
    Dense a(/*hiddenSize=*/7, /*initSeed=*/123u);
    Dense b(/*hiddenSize=*/7, /*initSeed=*/123u);

    expectTrue(maxAbsDiff(a.W, b.W) <= kTol,
               "same seed + same dimensions should produce identical Dense weights");
    expectNear(a.b, b.b, "same seed + same dimensions should produce identical Dense bias");
}

void test_different_seed_changes_weights() {
    Dense a(/*hiddenSize=*/7, /*initSeed=*/111u);
    Dense b(/*hiddenSize=*/7, /*initSeed=*/222u);

    expectTrue(maxAbsDiff(a.W, b.W) > kTol,
               "different seeds should produce different Dense weights in tested case");
}

void test_unseeded_constructor_still_has_expected_shapes_and_zero_bias() {
    Dense d(/*hiddenSize=*/5);

    expectTrue(d.W.size() == 5, "unseeded Dense should have expected weight shape");
    expectTrue(d.dW.size() == 5, "unseeded Dense should have expected gradient shape");
    expectNear(d.b, 0.0, "unseeded Dense bias should initialize to zero");
    expectNear(d.db, 0.0, "unseeded Dense db should initialize to zero");
}

void test_seeded_forward_repeatable_for_same_input() {
    Dense a(/*hiddenSize=*/4, /*initSeed=*/77u);
    Dense b(/*hiddenSize=*/4, /*initSeed=*/77u);

    Eigen::VectorXd h(4);
    h << 0.2, -0.3, 0.1, 0.5;

    const double outA = a.forward(h);
    const double outB = b.forward(h);

    expectNear(outA, outB, "seeded Dense forward should be repeatable for same input and seed");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"same seed + same dimensions -> identical Dense init", test_same_seed_same_dimensions_identical_weights_and_bias},
        {"different seed -> different Dense weights", test_different_seed_changes_weights},
        {"unseeded constructor remains valid", test_unseeded_constructor_still_has_expected_shapes_and_zero_bias},
        {"seeded Dense forward is repeatable", test_seeded_forward_repeatable_for_same_input},
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

    std::cout << "\nDense seed tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
