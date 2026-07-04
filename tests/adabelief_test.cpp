#include "model/ada_belief.h"

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

void expectNear(double actual, double expected, const std::string& msg) {
    if (std::abs(actual - expected) > kTol) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

void expectVecNear(const Eigen::VectorXd& actual, const Eigen::VectorXd& expected, const std::string& msg) {
    expectTrue(actual.size() == expected.size(), msg + " size mismatch");
    for (int i = 0; i < actual.size(); ++i) {
        expectNear(actual(i), expected(i), msg + " index=" + std::to_string(i));
    }
}

void test_zero_gradient_no_parameter_change() {
    Eigen::VectorXd params(3);
    params << 1.0, -2.0, 0.5;

    const Eigen::VectorXd original = params;
    const Eigen::VectorXd grads = Eigen::VectorXd::Zero(3);

    AdaBelief opt(/*vecSize=*/3, /*learningRate=*/1e-3);
    opt.update(params, grads);

    expectVecNear(params, original, "zero-gradient update should keep parameters unchanged");
}

void test_one_step_update_matches_hand_computed_formula() {
    const double lr = 1e-3;
    const double b1 = 0.9;
    const double b2 = 0.999;
    const double eps = 1e-8;

    Eigen::VectorXd params(2);
    params << 1.0, 1.0;

    Eigen::VectorXd grads(2);
    grads << 0.1, -0.2;

    // Hand-computed from implemented formula:
    // m1 = (1-b1) * g
    // s1 = (1-b2) * (g - m1)^2 = (1-b2) * (b1*g)^2
    // mHat1 = m1 / (1-b1) = g
    // sHat1 = s1 / (1-b2) = (b1*g)^2
    // param -= lr * mHat1 / (sqrt(sHat1) + eps)
    Eigen::VectorXd expected(2);
    expected(0) = 1.0 - (lr * 0.1) / (std::sqrt((b1 * 0.1) * (b1 * 0.1)) + eps);
    expected(1) = 1.0 - (lr * -0.2) / (std::sqrt((b1 * -0.2) * (b1 * -0.2)) + eps);

    AdaBelief opt(/*vecSize=*/2, lr, b1, b2, eps);
    opt.update(params, grads);

    expectVecNear(params, expected, "one-step AdaBelief update should match hand-computed implementation formula");
}

void test_opposite_gradient_signs_move_parameters_in_opposite_directions() {
    Eigen::VectorXd params(2);
    params << 0.5, 0.5;

    Eigen::VectorXd grads(2);
    grads << 0.3, -0.3;

    AdaBelief opt(/*vecSize=*/2, /*learningRate=*/1e-3);
    opt.update(params, grads);

    expectTrue(params(0) < 0.5, "positive gradient should reduce parameter value");
    expectTrue(params(1) > 0.5, "negative gradient should increase parameter value");
}

void test_two_step_update_matches_closed_form_for_constant_gradient() {
    const double lr = 0.01;
    const double b1 = 0.8;
    const double b2 = 0.9;
    const double eps = 1e-8;
    const double g = 0.5;

    Eigen::VectorXd params(1);
    params << 1.0;

    Eigen::VectorXd grads(1);
    grads << g;

    // Step 1:
    // mHat1 = g, sHat1 = (b1*g)^2
    const double sHat1 = (b1 * g) * (b1 * g);
    const double update1 = lr * g / (std::sqrt(sHat1) + eps);

    // Step 2 closed form under constant gradient:
    // m2 = (1-b1^2)g => mHat2 = g
    // s2 = (1-b2)(b2*b1^2 + b1^4)g^2
    // sHat2 = s2/(1-b2^2) = ((b2*b1^2 + b1^4)/(1+b2))g^2
    const double sHat2 = ((b2 * b1 * b1 + std::pow(b1, 4.0)) / (1.0 + b2)) * g * g;
    const double update2 = lr * g / (std::sqrt(sHat2) + eps);

    const double expectedAfterTwo = 1.0 - update1 - update2;

    AdaBelief opt(/*vecSize=*/1, lr, b1, b2, eps);
    opt.update(params, grads);
    opt.update(params, grads);

    expectNear(params(0), expectedAfterTwo,
               "two-step constant-gradient update should match closed-form implementation behavior");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"zero gradient no change", test_zero_gradient_no_parameter_change},
        {"one-step hand-computed update", test_one_step_update_matches_hand_computed_formula},
        {"opposite gradient signs move opposite", test_opposite_gradient_signs_move_parameters_in_opposite_directions},
        {"two-step closed-form constant gradient", test_two_step_update_matches_closed_form_for_constant_gradient},
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

    std::cout << "\nAdaBelief tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
