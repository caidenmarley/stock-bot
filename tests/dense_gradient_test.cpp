#include "model/dense.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace {

constexpr double kEps = 1e-6;
constexpr double kTol = 1e-6;

void expectNear(double actual, double expected, const std::string& msg, double tol = kTol) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(msg + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

double squaredLossFromDense(const Dense& dense, const Eigen::VectorXd& h, double target) {
    const double y = dense.forward(h);
    const double residual = y - target;
    return 0.5 * residual * residual;
}

Dense makeDeterministicDense() {
    Dense dense(3);
    dense.W << 0.5, -0.25, 0.75;
    dense.b = -0.1;
    dense.zeroGrad();
    return dense;
}

Eigen::VectorXd makeHidden() {
    Eigen::VectorXd h(3);
    h << 0.2, -0.4, 0.6;
    return h;
}

void test_dense_forward_expected() {
    Dense dense = makeDeterministicDense();
    const Eigen::VectorXd h = makeHidden();

    const double expected = dense.W.dot(h) + dense.b;
    const double actual = dense.forward(h);
    expectNear(actual, expected, "Dense forward y = W dot h + b mismatch");
}

void test_dense_backward_analytical() {
    Dense dense = makeDeterministicDense();
    const Eigen::VectorXd h = makeHidden();
    const double target = 0.3;

    const double y = dense.forward(h);
    const double dLdy = y - target; // for 0.5*(y-target)^2

    dense.zeroGrad();
    dense.backward(h, dLdy);

    const Eigen::RowVectorXd expectedDW = dLdy * h.transpose();
    const double expectedDb = dLdy;

    for (int i = 0; i < expectedDW.size(); ++i) {
        expectNear(dense.dW(i), expectedDW(i), "Dense dW analytical mismatch at index " + std::to_string(i));
    }
    expectNear(dense.db, expectedDb, "Dense db analytical mismatch");
}

void test_dense_numerical_gradient_weights() {
    Dense dense = makeDeterministicDense();
    const Eigen::VectorXd h = makeHidden();
    const double target = 0.3;

    const double y = dense.forward(h);
    const double dLdy = y - target;

    dense.zeroGrad();
    dense.backward(h, dLdy);

    for (int i = 0; i < dense.W.size(); ++i) {
        Dense plus = dense;
        Dense minus = dense;
        plus.W(i) += kEps;
        minus.W(i) -= kEps;

        const double lossPlus = squaredLossFromDense(plus, h, target);
        const double lossMinus = squaredLossFromDense(minus, h, target);
        const double numerical = (lossPlus - lossMinus) / (2.0 * kEps);

        expectNear(dense.dW(i), numerical, "Dense weight gradient check failed at index " + std::to_string(i), 1e-5);
    }
}

void test_dense_numerical_gradient_bias() {
    Dense dense = makeDeterministicDense();
    const Eigen::VectorXd h = makeHidden();
    const double target = 0.3;

    const double y = dense.forward(h);
    const double dLdy = y - target;

    dense.zeroGrad();
    dense.backward(h, dLdy);

    Dense plus = dense;
    Dense minus = dense;
    plus.b += kEps;
    minus.b -= kEps;

    const double lossPlus = squaredLossFromDense(plus, h, target);
    const double lossMinus = squaredLossFromDense(minus, h, target);
    const double numerical = (lossPlus - lossMinus) / (2.0 * kEps);

    expectNear(dense.db, numerical, "Dense bias gradient check failed", 1e-5);
}

void test_zero_grad_clears() {
    Dense dense = makeDeterministicDense();
    const Eigen::VectorXd h = makeHidden();

    dense.backward(h, 1.0);
    dense.backward(h, -0.5);

    dense.zeroGrad();

    for (int i = 0; i < dense.dW.size(); ++i) {
        expectNear(dense.dW(i), 0.0, "zeroGrad should clear dW");
    }
    expectNear(dense.db, 0.0, "zeroGrad should clear db");
}

void test_gradient_accumulation_without_zero_grad() {
    Dense dense = makeDeterministicDense();
    const Eigen::VectorXd h = makeHidden();

    const double g1 = 0.7;
    const double g2 = -0.2;

    dense.zeroGrad();
    dense.backward(h, g1);
    dense.backward(h, g2);

    const Eigen::RowVectorXd expectedDW = (g1 + g2) * h.transpose();
    const double expectedDb = (g1 + g2);

    for (int i = 0; i < expectedDW.size(); ++i) {
        expectNear(dense.dW(i), expectedDW(i), "Gradient accumulation dW mismatch at index " + std::to_string(i));
    }
    expectNear(dense.db, expectedDb, "Gradient accumulation db mismatch");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"Dense forward expected value", test_dense_forward_expected},
        {"Dense backward analytical gradients", test_dense_backward_analytical},
        {"Dense numerical gradient check for weights", test_dense_numerical_gradient_weights},
        {"Dense numerical gradient check for bias", test_dense_numerical_gradient_bias},
        {"Dense zeroGrad clears gradients", test_zero_grad_clears},
        {"Dense gradient accumulation without zeroGrad", test_gradient_accumulation_without_zero_grad},
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

    std::cout << "\nDense gradient tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
