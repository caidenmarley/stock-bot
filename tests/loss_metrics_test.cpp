#include "model/huber_loss_function.h"
#include "model/metrics.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

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

void test_huber_forward_quadratic_region() {
    HuberLossFunction loss(/*delta=*/1.0);
    const double prediction = 0.8;
    const double target = 1.0;
    const double residual = target - prediction; // 0.2
    const double expected = 0.5 * residual * residual; // 0.02
    expectNear(loss.forward(prediction, target), expected, "Huber forward quadratic region");
}

void test_huber_forward_linear_region() {
    HuberLossFunction loss(/*delta=*/1.0);
    const double prediction = 0.0;
    const double target = 2.0;
    const double residual = target - prediction; // 2.0
    const double expected = 1.0 * (std::abs(residual) - 0.5 * 1.0); // 1.5
    expectNear(loss.forward(prediction, target), expected, "Huber forward linear region");
}

void test_huber_batch_averaging() {
    HuberLossFunction loss(/*delta=*/1.0);

    Eigen::VectorXd predictions(2);
    Eigen::VectorXd targets(2);
    predictions << 0.8, 0.0;
    targets << 1.0, 2.0;

    // residuals: [0.2, 2.0]
    // losses: [0.5*0.2^2 = 0.02, 1*(2 - 0.5) = 1.5]
    // mean: 0.76
    const double expectedMean = (0.02 + 1.5) / 2.0;
    expectNear(loss.forward(predictions, targets), expectedMean, "Huber batch mean loss");
}

void test_huber_backward_gradient_sign_quadratic() {
    HuberLossFunction loss(/*delta=*/1.0);

    Eigen::VectorXd predictions(2);
    Eigen::VectorXd targets(2);
    predictions << 0.8, 1.1;
    targets << 1.0, 1.0;

    // residuals: [0.2, -0.1], quadratic for both
    // grad = -residual / n => [-0.1, +0.05]
    (void)loss.forward(predictions, targets);
    Eigen::VectorXd grad = loss.backward();

    expectNear(grad(0), -0.1, "Quadratic grad value for positive residual");
    expectNear(grad(1), 0.05, "Quadratic grad value for negative residual");
    expectTrue(grad(0) < 0.0, "Gradient should be negative for positive residual");
    expectTrue(grad(1) > 0.0, "Gradient should be positive for negative residual");
}

void test_huber_backward_linear_clipping() {
    HuberLossFunction loss(/*delta=*/1.0);

    Eigen::VectorXd predictions(2);
    Eigen::VectorXd targets(2);
    predictions << 0.0, 5.0;
    targets << 2.0, 1.0;

    // residuals: [2.0, -4.0], linear for both
    // grad magnitude should be delta / n = 1 / 2 = 0.5
    (void)loss.forward(predictions, targets);
    Eigen::VectorXd grad = loss.backward();

    expectNear(std::abs(grad(0)), 0.5, "Linear-region gradient magnitude #1");
    expectNear(std::abs(grad(1)), 0.5, "Linear-region gradient magnitude #2");
    expectTrue(grad(0) < 0.0, "Linear grad should be negative for positive residual");
    expectTrue(grad(1) > 0.0, "Linear grad should be positive for negative residual");
}

void test_predictions_to_positions() {
    const std::vector<double> predictions = {0.2, 0.0, -0.1, 0.01};
    const double threshold = 0.0;
    const std::vector<double> expected = {1.0, 0.0, 0.0, 1.0};

    const auto positions = metrics::predictionsToPositions(predictions, threshold);
    expectTrue(positions.size() == expected.size(), "positions size");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expectNear(positions[i], expected[i], "predictionsToPositions mismatch at index " + std::to_string(i));
    }
}

void test_calc_daily_pnl_and_turnover() {
    const std::vector<double> positions = {0.0, 1.0, 1.0, 0.0};
    const std::vector<double> actualReturns = {0.01, 0.02, -0.01, 0.03};
    const double costToChangePos = 0.001;

    const auto out = metrics::calcDailyPnLAndTurnover(positions, actualReturns, costToChangePos);

    const std::vector<double> expectedGross = {0.0, 0.02, -0.01, 0.0};
    const std::vector<double> expectedTurnover = {0.0, 1.0, 0.0, 1.0};
    const std::vector<double> expectedNet = {0.0, 0.019, -0.01, -0.001};

    for (std::size_t i = 0; i < positions.size(); ++i) {
        expectNear(out.grossReturn[i], expectedGross[i], "gross return mismatch " + std::to_string(i));
        expectNear(out.turnover[i], expectedTurnover[i], "turnover mismatch " + std::to_string(i));
        const double expectedTradingCost = costToChangePos * expectedTurnover[i];
        expectNear(out.grossReturn[i] - out.netReturn[i], expectedTradingCost, "trading cost mismatch " + std::to_string(i));
        expectNear(out.netReturn[i], expectedNet[i], "net return mismatch " + std::to_string(i));
    }
}

void test_mean_and_stddev() {
    const std::vector<double> vec = {1.0, 2.0, 3.0, 4.0};
    const double expectedMean = 2.5;
    const double expectedStd = std::sqrt(5.0 / 3.0); // sample stddev with Bessel correction

    expectNear(metrics::mean(vec), expectedMean, "mean mismatch");
    expectNear(metrics::stddev(vec), expectedStd, "sample stddev mismatch");
}

void test_sharpe_edge_and_nonzero() {
    // too-small input
    expectNear(metrics::sharpe({0.01}, 252), 0.0, "sharpe should be 0 for size<2");

    // zero stddev
    expectNear(metrics::sharpe({0.01, 0.01, 0.01}, 252), 0.0, "sharpe should be 0 for zero stddev");

    // simple non-zero case
    const std::vector<double> daily = {0.01, 0.03};
    const double m = 0.02;
    const double s = std::sqrt(0.0002); // sample std for two-point set
    const double expected = (m / s) * std::sqrt(252.0);
    expectNear(metrics::sharpe(daily, 252), expected, "non-zero sharpe mismatch", 1e-8);
}

void test_calc_sharpe_and_turnover_pipeline() {
    const std::vector<double> predictions = {0.02, -0.01, 0.03, 0.0};
    const std::vector<double> actualReturns = {0.01, 0.02, -0.01, 0.03};

    metrics::ProfitAndLossParams params{};
    params.thresholdToEnter = 0.0;
    params.costToChangePos = 0.001;
    params.periodsPerYear = 252;

    const auto out = metrics::calcSharpeAndTurnover(predictions, actualReturns, params);

    // Expected positions from threshold rule (> threshold): [1, 0, 1, 0]
    // Turnover with prev flat: [1, 1, 1, 1] => avg 1
    // Net returns: [0.009, -0.001, -0.011, -0.001]
    const std::vector<double> expectedNet = {0.009, -0.001, -0.011, -0.001};

    const double expectedMean = (-0.001);
    double sumSquares = 0.0;
    for (double v : expectedNet) {
        const double d = v - expectedMean;
        sumSquares += d * d;
    }
    const double expectedStd = std::sqrt(sumSquares / (expectedNet.size() - 1.0));
    const double expectedSharpe = (expectedMean / expectedStd) * std::sqrt(252.0);

    expectNear(out.avgTurnover, 1.0, "avgTurnover mismatch");
    expectNear(out.sharpeNet, expectedSharpe, "pipeline sharpe mismatch", 1e-8);
}

} // namespace

int main() {
    // Note: predictionsToScaledPositions is declared in headers but not implemented in src/model/metrics.cpp.
    // This test suite intentionally does not call it to avoid linker failures; the gap is documented in PROJECT_STATE.md.
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"Huber forward quadratic region", test_huber_forward_quadratic_region},
        {"Huber forward linear region", test_huber_forward_linear_region},
        {"Huber batch averaging", test_huber_batch_averaging},
        {"Huber backward gradient sign", test_huber_backward_gradient_sign_quadratic},
        {"Huber backward linear clipping", test_huber_backward_linear_clipping},
        {"predictionsToPositions threshold behavior", test_predictions_to_positions},
        {"calcDailyPnLAndTurnover hand-check", test_calc_daily_pnl_and_turnover},
        {"mean and sample stddev", test_mean_and_stddev},
        {"sharpe edge + non-zero", test_sharpe_edge_and_nonzero},
        {"calcSharpeAndTurnover pipeline", test_calc_sharpe_and_turnover_pipeline},
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

    std::cout << "\nLoss/Metrics tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
