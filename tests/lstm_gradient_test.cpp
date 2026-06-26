#include "model/lstm.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kInputSize = 2;
constexpr int kHiddenSize = 4;
constexpr int kSequenceLength = 3;
constexpr double kEpsilon = 1e-5;
constexpr double kAbsTol = 1e-4;
constexpr double kRelTol = 1e-3;

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

std::vector<Eigen::VectorXd> makeInputSequence() {
    std::vector<Eigen::VectorXd> seq;
    seq.reserve(kSequenceLength);

    Eigen::VectorXd x0(kInputSize), x1(kInputSize), x2(kInputSize);
    x0 << 0.10, -0.20;
    x1 << 0.05, 0.30;
    x2 << -0.15, 0.07;

    seq.push_back(x0);
    seq.push_back(x1);
    seq.push_back(x2);
    return seq;
}

Eigen::VectorXd makeTarget() {
    Eigen::VectorXd target(kHiddenSize);
    for (int j = 0; j < kHiddenSize; ++j) {
        target(j) = 0.03 * static_cast<double>(j + 1);
    }
    return target;
}

Eigen::VectorXd makeDeterministicParams(int paramCount) {
    Eigen::VectorXd params(paramCount);
    for (int i = 0; i < paramCount; ++i) {
        params(i) = 0.01 * std::sin(static_cast<double>(i + 1));
    }
    return params;
}

double computeLoss(LSTMCell& lstm,
                   const Eigen::VectorXd& params,
                   const std::vector<Eigen::VectorXd>& sequence,
                   const Eigen::VectorXd& target) {
    lstm.setParametersVector(params);
    lstm.reset();

    for (const auto& x : sequence) {
        (void)lstm.forwardPass(x);
    }

    const Eigen::VectorXd hFinal = lstm.getHiddenState();
    const Eigen::VectorXd diff = hFinal - target;
    return 0.5 * diff.squaredNorm();
}

Eigen::VectorXd computeAnalyticalGradient(LSTMCell& lstm,
                                          const Eigen::VectorXd& params,
                                          const std::vector<Eigen::VectorXd>& sequence,
                                          const Eigen::VectorXd& target) {
    lstm.setParametersVector(params);
    lstm.reset();
    lstm.zeroGrad();

    for (const auto& x : sequence) {
        (void)lstm.forwardPass(x);
    }

    Eigen::VectorXd deltaH = lstm.getHiddenState() - target; // dL/dh_final
    Eigen::VectorXd deltaC = Eigen::VectorXd::Zero(kHiddenSize); // dL/dc_final

    for (int t = static_cast<int>(sequence.size()) - 1; t >= 0; --t) {
        (void)t;
        const auto back = lstm.backwardPass(deltaH, deltaC);
        deltaH = back.first;
        deltaC = back.second;
    }

    return lstm.getGradientsVector();
}

void test_lstm_bptt_finite_difference_gradient() {
    LSTMCell lstm(kInputSize, kHiddenSize, kSequenceLength);

    const auto sequence = makeInputSequence();
    const auto target = makeTarget();

    const int paramCount = lstm.getParametersVector().size();
    const Eigen::VectorXd params = makeDeterministicParams(paramCount);

    const Eigen::VectorXd analytical = computeAnalyticalGradient(lstm, params, sequence, target);

    expectTrue(analytical.size() == paramCount, "Analytical gradient size must match parameter vector size");
    for (int i = 0; i < analytical.size(); ++i) {
        expectTrue(std::isfinite(analytical(i)), "Analytical gradient has NaN/Inf at index " + std::to_string(i));
    }

    Eigen::VectorXd numerical(paramCount);
    for (int i = 0; i < paramCount; ++i) {
        Eigen::VectorXd plus = params;
        Eigen::VectorXd minus = params;
        plus(i) += kEpsilon;
        minus(i) -= kEpsilon;

        const double lossPlus = computeLoss(lstm, plus, sequence, target);
        const double lossMinus = computeLoss(lstm, minus, sequence, target);
        numerical(i) = (lossPlus - lossMinus) / (2.0 * kEpsilon);
    }

    expectTrue(numerical.size() == paramCount, "Numerical gradient size must match parameter vector size");
    for (int i = 0; i < numerical.size(); ++i) {
        expectTrue(std::isfinite(numerical(i)), "Numerical gradient has NaN/Inf at index " + std::to_string(i));
    }

    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    int worstIndex = -1;
    double worstA = 0.0;
    double worstN = 0.0;

    for (int i = 0; i < paramCount; ++i) {
        const double a = analytical(i);
        const double n = numerical(i);
        const double absErr = std::abs(a - n);
        const double denom = std::max({1.0, std::abs(a), std::abs(n)});
        const double relErr = absErr / denom;

        if (absErr > maxAbsErr) {
            maxAbsErr = absErr;
        }
        if (relErr > maxRelErr) {
            maxRelErr = relErr;
            worstIndex = i;
            worstA = a;
            worstN = n;
        }
    }

    std::cout << "Max absolute error: " << maxAbsErr << "\n";
    std::cout << "Max relative error: " << maxRelErr << "\n";
    std::cout << "Worst index: " << worstIndex
              << " analytical=" << worstA
              << " numerical=" << worstN << "\n";

    expectTrue(maxAbsErr <= kAbsTol,
               "Max absolute error exceeds tolerance: " + std::to_string(maxAbsErr) +
               " > " + std::to_string(kAbsTol));
    expectTrue(maxRelErr <= kRelTol,
               "Max relative error exceeds tolerance: " + std::to_string(maxRelErr) +
               " > " + std::to_string(kRelTol));
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"LSTM BPTT finite-difference gradient check", test_lstm_bptt_finite_difference_gradient},
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

    std::cout << "\nLSTM gradient tests: " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
