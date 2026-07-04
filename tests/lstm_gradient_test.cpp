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

constexpr double kEpsilon = 1e-5;
constexpr double kAbsTol = 1e-4;
constexpr double kRelTol = 1e-3;

struct GradientCheckConfig {
    std::string name;
    int inputSize;
    int hiddenSize;
    int sequenceLength;
    bool includeCellLoss;
    double cellLossWeight;
};

struct GradientCheckResult {
    double maxAbsErr;
    double maxRelErr;
    int worstIndex;
    double worstAnalytical;
    double worstNumerical;
};

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

std::vector<Eigen::VectorXd> makeInputSequenceGeneral(int inputSize, int sequenceLength, double scale, double phase) {
    std::vector<Eigen::VectorXd> seq;
    seq.reserve(static_cast<std::size_t>(sequenceLength));

    for (int t = 0; t < sequenceLength; ++t) {
        Eigen::VectorXd x(inputSize);
        for (int j = 0; j < inputSize; ++j) {
            const double indexTerm = static_cast<double>((t + 1) * (j + 1));
            x(j) = scale * std::sin(0.37 * indexTerm + phase) + 0.03 * std::cos(0.19 * indexTerm + phase * 0.5);
        }
        seq.push_back(x);
    }

    return seq;
}

std::vector<Eigen::VectorXd> makeTinyCaseSequence() {
    std::vector<Eigen::VectorXd> seq;
    seq.reserve(3);

    Eigen::VectorXd x0(2), x1(2), x2(2);
    x0 << 0.10, -0.20;
    x1 << 0.05, 0.30;
    x2 << -0.15, 0.07;

    seq.push_back(x0);
    seq.push_back(x1);
    seq.push_back(x2);
    return seq;
}

Eigen::VectorXd makeTarget(int hiddenSize, double scale, double phase) {
    Eigen::VectorXd target(hiddenSize);
    for (int j = 0; j < hiddenSize; ++j) {
        target(j) = scale * std::cos(0.23 * static_cast<double>(j + 1) + phase);
    }
    return target;
}

Eigen::VectorXd makeTinyCaseTarget() {
    Eigen::VectorXd target(4);
    for (int j = 0; j < 4; ++j) {
        target(j) = 0.03 * static_cast<double>(j + 1);
    }
    return target;
}

Eigen::VectorXd makeDeterministicParamsSine(int paramCount, double scale, double phase) {
    Eigen::VectorXd params(paramCount);
    for (int i = 0; i < paramCount; ++i) {
        params(i) = scale * std::sin(0.31 * static_cast<double>(i + 1) + phase);
    }
    return params;
}

Eigen::VectorXd makeDeterministicParamsCosine(int paramCount, double scale, double phase) {
    Eigen::VectorXd params(paramCount);
    for (int i = 0; i < paramCount; ++i) {
        params(i) = scale * std::cos(0.29 * static_cast<double>(i + 1) + phase);
    }
    return params;
}

double computeLoss(LSTMCell& lstm,
                   const Eigen::VectorXd& params,
                   const std::vector<Eigen::VectorXd>& sequence,
                   const Eigen::VectorXd& target,
                   bool includeCellLoss,
                   double cellLossWeight,
                   const Eigen::VectorXd& cellTarget) {
    lstm.setParametersVector(params);
    lstm.reset();

    for (const auto& x : sequence) {
        (void)lstm.forwardPass(x);
    }

    const Eigen::VectorXd hFinal = lstm.getHiddenState();
    const Eigen::VectorXd diff = hFinal - target;
    double loss = 0.5 * diff.squaredNorm();

    if (includeCellLoss) {
        const Eigen::VectorXd cFinal = lstm.getCellState();
        const Eigen::VectorXd cDiff = cFinal - cellTarget;
        loss += 0.5 * cellLossWeight * cDiff.squaredNorm();
    }

    return loss;
}

Eigen::VectorXd computeAnalyticalGradient(LSTMCell& lstm,
                                          const Eigen::VectorXd& params,
                                          const std::vector<Eigen::VectorXd>& sequence,
                                          const Eigen::VectorXd& target,
                                          bool includeCellLoss,
                                          double cellLossWeight,
                                          const Eigen::VectorXd& cellTarget) {
    lstm.setParametersVector(params);
    lstm.reset();
    lstm.zeroGrad();

    for (const auto& x : sequence) {
        (void)lstm.forwardPass(x);
    }

    Eigen::VectorXd deltaH = lstm.getHiddenState() - target; // dL/dh_final
    Eigen::VectorXd deltaC = Eigen::VectorXd::Zero(target.size()); // dL/dc_final
    if (includeCellLoss) {
        deltaC = cellLossWeight * (lstm.getCellState() - cellTarget);
    }

    for (int t = static_cast<int>(sequence.size()) - 1; t >= 0; --t) {
        (void)t;
        const auto back = lstm.backwardPass(deltaH, deltaC);
        deltaH = back.first;
        deltaC = back.second;
    }

    return lstm.getGradientsVector();
}

GradientCheckResult runGradientCheck(const GradientCheckConfig& cfg,
                                     const std::vector<Eigen::VectorXd>& sequence,
                                     const Eigen::VectorXd& target,
                                     const Eigen::VectorXd& params,
                                     const Eigen::VectorXd& cellTarget) {
    LSTMCell lstm(cfg.inputSize, cfg.hiddenSize, cfg.sequenceLength);

    const int paramCount = lstm.getParametersVector().size();
    expectTrue(target.size() == cfg.hiddenSize, "Target size must match hidden size");
    expectTrue(params.size() == paramCount, "Parameter vector size must match LSTM parameter count");
    expectTrue(static_cast<int>(sequence.size()) == cfg.sequenceLength, "Sequence length must match LSTM sequence length");
    for (const auto& x : sequence) {
        expectTrue(x.size() == cfg.inputSize, "Input vector size must match configured input size");
    }
    if (cfg.includeCellLoss) {
        expectTrue(cellTarget.size() == cfg.hiddenSize, "Cell target size must match hidden size when cell loss is enabled");
    }

    const Eigen::VectorXd analytical = computeAnalyticalGradient(
        lstm,
        params,
        sequence,
        target,
        cfg.includeCellLoss,
        cfg.cellLossWeight,
        cellTarget);

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

        const double lossPlus = computeLoss(
            lstm,
            plus,
            sequence,
            target,
            cfg.includeCellLoss,
            cfg.cellLossWeight,
            cellTarget);
        const double lossMinus = computeLoss(
            lstm,
            minus,
            sequence,
            target,
            cfg.includeCellLoss,
            cfg.cellLossWeight,
            cellTarget);
        numerical(i) = (lossPlus - lossMinus) / (2.0 * kEpsilon);
    }

    expectTrue(numerical.size() == paramCount, "Numerical gradient size must match parameter vector size");
    for (int i = 0; i < numerical.size(); ++i) {
        expectTrue(std::isfinite(numerical(i)), "Numerical gradient has NaN/Inf at index " + std::to_string(i));
    }

    GradientCheckResult result{0.0, 0.0, -1, 0.0, 0.0};

    for (int i = 0; i < paramCount; ++i) {
        const double a = analytical(i);
        const double n = numerical(i);
        const double absErr = std::abs(a - n);
        const double denom = std::max({1.0, std::abs(a), std::abs(n)});
        const double relErr = absErr / denom;

        if (absErr > result.maxAbsErr) {
            result.maxAbsErr = absErr;
        }
        if (relErr > result.maxRelErr) {
            result.maxRelErr = relErr;
            result.worstIndex = i;
            result.worstAnalytical = a;
            result.worstNumerical = n;
        }
    }

    std::cout << "[Case] " << cfg.name << "\n";
    std::cout << "  Max absolute error: " << result.maxAbsErr << "\n";
    std::cout << "  Max relative error: " << result.maxRelErr << "\n";
    std::cout << "  Worst index: " << result.worstIndex
              << " analytical=" << result.worstAnalytical
              << " numerical=" << result.worstNumerical << "\n";

    expectTrue(result.maxAbsErr <= kAbsTol,
               "Case '" + cfg.name + "' max absolute error exceeds tolerance: " + std::to_string(result.maxAbsErr) +
               " > " + std::to_string(kAbsTol));
    expectTrue(result.maxRelErr <= kRelTol,
               "Case '" + cfg.name + "' max relative error exceeds tolerance: " + std::to_string(result.maxRelErr) +
               " > " + std::to_string(kRelTol));

    return result;
}

void test_lstm_bptt_finite_difference_gradient_tiny_case() {
    const GradientCheckConfig cfg{
        "tiny_sequence3_input2_hidden4_hloss_only",
        2,
        4,
        3,
        false,
        0.0,
    };

    const auto sequence = makeTinyCaseSequence();
    const Eigen::VectorXd target = makeTinyCaseTarget();

    LSTMCell tmp(cfg.inputSize, cfg.hiddenSize, cfg.sequenceLength);
    const int paramCount = tmp.getParametersVector().size();
    Eigen::VectorXd params(paramCount);
    for (int i = 0; i < paramCount; ++i) {
        params(i) = 0.01 * std::sin(static_cast<double>(i + 1));
    }
    const Eigen::VectorXd cellTarget = Eigen::VectorXd::Zero(cfg.hiddenSize);

    (void)runGradientCheck(cfg, sequence, target, params, cellTarget);
}

void test_lstm_bptt_finite_difference_gradient_sequence1_case() {
    const GradientCheckConfig cfg{
        "sequence1_input3_hidden2_hloss_only",
        3,
        2,
        1,
        false,
        0.0,
    };

    const auto sequence = makeInputSequenceGeneral(cfg.inputSize, cfg.sequenceLength, 0.11, 0.20);
    const Eigen::VectorXd target = makeTarget(cfg.hiddenSize, 0.035, 0.10);

    LSTMCell tmp(cfg.inputSize, cfg.hiddenSize, cfg.sequenceLength);
    const int paramCount = tmp.getParametersVector().size();
    const Eigen::VectorXd params = makeDeterministicParamsSine(paramCount, 0.012, 0.35);
    const Eigen::VectorXd cellTarget = Eigen::VectorXd::Zero(cfg.hiddenSize);

    (void)runGradientCheck(cfg, sequence, target, params, cellTarget);
}

void test_lstm_bptt_finite_difference_gradient_sequence5_with_cell_loss_case() {
    const GradientCheckConfig cfg{
        "sequence5_input1_hidden3_hloss_plus_cellloss",
        1,
        3,
        5,
        true,
        0.25,
    };

    const auto sequence = makeInputSequenceGeneral(cfg.inputSize, cfg.sequenceLength, 0.09, 0.55);
    const Eigen::VectorXd target = makeTarget(cfg.hiddenSize, 0.028, 0.65);
    const Eigen::VectorXd cellTarget = makeTarget(cfg.hiddenSize, 0.017, 0.40);

    LSTMCell tmp(cfg.inputSize, cfg.hiddenSize, cfg.sequenceLength);
    const int paramCount = tmp.getParametersVector().size();
    const Eigen::VectorXd params = makeDeterministicParamsCosine(paramCount, 0.010, 0.25);

    (void)runGradientCheck(cfg, sequence, target, params, cellTarget);
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"LSTM BPTT finite-difference gradient check (tiny base case)", test_lstm_bptt_finite_difference_gradient_tiny_case},
        {"LSTM BPTT finite-difference gradient check (sequence length 1)", test_lstm_bptt_finite_difference_gradient_sequence1_case},
        {"LSTM BPTT finite-difference gradient check (sequence length 5 + cell loss)", test_lstm_bptt_finite_difference_gradient_sequence5_with_cell_loss_case},
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
