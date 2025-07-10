#include "model/lstm.h"
#include <random>

LSTMCell::LSTMCell(int numFeatures, int hiddenSize)
    : numFeatures(numFeatures), hiddenSize(hiddenSize),
      Wf(hiddenSize, numFeatures), Uf(hiddenSize, hiddenSize), bf(Eigen::VectorXd::Constant(hiddenSize, 1.0)), // set to 1 so lstm doesnt forget at the start
      Wi(hiddenSize, numFeatures), Ui(hiddenSize, hiddenSize), bi(Eigen::VectorXd::Zero(hiddenSize)),
      Wo(hiddenSize, numFeatures), Uo(hiddenSize, hiddenSize), bo(Eigen::VectorXd::Zero(hiddenSize)),
      Wc(hiddenSize, numFeatures), Uc(hiddenSize, hiddenSize), bc(Eigen::VectorXd::Zero(hiddenSize)),
      cellState(Eigen::VectorXd::Zero(hiddenSize)), hiddenState(Eigen::VectorXd::Zero(hiddenSize)) {
    xavierWeightsInit();
}

// For each weight draw a rngom value from a normal distribution with mean 0 and
// standard deviation = sqrt(2/(numInputs + numOutputs))
void LSTMCell::xavierWeightsInit() {
    // static prevents reinitialsing mt19937 again (expensive time cost)
    static std::mt19937 rng(std::random_device {}());

    double stdDevInput = sqrt(2.0 / (static_cast<double>(numFeatures) + static_cast<double>(hiddenSize)));
    double stdDevHidden = sqrt(2.0 / (static_cast<double>(hiddenSize) + static_cast<double>(hiddenSize)));
    std::normal_distribution<double> normDistInput(0.0, stdDevInput);
    std::normal_distribution<double> normDistHidden(0.0, stdDevHidden);
    initWeights(Wf, rng, normDistInput);
    initWeights(Uf, rng, normDistHidden);
    initWeights(Wi, rng, normDistInput);
    initWeights(Ui, rng, normDistHidden);
    initWeights(Wo, rng, normDistInput);
    initWeights(Uo, rng, normDistHidden);
    initWeights(Wc, rng, normDistInput);
    initWeights(Uc, rng, normDistHidden);
}