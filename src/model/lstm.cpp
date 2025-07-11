#include "model/lstm.h"
#include <cmath>

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

    double stdDevInput = std::sqrt(2.0 / (static_cast<double>(numFeatures) + static_cast<double>(hiddenSize)));
    double stdDevHidden = std::sqrt(2.0 / (static_cast<double>(hiddenSize) + static_cast<double>(hiddenSize)));
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

Eigen::VectorXd LSTMCell::forwardPass(const Eigen::VectorXd& input) {
    // forget gate
	// forget gate output vector(Ft) = sigmoid(Wf*inputVec + Uf*PrevHiddenStateVec + bf)
	Eigen::VectorXd forgetGateOutput = this->Wf*input + this->Uf*this->hiddenState + this->bf;
	forgetGateOutput = forgetGateOutput.unaryExpr(this->sigmoid);

	// input gate
	// input gate output vector(It) = sigmoid(Wi*inputVec + Ui*PrevHiddenStateVec + bi)
	Eigen::VectorXd inputGateOutput = this->Wi*input + this->Ui*this->hiddenState + this->bi;
	inputGateOutput = inputGateOutput.unaryExpr(this->sigmoid);

	// cell input
	// cell input vector(Ct) = tanh(Wc*inputVec + Uc*PrevHiddenStateVec + bc)
	Eigen::VectorXd cellInput = this->Wc*input + this->Uc*this->hiddenState + this->bc;
	cellInput = cellInput.unaryExpr(this->tanhLambda);

	// output gate
	// output gate output vector(Ot) = sigmoid(Wo*inputVec + Uo*PrevHiddenStateVec + bo)
	Eigen::VectorXd outputGateOutput = this->Wo*input + this->Uo*this->hiddenState + this->bo;
	outputGateOutput = outputGateOutput.unaryExpr(this->sigmoid);

	// cell state vector = (Ft Hadamard product previous cell state vector) + (It Hadarmard product Ct)
	// use eigen array api to conv to array and do element wise multiplication (better optimisation for complier)
	this->cellState = (forgetGateOutput.array()*this->cellState.array() + inputGateOutput.array()*cellInput.array()).matrix();

	// hidden state vector = Ot Hadamard product tanh(cell state vector)
	this->hiddenState = (outputGateOutput.array()*this->cellState.array().tanh()).matrix();

	return this->hiddenState;
}

void LSTMCell::backwardPass(){
	
}