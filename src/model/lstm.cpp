#include "model/lstm.h"
#include <cmath>

LSTMCell::LSTMCell(int numFeatures, int hiddenSize)
    : numFeatures(numFeatures), hiddenSize(hiddenSize),
      Wf(hiddenSize, numFeatures), Uf(hiddenSize, hiddenSize), bf(Eigen::VectorXd::Constant(hiddenSize, 1.0)), // set to 1 so lstm doesnt forget at the start
      Wi(hiddenSize, numFeatures), Ui(hiddenSize, hiddenSize), bi(Eigen::VectorXd::Zero(hiddenSize)),
      Wo(hiddenSize, numFeatures), Uo(hiddenSize, hiddenSize), bo(Eigen::VectorXd::Zero(hiddenSize)),
      Wc(hiddenSize, numFeatures), Uc(hiddenSize, hiddenSize), bc(Eigen::VectorXd::Zero(hiddenSize)),
      cellState(Eigen::VectorXd::Zero(hiddenSize)), hiddenState(Eigen::VectorXd::Zero(hiddenSize)),
	        // zero initialize all gradient mat/vecs
      dWf(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      dUf(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      dbf(Eigen::VectorXd::Zero(hiddenSize)),

      dWi(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      dUi(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      dbi(Eigen::VectorXd::Zero(hiddenSize)),

      dWc(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      dUc(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      dbc(Eigen::VectorXd::Zero(hiddenSize)),

      dWo(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      dUo(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      dbo(Eigen::VectorXd::Zero(hiddenSize)) {
    xavierWeightsInit();

	//stepData.reserve(SOMETHING) TODO
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
	Eigen::VectorXd prevHidden = this->hiddenState;
	Eigen::VectorXd prevCell = this->cellState;

    // forget gate
	// forget gate output vector(Ft) = sigmoid(Wf*inputVec + Uf*PrevHiddenStateVec + bf)
	Eigen::VectorXd forgetGateOutput = this->Wf*input + this->Uf*prevHidden + this->bf;
	forgetGateOutput = forgetGateOutput.unaryExpr(this->sigmoid);

	// input gate
	// input gate output vector(It) = sigmoid(Wi*inputVec + Ui*PrevHiddenStateVec + bi)
	Eigen::VectorXd inputGateOutput = this->Wi*input + this->Ui*prevHidden + this->bi;
	inputGateOutput = inputGateOutput.unaryExpr(this->sigmoid);

	// cell input (c~)
	// cell input vector(Ct) = tanh(Wc*inputVec + Uc*PrevHiddenStateVec + bc)
	Eigen::VectorXd cellInput = this->Wc*input + this->Uc*prevHidden + this->bc;
	cellInput = cellInput.unaryExpr(this->tanhLambda);

	// output gate
	// output gate output vector(Ot) = sigmoid(Wo*inputVec + Uo*PrevHiddenStateVec + bo)
	Eigen::VectorXd outputGateOutput = this->Wo*input + this->Uo*prevHidden + this->bo;
	outputGateOutput = outputGateOutput.unaryExpr(this->sigmoid);

	// cell state vector = (Ft Hadamard product previous cell state vector) + (It Hadarmard product Ct)
	// use eigen array api to conv to array and do element wise multiplication (better optimisation for complier)
	this->cellState = (forgetGateOutput.array()*prevCell.array() + inputGateOutput.array()*cellInput.array()).matrix();

	// hidden state vector = Ot Hadamard product tanh(cell state vector)
	this->hiddenState = (outputGateOutput.array()*this->cellState.array().tanh()).matrix();

	// Data for bptt
	StepData stepDataS;
	stepDataS.input = input;
	stepDataS.prevHiddenState = prevHidden;
	stepDataS.prevCellState = prevCell;
	stepDataS.f = forgetGateOutput;
	stepDataS.i = inputGateOutput;
	stepDataS.o = outputGateOutput;
	stepDataS.cTilde = cellInput;
	stepDataS.c = this->cellState;
	this->stepData.push_back(stepDataS);

	return this->hiddenState;
}

// delta_h = dL/dh_t
// delta_c = dL/dc_t
// deltaX = dL/dx - how a change in X affects the Loss
// returns pair dh_t-1 and dc_t-1
std::pair<Eigen::VectorXd,Eigen::VectorXd> LSTMCell::backwardPass(const Eigen::VectorXd& deltaH, const Eigen::VectorXd& deltaC){
	const StepData& stepDataS = this->stepData.back();

	// recompute tanh(c_t) as last forward step was h_t = o_t cwiseProd tanh(c_t)
	Eigen::VectorXd tanhC = stepDataS.c.array().tanh().matrix();

	// dh_t/do_t = tanh(c_t) [chain rule from h_t = o_t * tanh(c_t)]
	// dL/do = dL/dh * dh/do = dh x tanh(c)
	Eigen::VectorXd deltaO = deltaH.array() * tanhC.array();

	// because h_t uses c_t, during bptt any loss in h_t ripples back into c_t
	// so dh_t/dc_t is needed from h_t = o_t cwiseProd tanh(c_t)
	// dh_t/dc_t = o_t * (1 - tanh^2(c_t))
	// dL/dc (through h) = dL/dh * dh_t/dc_t
	Eigen::VectorXd deltaCThroughH = (deltaH.array() * stepDataS.o.array() * (1.0 - tanhC.array().square())).matrix();

	// gradients into c come from both nexts time steps deltaC and through hidden state
	Eigen::VectorXd deltaCTotal = deltaC + deltaCThroughH;

	// next step backwards was c_t = (f_t cwiseP c_t-1) + (i_t cwiseP c~_t)
	// dc_t/df_t = c_t-1
	// dL/df = dc_t * c_t-1
	Eigen::VectorXd deltaF = (deltaCTotal.array() * stepDataS.prevCellState.array()).matrix();

	// dc_t/di_t = c~_t
	// dL/di = dc_t * c~_t
	Eigen::VectorXd deltaI = (deltaCTotal.array() * stepDataS.cTilde.array()).matrix();

	// dc_t/dc~_t = i_t
	// dL/dc~ = dc_t * i_t
	Eigen::VectorXd deltaCTilde = (deltaCTotal.array() * stepDataS.i.array()).matrix();

	// dc_t/dc_t-1 = f_t
	// dL/dc_t-1 = dc_t * f_t
	Eigen::VectorXd deltaCPrev = (deltaCTotal.array() * stepDataS.f.array()).matrix();
}