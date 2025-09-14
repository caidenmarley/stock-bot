#include "model/lstm.h"
#include <cmath>
#include <cstring>
#include <fstream>

static std::mt19937 gLSTMRNG{std::random_device{}()};

LSTMCell::LSTMCell(int numFeatures, int hiddenSize, int sequenceLength)
    : numFeatures(numFeatures), hiddenSize(hiddenSize),
      Wf(hiddenSize, numFeatures), Uf(hiddenSize, hiddenSize), bf(Eigen::VectorXd::Constant(hiddenSize, 1.0)), // set to 1 so lstm doesnt forget at the start
      Wi(hiddenSize, numFeatures), Ui(hiddenSize, hiddenSize), bi(Eigen::VectorXd::Zero(hiddenSize)),
      Wo(hiddenSize, numFeatures), Uo(hiddenSize, hiddenSize), bo(Eigen::VectorXd::Zero(hiddenSize)),
      Wc(hiddenSize, numFeatures), Uc(hiddenSize, hiddenSize), bc(Eigen::VectorXd::Zero(hiddenSize)),
      cellState(Eigen::VectorXd::Zero(hiddenSize)), hiddenState(Eigen::VectorXd::Zero(hiddenSize)),
      // zero initialize all gradient mat/vecs
      deltaWf(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      deltaUf(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      deltaBf(Eigen::VectorXd::Zero(hiddenSize)),

      deltaWi(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      deltaUi(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      deltaBi(Eigen::VectorXd::Zero(hiddenSize)),

      deltaWc(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      deltaUc(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      deltaBc(Eigen::VectorXd::Zero(hiddenSize)),

      deltaWo(Eigen::MatrixXd::Zero(hiddenSize, numFeatures)),
      deltaUo(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize)),
      deltaBo(Eigen::VectorXd::Zero(hiddenSize)),
      parameterCount(
        Wf.size() + Wi.size() + Wo.size() + Wc.size() +
        Uf.size() + Ui.size() + Uo.size() + Uc.size() +
        bf.size() + bi.size() + bo.size() + bc.size()
      ), paramVec(Eigen::VectorXd(parameterCount)), gradientVec(Eigen::VectorXd(parameterCount)) {
    xavierWeightsInit();

    stepData.reserve(sequenceLength);
}

void LSTMCell::xavierWeightsInit() {
    // static prevents reinitialsing mt19937 again (expensive time cost)
    //static std::mt19937 rng(std::random_device {}());

    double stdDevInput = std::sqrt(2.0 / (static_cast<double>(numFeatures) + static_cast<double>(hiddenSize)));
    double stdDevHidden = std::sqrt(2.0 / (static_cast<double>(hiddenSize) + static_cast<double>(hiddenSize)));
    std::normal_distribution<double> normDistInput(0.0, stdDevInput);
    std::normal_distribution<double> normDistHidden(0.0, stdDevHidden);
    initWeights(Wf, gLSTMRNG, normDistInput);
    initWeights(Uf, gLSTMRNG, normDistHidden);
    initWeights(Wi, gLSTMRNG, normDistInput);
    initWeights(Ui, gLSTMRNG, normDistHidden);
    initWeights(Wo, gLSTMRNG, normDistInput);
    initWeights(Uo, gLSTMRNG, normDistHidden);
    initWeights(Wc, gLSTMRNG, normDistInput);
    initWeights(Uc, gLSTMRNG, normDistHidden);
}

Eigen::VectorXd LSTMCell::forwardPass(const Eigen::VectorXd& input) {
    Eigen::VectorXd prevHidden = this->hiddenState;
    Eigen::VectorXd prevCell = this->cellState;

    // forget gate
    // forget gate output vector(Ft) = sigmoid(Wf*inputVec + Uf*PrevHiddenStateVec + bf)
    Eigen::VectorXd forgetGateOutput(hiddenSize);
    // uses noalias to prevent unnecessary tempory vectors being allocated
    forgetGateOutput.noalias() = this->Wf * input;
    forgetGateOutput.noalias() += this->Uf * prevHidden;
    forgetGateOutput += this->bf;
    forgetGateOutput = forgetGateOutput.unaryExpr(this->sigmoid);

    // input gate
    // input gate output vector(It) = sigmoid(Wi*inputVec + Ui*PrevHiddenStateVec + bi)
    Eigen::VectorXd inputGateOutput(hiddenSize); 
    inputGateOutput.noalias() = this->Wi * input;
    inputGateOutput.noalias() += this->Ui * prevHidden;
    inputGateOutput += this->bi;
    inputGateOutput = inputGateOutput.unaryExpr(this->sigmoid);

    // cell input (c~)
    // cell input vector(Ct) = tanh(Wc*inputVec + Uc*PrevHiddenStateVec + bc)
    Eigen::VectorXd cellInput(hiddenSize);
    cellInput.noalias() = this->Wc * input;
    cellInput.noalias() += this->Uc * prevHidden;
    cellInput += this->bc;
    cellInput = cellInput.unaryExpr(this->tanhLambda);

    // output gate
    // output gate output vector(Ot) = sigmoid(Wo*inputVec + Uo*PrevHiddenStateVec + bo)
    Eigen::VectorXd outputGateOutput(hiddenSize);
    outputGateOutput.noalias() = this->Wo * input;
    outputGateOutput.noalias() += this->Uo * prevHidden;
    outputGateOutput += this->bo;
    outputGateOutput = outputGateOutput.unaryExpr(this->sigmoid);

    // cell state vector = (Ft Hadamard product previous cell state vector) + (It Hadarmard product Ct)
    // use eigen array api to conv to array and do element wise multiplication (better optimisation for complier)
    this->cellState = (forgetGateOutput.array() * prevCell.array() + inputGateOutput.array() * cellInput.array()).matrix();

    // hidden state vector = Ot Hadamard product tanh(cell state vector)
    this->hiddenState = (outputGateOutput.array() * this->cellState.array().tanh()).matrix();

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

std::pair<Eigen::VectorXd, Eigen::VectorXd> LSTMCell::backwardPass(const Eigen::VectorXd& deltaH, const Eigen::VectorXd& deltaC) {
    // NOTE deltaX = dL/dx - how a change in X affects the Loss

    const StepData& stepDataS = this->stepData.back();

    // 1. last forward step was h_t = o_t cwiseProd tanh(c_t)

    // recalc tanh(c_t)
    Eigen::VectorXd tanhC = stepDataS.c.array().tanh().matrix();

    // dh_t/do_t = tanh(c_t) [chain rule from h_t = o_t * tanh(c_t)]
    // dL/do = dL/dh * dh/do = dh x tanh(c)
    Eigen::VectorXd deltaO = (deltaH.array() * tanhC.array()).matrix();

    // because h_t uses c_t, during bptt any loss in h_t ripples back into c_t
    // so dh_t/dc_t is needed from h_t = o_t cwiseProd tanh(c_t)
    // dh_t/dc_t = o_t * (1 - tanh^2(c_t))
    // dL/dc (through h) = dL/dh * dh_t/dc_t
    Eigen::VectorXd deltaCThroughH = (deltaH.array() * stepDataS.o.array() * (1.0 - tanhC.array().square())).matrix();

    // gradients into c come from both nexts time steps deltaC and through hidden state
    Eigen::VectorXd deltaCTotal = deltaC + deltaCThroughH;

    // 2. next step backwards was c_t = (f_t cwiseP c_t-1) + (i_t cwiseP c~_t)

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

    // 3. next step backwards was c~_t = tanh(c~PreTanh)

    // dc~_t/dc~PreFunc = (sech(c~PreFunc))^2 = 1 - (tanh(c~PreFunc))^2
    // dc~_t/dc~PreFunc = 1 - (c~_t)^2
    // dL/dc~PreFunc = dL/dc~ * c~_t/dc~PreFunc
    Eigen::VectorXd deltaCTildePreFunc = (deltaCTilde.array() * (1.0 - stepDataS.cTilde.array().square())).matrix();

    // 4. next step backwards o_t = sigmoid(oPreSigmoid) 1/1+e^(-oPreSigmoid)

    // d/dx sigmoid = e^-x/(1+e^-x)^2
    // do_t/doPreSigmoid = o_t - o_t^2 == o_t(1-o_t)
    // dL/doPreSigmoid = dL/do * do_t/do~PreSigmoid
    Eigen::VectorXd deltaOPreFunc = (deltaO.array() * stepDataS.o.array() * (1.0 - stepDataS.o.array())).matrix();

    // 5. next step backwards i_t = sigmoid(iPreSigmoid)

    // di_t/diPreSigmoid = i_t(1-i_t)
    // dL/doPreSigmoid = dL/di * di_t/di~PreSigmoid
    Eigen::VectorXd deltaIPreFunc = (deltaI.array() * stepDataS.i.array() * (1.0 - stepDataS.i.array())).matrix();

    // 6. next step backwards f_t = sigmoid(fPreSigmoid)

    // df_t/dfPreSigmoid = f_t(1-f_t)
    // dL/doPreSigmoid = dL/df * df_t/df~PreSigmoid
    Eigen::VectorXd deltaFPreFunc = (deltaF.array() * stepDataS.f.array() * (1.0 - stepDataS.f.array())).matrix();

    // TODO add notes as notes for project in some dir
    // add gradients to total gradient for weights in the sequence
    // no aliasing means no matrix is on both left and right side
    // dL/dW(f/i/o/c~) = dL/d(f/i/o/c~)PreFunc * inputs^T
    this->deltaWf.noalias() += deltaFPreFunc * stepDataS.input.transpose();
    this->deltaWi.noalias() += deltaIPreFunc * stepDataS.input.transpose();
    this->deltaWo.noalias() += deltaOPreFunc * stepDataS.input.transpose();
    this->deltaWc.noalias() += deltaCTildePreFunc * stepDataS.input.transpose();

    // dL/dU(f/i/o/c~) = dL/d(f/i/o/c~)PreFunc * prevHiddenState^T
    this->deltaUf.noalias() += deltaFPreFunc * stepDataS.prevHiddenState.transpose();
    this->deltaUi.noalias() += deltaIPreFunc * stepDataS.prevHiddenState.transpose();
    this->deltaUo.noalias() += deltaOPreFunc * stepDataS.prevHiddenState.transpose();
    this->deltaUc.noalias() += deltaCTildePreFunc * stepDataS.prevHiddenState.transpose();

    // dL/dB(f/i/o/c~) = dL/d(f/i/o/c~)PreFunc
    this->deltaBf.noalias() += deltaFPreFunc;
    this->deltaBi.noalias() += deltaIPreFunc;
    this->deltaBo.noalias() += deltaOPreFunc;
    this->deltaBc.noalias() += deltaCTildePreFunc;

    // deltaHPrev for return value
    // dL/d_hPrev = sum (dL/d(f/i/o/c~)PreFunc x d(f/i/o/c))/d_hPrev
    // d(f/i/o/c))/d_hPrev = U(f/i/o/c) so need to tranpose for same reason as before
    Eigen::VectorXd deltaHPrev(hiddenSize);
    deltaHPrev.noalias() = this->Uf.transpose() * deltaFPreFunc;
    deltaHPrev.noalias() += this->Ui.transpose() * deltaIPreFunc;
    deltaHPrev.noalias() += this->Uo.transpose() * deltaOPreFunc;
    deltaHPrev.noalias() += this->Uc.transpose() * deltaCTildePreFunc;

    this->stepData.pop_back();
    return {
        deltaHPrev, deltaCPrev
    };
}

size_t LSTMCell::getParameterCount() const{
    return this->parameterCount;
}

void LSTMCell::reset(){
    this->hiddenState.setZero();
    this->cellState.setZero();
    this->stepData.clear();
}

Eigen::VectorXd& LSTMCell::getParametersVector() {
    double* destination = paramVec.data();

    // cpy Wf
    std::memcpy(destination, Wf.data(), sizeof(double) * static_cast<size_t>(Wf.size()));
    destination += Wf.size();

    // cpy Uf
    std::memcpy(destination, Uf.data(), sizeof(double) * static_cast<size_t>(Uf.size()));
    destination += Uf.size();   

    // cpy bf
    std::memcpy(destination, bf.data(), sizeof(double) * static_cast<size_t>(bf.size()));
    destination += bf.size();

    // cpy Wi
    std::memcpy(destination, Wi.data(), sizeof(double) * static_cast<size_t>(Wi.size()));
    destination += Wi.size();

    // cpy Ui
    std::memcpy(destination, Ui.data(), sizeof(double) * static_cast<size_t>(Ui.size()));
    destination += Ui.size();

    // cpy bi
    std::memcpy(destination, bi.data(), sizeof(double) * static_cast<size_t>(bi.size()));
    destination += bi.size();

    // cpy Wc
    std::memcpy(destination, Wc.data(), sizeof(double) * static_cast<size_t>(Wc.size()));
    destination += Wc.size();

    // cpy Uc
    std::memcpy(destination, Uc.data(), sizeof(double) * static_cast<size_t>(Uc.size()));
    destination += Uc.size();

    // cpy bc
    std::memcpy(destination, bc.data(), sizeof(double) * static_cast<size_t>(bc.size()));
    destination += bc.size();

    // cpy Wo
    std::memcpy(destination, Wo.data(), sizeof(double) * static_cast<size_t>(Wo.size()));
    destination += Wo.size();

    // cpy Uo
    std::memcpy(destination, Uo.data(), sizeof(double) * static_cast<size_t>(Uo.size()));
    destination += Uo.size();

    // cpy bo
    std::memcpy(destination, bo.data(), sizeof(double) * static_cast<size_t>(bo.size()));
    destination += bo.size();

    return paramVec;
}

Eigen::VectorXd& LSTMCell::getGradientsVector(){
    double* destination = gradientVec.data();

    // cpy dWf
    std::memcpy(destination, deltaWf.data(), sizeof(double) * static_cast<size_t>(deltaWf.size()));
    destination += deltaWf.size();

    // cpy dUf
    std::memcpy(destination, deltaUf.data(), sizeof(double) * static_cast<size_t>(deltaUf.size()));
    destination += deltaUf.size();   

    // cpy dbf
    std::memcpy(destination, deltaBf.data(), sizeof(double) * static_cast<size_t>(deltaBf.size()));
    destination += deltaBf.size();

    // cpy dWi
    std::memcpy(destination, deltaWi.data(), sizeof(double) * static_cast<size_t>(deltaWi.size()));
    destination += deltaWi.size();

    // cpy dUi
    std::memcpy(destination, deltaUi.data(), sizeof(double) * static_cast<size_t>(deltaUi.size()));
    destination += deltaUi.size();

    // cpy dbi
    std::memcpy(destination, deltaBi.data(), sizeof(double) * static_cast<size_t>(deltaBi.size()));
    destination += deltaBi.size();

    // cpy dWc
    std::memcpy(destination, deltaWc.data(), sizeof(double) * static_cast<size_t>(deltaWc.size()));
    destination += deltaWc.size();

    // cpy dUc
    std::memcpy(destination, deltaUc.data(), sizeof(double) * static_cast<size_t>(deltaUc.size()));
    destination += deltaUc.size();

    // cpy dbc
    std::memcpy(destination, deltaBc.data(), sizeof(double) * static_cast<size_t>(deltaBc.size()));
    destination += deltaBc.size();

    // cpy dWo
    std::memcpy(destination, deltaWo.data(), sizeof(double) * static_cast<size_t>(deltaWo.size()));
    destination += deltaWo.size();

    // cpy dUo
    std::memcpy(destination, deltaUo.data(), sizeof(double) * static_cast<size_t>(deltaUo.size()));
    destination += deltaUo.size();

    // cpy dbo
    std::memcpy(destination, deltaBo.data(), sizeof(double) * static_cast<size_t>(deltaBo.size()));
    destination += deltaBo.size();

    return gradientVec;
}

void LSTMCell::setParametersVector(const Eigen::VectorXd& v) {
    size_t offset = 0;
    auto setMat = [&](auto& M) {
        const size_t sz = static_cast<size_t>(M.size());
        std::memcpy(M.data(), v.data() + offset, sizeof(double)*sz);
        offset += sz;
    };
    // Weights and biases (same order as getParametersVector)
    setMat(Wf); setMat(Uf); setMat(bf);
    setMat(Wi); setMat(Ui); setMat(bi);
    setMat(Wc); setMat(Uc); setMat(bc);
    setMat(Wo); setMat(Uo); setMat(bo);
}

void LSTMCell::zeroGrad() {
    deltaWf.setZero(); deltaUf.setZero(); deltaBf.setZero();
    deltaWi.setZero(); deltaUi.setZero(); deltaBi.setZero();
    deltaWc.setZero(); deltaUc.setZero(); deltaBc.setZero();
    deltaWo.setZero(); deltaUo.setZero(); deltaBo.setZero();
}

void LSTMCell::saveParameters(const std::string& path) const{
    // strip const so you can call non const method getParametersVector()
    const Eigen::VectorXd& paramVector = const_cast<LSTMCell*>(this)->getParametersVector();

    std::ofstream out{path, std::ios::binary};
    if(!out){
        throw std::runtime_error("Failed to open binary file");
    }

    // first writes how many parameters there are
    out.write(reinterpret_cast<const char*>(&parameterCount), sizeof(parameterCount));

    // then write actual parameter data
    out.write(reinterpret_cast<const char*>(paramVector.data()), sizeof(double) * parameterCount);

    if(!out){
        throw std::runtime_error("Error writing binary file");
    }
}

void LSTMCell::loadParameters(const std::string& path){
    std::ifstream in{path, std::ios::binary};
    if(!in){
        throw std::runtime_error("Failed to open binary file");
    }

    size_t numParamsInBin;

    in.read(reinterpret_cast<char*>(&numParamsInBin), sizeof(numParamsInBin));

    if(!in){
        throw std::runtime_error("Error reading parameter count");
    }

    if(numParamsInBin != this->parameterCount){
        throw std::runtime_error("parameter count doesnt match, paramsInBin: " + std::to_string(numParamsInBin) + "paramsInClass: " + std::to_string(this->parameterCount));
    }

    Eigen::VectorXd loadedParams(numParamsInBin);

    in.read(reinterpret_cast<char*>(loadedParams.data()), sizeof(double) * static_cast<std::size_t>(numParamsInBin));

    if(!in){
        throw std::runtime_error("Error reading parameter data from bin");
    }

    this->setParametersVector(loadedParams);
}

void LSTMCell::setGlobalInitSeed(uint32_t seed){
    gLSTMRNG.seed(seed);
}