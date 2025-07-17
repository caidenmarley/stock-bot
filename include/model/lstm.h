#pragma once
#include <Eigen/Dense>
#include <random>
#include <utility>
#include <vector>

class LSTMCell {
  public:
    LSTMCell(int numFeatures, int hiddenSize, int sequenceLength);
    void xavierWeightsInit();
    inline void initWeights(Eigen::MatrixXd& W, std::mt19937& rng, std::normal_distribution<double>& dist) {
        for (int i = 0; i < W.rows(); ++i) {
            for (int j = 0; j < W.cols(); ++j) {
                W(i, j) = dist(rng);
            }
        }
    }

    static auto inline sigmoid = [](double val) { return 1.0 / (1.0 + std::exp(-val)); };
    static auto inline tanhLambda = [](double val) { return std::tanh(val); };

    Eigen::VectorXd forwardPass(const Eigen::VectorXd& input);
    std::pair<Eigen::VectorXd, Eigen::VectorXd> backwardPass(const Eigen::VectorXd& deltaH, const Eigen::VectorXd& deltaC);

    size_t getParameterCount() const;    // get total num of elements in weights and biases

    void reset();   // clear stepdata and clear hidden and cell state

    Eigen::VectorXd getHiddenState() const {return this->hiddenState;}

    Eigen::VectorXd getCellState() const {return this->cellState;}

    Eigen::VectorXd& getParametersVector();

    Eigen::VectorXd& getGradientsVector();

    void setParametersVector(const Eigen::VectorXd& v);

    void zeroGrad();

  private:
    int numFeatures;
    int hiddenSize;

    // Forward vars
    // Forget weight(Wf), hidden forget weight(Uf), forget bias(bf)
    // forget gate output vector(Ft) = sigmoid(Wf*inputVec + Uf*PrevHiddenStateVec + bf)
    Eigen::MatrixXd Wf, Uf;
    Eigen::VectorXd bf;

    // input weight(Wi), hidden input weight(Ui), input bias(bi)
    // input gate output vector(It) = sigmoid(Wi*inputVec + Ui*PrevHiddenStateVec + bi)
    Eigen::MatrixXd Wi, Ui;
    Eigen::VectorXd bi;

    // output weight(Wo), hidden output weight(Uo), output bias(bo)
    // output gate output vector(Ot) = sigmoid(Wo*inputVec + Uo*PrevHiddenStateVec + bo)
    Eigen::MatrixXd Wo, Uo;
    Eigen::VectorXd bo;

    // cell input weight(Wc), hidden cell input weight(Uc), cell input bias(bc)
    // cell input vector(Ct) = tanh(Wc*inputVec + Uc*PrevHiddenStateVec + bc)
    Eigen::MatrixXd Wc, Uc;
    Eigen::VectorXd bc;

    // cell state vector = (Ft Hadamard product previous cell state vector) + (It Hadarmard product Ct)
    // hidden state vector = Ot Hadamard product tanh(cell state vector)
    Eigen::VectorXd cellState, hiddenState;

    // Backward vars
    // Struct for data for each time step
    struct StepData {
        Eigen::VectorXd input;
        Eigen::VectorXd prevHiddenState, prevCellState;
        Eigen::VectorXd f, i, o; // gate outputs
        Eigen::VectorXd cTilde;  // c~ output
        Eigen::VectorXd c;       // new cell state
    };

    std::vector<StepData> stepData;

    // Gradients
    // forget gate
    Eigen::MatrixXd deltaWf, deltaUf;
    Eigen::VectorXd deltaBf;

    // input gate
    Eigen::MatrixXd deltaWi, deltaUi;
    Eigen::VectorXd deltaBi;

    // c~
    Eigen::MatrixXd deltaWc, deltaUc;
    Eigen::VectorXd deltaBc;

    // output gate
    Eigen::MatrixXd deltaWo, deltaUo;
    Eigen::VectorXd deltaBo;

    size_t parameterCount;

    Eigen::VectorXd paramVec;
    Eigen::VectorXd gradientVec;
};