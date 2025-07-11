#pragma once
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <utility>

class LSTMCell {
  public:
    LSTMCell(int numFeatures, int hiddenSize);
    void xavierWeightsInit();
    inline void initWeights(Eigen::MatrixXd& W, std::mt19937& rng, std::normal_distribution<double>& dist) {
        for (int i = 0; i < W.rows(); ++i) {
            for (int j = 0; j < W.cols(); ++j) {
                W(i, j) = dist(rng);
            }
        }
    }

    static auto inline sigmoid = [](double val){return 1.0/(1.0 + std::exp(-val));};
    static auto inline tanhLambda = [](double val){return std::tanh(val);};

    Eigen::VectorXd forwardPass(const Eigen::VectorXd& input);
    std::pair<Eigen::VectorXd,Eigen::VectorXd> backwardPass(const Eigen::VectorXd& deltaH, const Eigen::VectorXd& deltaC);

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
    struct StepData{
      Eigen::VectorXd input;
      Eigen::VectorXd prevHiddenState, prevCellState;
      Eigen::VectorXd f, i, o; // gate outputs
      Eigen::VectorXd cTilde; // c~ output
      Eigen::VectorXd c; // new cell state
    };

    std::vector<StepData> stepData;

    // Gradients
    // forget gate
    Eigen::MatrixXd dWf, dUf;
    Eigen::VectorXd dbf;

    // input gate
    Eigen::MatrixXd dWi, dUi;
    Eigen::VectorXd dbi;

    // c~
    Eigen::MatrixXd dWc, dUc;
    Eigen::VectorXd dbc;

    // output gate
    Eigen::MatrixXd dWo, dUo;
    Eigen::VectorXd dbo;

};