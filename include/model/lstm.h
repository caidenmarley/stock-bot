#pragma once
#include <Eigen/Dense>
#include <random>
#include <utility>
#include <vector>

class LSTMCell {
  public:
    /**
     * Initialises all matrices to their correct sizes, init forget get bias to 1 and all other bias to 0,
     * init all gradient matrices + biases to 0 and hidden and cell states to 0, calculates parameter count 
     * and init param and grad vecs to size of this count, then runs the xavierWeightsInit function to init the weights
     * 
     * @param numFeatures number of features in the input
     * @param hiddenSize size of the hidden layer
     * @param sequenceLength number of days in each sequence
     */
    LSTMCell(int numFeatures, int hiddenSize, int sequenceLength);
    /**
     * Uses a rng and normal distribution defined by mean = 0 and 
     * stddev = sqrt(2/(number of inputs in the inputs layer x number of outputs in the output layer)),
     */
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

    /**
     * Performs the forward pass on the inputs following the defined lstm formulas
     * 
     * @param input vector of the input features
     * @return hidden state vector after passing through the various gates
     */
    Eigen::VectorXd forwardPass(const Eigen::VectorXd& input);
    /**
     * Performs the backward pass unrolling the forward functions, manipulating them using the chain rule 
     * to calculate the gradients of loss with respect to the weights and biases to be used for optimisation
     * 
     * @param deltaH the gradient of loss with respect to the hidden state dL/dh
     * @param deltaC the gradient of loss with respect to the cell state dL/dc
     * @return std::pair containing dL/dh_(t-1) and dL/dc_(t-1), previous gradients of loss wrt. hidden and cell states
     */
    std::pair<Eigen::VectorXd, Eigen::VectorXd> backwardPass(const Eigen::VectorXd& deltaH, const Eigen::VectorXd& deltaC);

    /**
     * Get total num of elements in the weights and biases
     */
    size_t getParameterCount() const; 

    /**
     * clear stepdata and clear hidden and cell state
     */
    void reset(); 

    Eigen::VectorXd getHiddenState() const {return this->hiddenState;}

    Eigen::VectorXd getCellState() const {return this->cellState;}

    Eigen::VectorXd& getParametersVector();

    Eigen::VectorXd& getGradientsVector();

    /**
     * Sets the parameters vector to the passed in vector
     * 
     * @param v the vector to set the parameters vector to
     */
    void setParametersVector(const Eigen::VectorXd& v);

    /**
     * reset all gradient accumulators (deltaW*, deltaU*, deltaB*) to zero
     */
    void zeroGrad();

    /**
     * Saves the parameters in binary to the definied filepath
     * 
     * @param path the path to save the binary to
     */
    void saveParameters(const std::string& path) const;

    /**
     * loads the parameters from the binary at the location of the filepath
     * 
     * @param path filepath to load from
     */
    void loadParameters(const std::string& path);

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