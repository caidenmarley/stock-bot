#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <cstdint>
#include <optional>
#include <random>

struct Dense {
    Eigen::RowVectorXd W;  // [1×hiddenSize] weight
    double b;   // [1] bias
    Eigen::RowVectorXd dW; // grad for W
    double db; // grad for b

    /**
     * Initalises member variables, randomly initialises the weights and sets the bias and dW all to 0
     * 
     * @param hiddenSize the size of the hidden layer
     */
    Dense(int hiddenSize)
      : W(Eigen::RowVectorXd::Random(hiddenSize)*0.01),
        b(0.0),
        dW(Eigen::RowVectorXd::Zero(hiddenSize)),
        db(0.0) {}

        /**
         * Optional deterministic initialization path for reproducibility-sensitive runs.
         * If no seed is provided, behavior should match the default constructor behavior.
         *
         * @param hiddenSize the size of the hidden layer
         * @param initSeed optional RNG seed for deterministic initial weights
         */
        Dense(int hiddenSize, std::optional<uint32_t> initSeed)
            : W(initSeed ? makeSeededWeights(hiddenSize, *initSeed)
                                     : (Eigen::RowVectorXd::Random(hiddenSize) * 0.01)),
                b(0.0),
                dW(Eigen::RowVectorXd::Zero(hiddenSize)),
                db(0.0) {}

    /**
     * Linear forward pass y = Wh + b
     * 
     * @param h hidden state
     * @return the result of the linear pass
     */
    double forward(const Eigen::VectorXd &h) const {
        assert(h.size() == W.size());   // ensures size is the same, if not terminate
        // y = Wh + b
        // [1 x n] x [n x 1] = [1 x 1]
        return W.dot(h) + b;
    }

    /**
     * Linear backward pass to accumulate gradients
     * 
     * @param h hidden state
     * @param dLdy rate of change of loss with respect to output of forward pass
     */
    void backward(const Eigen::VectorXd &h, double dLdy) {
        assert(h.size() == W.size());   // ensures size is the same, if not terminate
        // accumulate grads
        dW += dLdy * h.transpose();
        db += dLdy;
    }

    /**
     * sets dW and db to 0
     */
    void zeroGrad() {
        dW.setZero();
        db = 0.0;
    }

private:
    static Eigen::RowVectorXd makeSeededWeights(int hiddenSize, uint32_t seed) {
        Eigen::RowVectorXd out(hiddenSize);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < hiddenSize; ++i) {
            out(i) = dist(rng) * 0.01;
        }

        return out;
    }
};