#pragma once
#include <Eigen/Dense>
#include <cassert>

struct Dense {
    Eigen::RowVectorXd W;  // [1×hiddenSize] weight
    double b;   // [1] bias
    Eigen::RowVectorXd dW; // grad for W
    double db; // grad for b

    Dense(int hiddenSize)
      : W(Eigen::RowVectorXd::Random(hiddenSize)*0.01),
        b(0.0),
        dW(Eigen::RowVectorXd::Zero(hiddenSize)),
        db(0.0) {}

    double forward(const Eigen::VectorXd &h) const {
        assert(h.size() == W.size());   // ensures size is the same, if not terminate
        // y = W·h + b
        // [1 x n] x [n x 1] = [1 x 1]
        return W.dot(h) + b;
    }

    void backward(const Eigen::VectorXd &h, double dLdy) {
        assert(h.size() == W.size());   // ensures size is the same, if not terminate
        // accumulate grads
        dW += dLdy * h.transpose();
        db += dLdy;
    }

    void zeroGrad() {
        dW.setZero();
        db = 0.0;
    }
};