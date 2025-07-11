#include "model/huber_loss_function.h"

HuberLossFunction::HuberLossFunction(double delta) : delta(delta){}

double HuberLossFunction::forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets){
    this->residuals = targets - predictions;

    // quadratic term 0.5 * r^2
    Eigen::VectorXd quadraticTerm = 0.5*residuals.array().square();

    // delta * (abs(r) - 0.5 * delta)
    Eigen::VectorXd linearTerm = delta*(residuals.array().abs() - 0.5*delta);

    // select quadratic if abs(r) <= delta else linear
    // .select: conditionType, then expression, else expression
    Eigen::VectorXd lossVector = (residuals.array().abs() <= delta).select(quadraticTerm, linearTerm);

    return lossVector.mean();
}

Eigen::VectorXd HuberLossFunction::backward(){
    // dL/dr (L is loss, r is residuals)
    // r if abs(r) <= delta, delta
    // delta*(d/dr)*abs(r) so delta*sign(r) if abs(r) > delta

    Eigen::VectorXd rGradients = (residuals.array().abs() <= delta).select(residuals.array(), delta*residuals.array().sign());

    // r = target - prediction so dr/d(prediction) = -1
    // so dL/d(prediction) = -dL/dr, then average over n (mean loss)
    return -rGradients / residuals.size();
}