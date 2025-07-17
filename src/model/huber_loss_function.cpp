#include "model/huber_loss_function.h"

HuberLossFunction::HuberLossFunction(double delta) : delta(delta), halfDelta(delta*0.5){}

double HuberLossFunction::forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets){
    this->residuals = (targets - predictions).array();

    auto residualsAbs = residuals.abs();

    // quadratic term 0.5 * r^2
    Eigen::ArrayXd quadraticTerm = 0.5*residuals.square();

    // delta * (abs(r) - 0.5 * delta)
    Eigen::ArrayXd linearTerm = delta*(residualsAbs - halfDelta);

    // select quadratic if abs(r) <= delta else linear
    // .select: conditionType, then expression, else expression
    Eigen::ArrayXd lossVector = (residualsAbs <= delta).select(quadraticTerm, linearTerm);

    return lossVector.mean();
}

Eigen::VectorXd HuberLossFunction::backward(){
    // dL/dr (L is loss, r is residuals)
    // r if abs(r) <= delta, delta
    // delta*(d/dr)*abs(r) so delta*sign(r) if abs(r) > delta

    Eigen::VectorXd rGradients = (residuals.abs() <= delta).select(residuals, delta*residuals.sign());

    // r = target - prediction so dr/d(prediction) = -1
    // so dL/d(prediction) = -dL/dr, then average over n (mean loss)
    return -rGradients / residuals.size();
}