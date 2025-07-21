#include "model/huber_loss_function.h"

HuberLossFunction::HuberLossFunction(double delta) : delta(delta), halfDelta(delta*0.5){}

// overloaded forward pass, which converts doubles into singular vectors to be used in the regular forward pass func
double HuberLossFunction::forward(const double& prediction, const double& target){
    Eigen::VectorXd predictionVector(1), targetVector(1);
    predictionVector(0) = prediction;
    targetVector(0) = target;
    return this->forward(predictionVector, targetVector);
}

double HuberLossFunction::forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets){
    this->residuals = (targets - predictions).array();

    auto residualsAbs = residuals.abs();

    if(buffer.size() != residuals.size()){
        // single resize so you can reuse buffer on later calls
        buffer.resize(residuals.size());
    }

    // quadratic term: 0.5 * r^2

    // linear term: delta * (abs(r) - 0.5 * delta)

    // select quadratic if abs(r) <= delta else linear
    // .select: conditionType, then expression, else expression

    // does the selection in a single pass
    buffer = (residualsAbs <= delta).select(0.5*residuals.square(), delta*(residualsAbs - halfDelta));

    return buffer.mean();
}

Eigen::VectorXd HuberLossFunction::backward(){
    if(buffer.size() != residuals.size()){
        // single resize so you can reuse buffer on later calls
        buffer.resize(residuals.size());
    }

    // dL/dr (L is loss, r is residuals)
    // r if abs(r) <= delta, delta
    // delta*(d/dr)*abs(r) so delta*sign(r) if abs(r) > delta

    // reuse buffer to save allocation
    buffer = (residuals.abs() <= delta).select(residuals, delta*residuals.sign());

    // r = target - prediction so dr/d(prediction) = -1
    // so dL/d(prediction) = -dL/dr, then average over n (mean loss)
    return (-buffer / residuals.size()).matrix();
}