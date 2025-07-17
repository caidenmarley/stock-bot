#pragma once
#include <Eigen/Dense>

class HuberLossFunction{
public:
    HuberLossFunction(double delta);
    double forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets);

    Eigen::VectorXd backward();
private:
    double delta;
    double halfDelta;
    Eigen::ArrayXd residuals; // target - prediction
};