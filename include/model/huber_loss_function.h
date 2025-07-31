#pragma once
#include <Eigen/Dense>

class HuberLossFunction{
public:
    /**
     * Initialises member variables delta and halfDelta
     * 
     * @param delta value to determine at what point to use linear and quadratic terms
     */
    HuberLossFunction(double delta);
    /**
     * Compares difference between targets and predictions and chooses either the linear
     * or quadratic term based on whether the absolute value of the difference it greater or less
     * than delta
     * 
     * @param prediction single value the model predicted
     * @param target the single actual value
     * @return the mean of the buffer which contains all the selected loss values
     */
    double forward(const double& prediction, const double& target);
    /**
     * Compares difference between targets and predictions and chooses either the linear
     * or quadratic term based on whether the absolute value of the difference it greater or less
     * than delta
     * 
     * @param prediction vector of the models predictions
     * @param target vector of the actual values
     * @return the mean of the buffer which contains all the selected loss values
     */
    double forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets);

    /**
     * Computes the gradient of loss with respect ot the models predictions
     * 
     * @return vector of gradients
     */
    Eigen::VectorXd backward();
private:
    double delta;
    double halfDelta;
    Eigen::ArrayXd residuals; // target - prediction
    Eigen::ArrayXd buffer; // full buffer to select quadratic or linear term wihtout having to declare multiple ArrayXd
};