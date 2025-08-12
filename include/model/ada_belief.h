#pragma once

#include <Eigen/Dense>
#include <cstdint>

class AdaBelief{
public:
    /**
     * Initialise all member variables and set m and s arrays to all be 0
     * 
     * @param vecSize the number of parameters to be optimised
     * @param learningRate learning rate of model
     * @param b1 decay rate for running average of past gradients
     * @param b2 decay rate for the running average of the squared difference between the gradient and its belief
     * @param e avoid division by 0
     */
    AdaBelief(std::size_t vecSize, double learningRate, double b1 = 0.9, double b2 = 0.999, double e = 1e-8);
    /**
     * Performs the optimisation algorithm to tweat the params based on the gradients
     * 
     * @param params the parameters passed in (the weights and biases)
     * @param gradients the calculated gradients from the backwards pass
     */
    void update(Eigen::VectorXd& params, const Eigen::VectorXd& gradients);

    double getLearningRate() const { return learningRate; }
    void setLearningRate(double lr) { learningRate = lr; }
private:
    double learningRate;
    double b1, b2;
    double e;
    size_t time;
    double b1Power, b2Power;    // runnning powers so no need for pow call
    double oneMinusb1, oneMinusb2; // precompute values so you dont need to calculate every update call

    Eigen::ArrayXd m;  // exponential moving average of gradient
    Eigen::ArrayXd s;  // EMA of (g - m)^2 (belief in step size)
};