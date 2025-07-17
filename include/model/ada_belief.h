#pragma once

#include <Eigen/Dense>
#include <cstdint>

class AdaBelief{
public:
    AdaBelief(std::size_t vecSize, double learningRate, double b1 = 0.9, double b2 = 0.999, double e = 1e-8);
    void update(Eigen::VectorXd& params, const Eigen::VectorXd& gradients);
private:
    double learningRate;
    double b1, b2;
    double e;
    size_t time;
    double b1Power, b2Power;

    Eigen::ArrayXd m;  // exponential moving average of gradient
    Eigen::ArrayXd s;  // EMA of (g - m)^2 (belief in step size)
};