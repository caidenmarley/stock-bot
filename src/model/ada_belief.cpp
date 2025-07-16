#include "model/ada_belief.h"

AdaBelief::AdaBelief(std::size_t vecSize, double learningRate, double b1 = 0.9, double b2 = 0.999, double e = 1e-8) :
    learningRate(learningRate), b1(b1), b2(b2), e(e), time(0),
    m(Eigen::VectorXd::Zero(vecSize)), s(Eigen::VectorXd::Zero(vecSize)){}

void AdaBelief::update(Eigen::VectorXd& params, const Eigen::VectorXd& gradients){
    this->time++; // you want to start at t=1 so you arent dividing by 0

    // m_t = b_1*m_t-1 + (1-b_1)g_t
    this->m = b1*m + (1.0-b1)*gradients;
    
    // s_t = b_2*s_t-1 + (1-b_2)(g_t-m_t)^2 + e
    this->s = b2*s + (1.0-b2)*(gradients-m).array().square().matrix();

    // m_t / 1 - b_1^t
    Eigen::VectorXd mBiasCorrection = m / (1-std::pow(b1, time));

    // s_t / 1 - b_2^t
    Eigen::VectorXd sBiasCorrection = s / (1-std::pow(b2, time));

    params.array() -= learningRate*mBiasCorrection.array() / (sBiasCorrection.array().sqrt() + e);
}