#include "model/ada_belief.h"

AdaBelief::AdaBelief(std::size_t vecSize, double learningRate, double b1, double b2, double e) :
    learningRate(learningRate), b1(b1), b2(b2), e(e), time(0), b1Power(1.0), b2Power(1.0), // b1^0 and b2^0
    m(Eigen::VectorXd::Zero(vecSize)), s(Eigen::VectorXd::Zero(vecSize)){}

void AdaBelief::update(Eigen::VectorXd& params, const Eigen::VectorXd& gradients){
    this->time++; // you want to start at t=1 so you arent dividing by 0

    // Running power to increase b1 power to the power of time as you go to avoid pow() calls
    b1Power *= b1;
    b2Power *= b2;

    auto gradArr = gradients.array();

    // m_t = b_1*m_t-1 + (1-b_1)g_t
    this->m = b1*m + (1.0-b1)*gradArr;
    
    // s_t = b_2*s_t-1 + (1-b_2)(g_t-m_t)^2 + e
    this->s = b2*s + (1.0-b2)*(gradArr-m).square();

    // m_t / 1 - b_1^t
    Eigen::ArrayXd mBiasCorrection = m / (1-b1Power);

    // s_t / 1 - b_2^t
    Eigen::ArrayXd sBiasCorrection = s / (1-b2Power);

    params.array() -= learningRate*mBiasCorrection / (sBiasCorrection.sqrt() + e);
}