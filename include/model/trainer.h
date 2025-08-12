#pragma once

#include "model/ada_belief.h"
#include "model/dense.h"
#include "model/huber_loss_function.h"
#include "model/lstm.h"
#include "inputs/shape_inputs.h"
#include <Eigen/Dense>

struct TrainingResult{
    double bestValLoss;
    int epochOfBestValLoss;
    int totalEpochs;
};

class Trainer{
public:
    Trainer(int numFeatures, int hiddenSize, int sequenceLength, int batchSize, double learningRate, double delta,
        size_t windowSize, double maxNorm, double decayFactor, double minLR, int lrDecayMaxTries,
        const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData);
    TrainingResult run(const int epochs, double stoppingToleranceLoss, int maxEpochsWithNoImprovement);
private:
    LSTMCell lstm;
    Dense outputLayer;
    AdaBelief optimiser;
    HuberLossFunction huberLoss;
    int sequenceLength;
    int batchSize;
    double learningRate;
    size_t windowSize;
    double maxNorm;
    double decayFactor;
    double minLR;
    int lrDecayMaxTries;
    StockData trainingData;
    StockData validationData;

    static inline void clipGlobalNorm(Eigen::VectorXd& g, double maxNorm){
        double n = g.norm(); // magnitude of grads
        if(n > maxNorm && n > 0.0){
            g *= (maxNorm / n);
        }
    }
};