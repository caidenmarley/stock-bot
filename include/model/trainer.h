#pragma once

#include "model/ada_belief.h"
#include "model/dense.h"
#include "model/huber_loss_function.h"
#include "model/lstm.h"
#include "inputs/shape_inputs.h"

struct TrainingResult{
    double bestValLoss;
    int epochOfBestValLoss;
    int totalEpochs;
};

class Trainer{
public:
    Trainer(int numFeatures, int hiddenSize, int sequenceLength, int batchSize, double learningRate, double delta,
        size_t windowSize, const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData);
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
    StockData trainingData;
    StockData validationData;
};