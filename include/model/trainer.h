#pragma once

#include "model/ada_belief.h"
#include "model/dense.h"
#include "model/huber_loss_function.h"
#include "model/lstm.h"
#include "inputs/shape_inputs.h"

class Trainer{
public:
    Trainer(const int numFeatures, const int hiddenSize, const int sequenceLength, const int batchSize, const double learningRate, const double delta,
        const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData);
    void run(const int epochs);
private:
    LSTMCell lstm;
    Dense outputLayer;
    AdaBelief optimiser;
    HuberLossFunction huberLoss;
    int sequenceLength;
    int batchSize;
    double learningRate;
    StockData trainingData;
    StockData validationData;
};