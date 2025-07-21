#pragma once

#include "model/ada_belief.h"
#include "model/dense.h"
#include "model/huber_loss_function.h"
#include "model/lstm.h"
#include "parser/shape_inputs.h"

class Trainer{
public:
    Trainer(int numFeatures, int hiddenSize, int sequenceLength, int batchSize, double learningRate, double delta,
        const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData);
    void run(int epochs);
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