#pragma once

#include <string>
#include <vector>
#include "model/trainer.h"
#include "inputs/shape_inputs.h"

enum class ParamType {INTEGER, DOUBLE};

struct HyperParam{
    std::string name;   // e.g. learning rate
    ParamType type; // INTEGER or DOUBLE, type to cast value to
    std::vector<double> values; // values to sweep over
};

/**
 * Performs a grid search based on multiple options for values on each parameter type, iterating over every possibility
 * 
 * @param params vector of HyperParam structs containing all testing info for that hyperparam
 * @param ALLHYPERPARAMS default values incase you dont want to include these in the search
 */
void gridSearch(
    const std::vector<HyperParam>& params,
    int numFeatures, 
    int hiddenSize, 
    int sequenceLength, 
    int batchSize, 
    double learningRate, 
    double delta,
    size_t windowSize, 
    double maxNorm,
    double decayFactor,
    double minLR,
    int lrDecayMaxTries,
    const std::vector<PriceData>& rawTrainingData, 
    const std::vector<PriceData>& rawValidationData,
    int epochs,
    double stoppingToleranceLoss, 
    int maxEpochsWithNoImprovement
);

void randomSearch(
    const std::vector<HyperParam>& params,
    int numFeatures, 
    int hiddenSize, 
    int sequenceLength, 
    int batchSize, 
    double learningRate, 
    double delta,
    size_t windowSize, 
    double maxNorm,
    double decayFactor,
    double minLR,
    int lrDecayMaxTries,
    const std::vector<PriceData>& rawTrainingData, 
    const std::vector<PriceData>& rawValidationData,
    int epochs,
    double stoppingToleranceLoss, 
    int maxEpochsWithNoImprovement,
    int nTrials
);