#pragma once

#include "model/ada_belief.h"
#include "model/dense.h"
#include "model/huber_loss_function.h"
#include "model/lstm.h"
#include "model/metrics.h"
#include "inputs/stock_data.h"
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <string>

struct TrainingResult{
    double bestValLoss;
    int epochOfBestValLoss;
    int totalEpochs;
    double finalValSharpeNet{0.0};
    double finalValAvgTurnover{0.0};
    std::vector<metrics::BenchmarkSummary> finalValBenchmarkRows{};
};

class Trainer{
public:
    Trainer(int numFeatures, int hiddenSize, int sequenceLength, int batchSize, double learningRate, double delta,
        size_t windowSize, double maxNorm, double decayFactor, double minLR, int lrDecayMaxTries,
        const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData,
        const std::string& resultsFilePath = "",
        std::optional<uint32_t> denseInitSeed = std::nullopt);
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
    std::string resultsFilePath;
    bool writeResults;
    StockData trainingData;
    StockData validationData;

    static inline void clipGlobalNorm(Eigen::VectorXd& g, double maxNorm){
        double n = g.norm(); // magnitude of grads
        if(n > maxNorm && n > 0.0){
            g *= (maxNorm / n);
        }
    }
};