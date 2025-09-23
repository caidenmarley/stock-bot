#include "inputs/parser.h"
#include "inputs/stock_data.h"
#include "model/lstm.h"
#include "model/ada_belief.h"
#include "model/huber_loss_function.h"
#include "model/dense.h"
#include "model/trainer.h"
#include "search/hyperparam_search.h"
#include <iostream>
#include <vector>
#include <iomanip>

// struct TrainerParams{
//     // hyperparameters
//     int numFeatures = 6;
//     int hiddenSize = 64;  // dimension of lstm matrices
//     int sequenceLength = 15; // number of days per sequence
//     int batchSize = 10; // number of sequences per batch
//     double learningRate = 1e-3;
//     size_t windowSize = 256; // number of days in each scaler window
//     double maxNorm = 0.001;
//     double decayFactor = 0.5; // learning rate halves at each plateau
//     double minLR = 1e-6; // minimum learning rate
//     int lrDecayMaxTries = 3; // gives up after 3 decays
//     double delta = 1.0; // huber loss delta value
//     int epochs = 30;
//     double stoppingToleranceLoss = 1e-4;
//     int maxEpochsWithNoImprovement = 3;
// };

static TrainingResult runFold(
    const std::vector<PriceData>& rawData,
    int splitStart,
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
    int epochs,
    double stoppingToleranceLoss,
    int maxEpochsWithNoImprovement
){
    std::vector<PriceData> trainingData(rawData.begin(), rawData.begin() + splitStart);
    std::vector<PriceData> validationData(rawData.begin() + splitStart, rawData.end());

    Trainer trainer(
        numFeatures, hiddenSize, sequenceLength, batchSize, learningRate, delta,
        windowSize, maxNorm, decayFactor, minLR, lrDecayMaxTries, 
        trainingData, validationData
    );

    return trainer.run(epochs, stoppingToleranceLoss, maxEpochsWithNoImprovement);
}

int main(int argc, char* argv[]) {
    try {
        //bool test = false;

        // TrainerParams trainerParams;

        // ---HyperParameters---
        int numFeatures = 6;
        int hiddenSize = 64;  // dimension of lstm matrices
        int sequenceLength = 15; // number of days per sequence
        int batchSize = 10; // number of sequences per batch
        double learningRate = 1e-3;
        size_t windowSize = 256; // number of days in each scaler window
        double maxNorm = 1.0;
        double decayFactor = 0.5; // learning rate halves at each plateau
        double minLR = 1e-6; // minimum learning rate
        int lrDecayMaxTries = 3; // gives up after 3 decays
        double delta = 1.0; // huber loss delta value
        int epochs = 10;
        double stoppingToleranceLoss = 1e-6;
        int maxEpochsWithNoImprovement = 3;

        // for rng so you can compare changes
        int seed = 0; // default
        bool givenSeed = false;

        // ---Arg Parsing---
        for(int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if(arg == "--epochs" && i+1 < argc) {
                epochs = std::stoi(argv[++i]);
            }else if(arg == "--early-stop-eps" && i+1 < argc) {
                stoppingToleranceLoss = std::stod(argv[++i]);
            }else if(arg == "--early-stop-patience" && i+1 < argc) {
                maxEpochsWithNoImprovement = std::stoi(argv[++i]);
            }else if(arg == "--test"){
                //test = true;
            }else if(arg == "--seed"){
                seed = std::stoi(argv[++i]);
                givenSeed = true;
            }else if(arg == "--help") {
                std::cout
                  << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  --epochs N                     Train up to N epochs (default 10)\n"
                  << "  --early-stop-eps X             Early-stop tol. (default 1e-3)\n"
                  << "  --early-stop-patience P        Early-stop patience (default 3)\n"
                  << "  --seed S                       Set RNG seed \n"
                  << "  --test                         Run hyperparameter search\n";
                return 0;
            }
        }

        if(givenSeed){
            LSTMCell::setGlobalInitSeed(seed);
        }

        // ---Load Data From CSV--
        const std::string csvPath = "data/AAAU.csv";
        CSVLoader loader(csvPath);
        const std::vector<PriceData>& rawData = loader.getData();
        std::cout << "parsed " << rawData.size() << " rows from CSV" << std::endl;

        // ---Rolling validation---
        int maxStartSequence = rawData.size() - sequenceLength;

        // fold cut points at 60%, 70%, 80% of maxStartSequence
        std::vector<int> cutPoints{
            static_cast<int>(0.6*maxStartSequence),
            static_cast<int>(0.7*maxStartSequence),
            static_cast<int>(0.8*maxStartSequence)
        };

        std::vector<TrainingResult> foldResults;
        foldResults.reserve(cutPoints.size());

        // Run trainer on each split
        for(size_t i = 0; i < cutPoints.size(); i++){
            int splitStart = cutPoints[i];
            std::cout << "[FOLD " << (i+1) << "] train=[0," << splitStart << "] val=["
                      << splitStart << "," << rawData.size() << "]" << std::endl; 

            TrainingResult result = runFold(
                rawData, splitStart, numFeatures, hiddenSize,
                sequenceLength, batchSize, learningRate, delta,
                windowSize, maxNorm, decayFactor, minLR, lrDecayMaxTries,
                epochs, stoppingToleranceLoss, maxEpochsWithNoImprovement
            );

            foldResults.push_back(result);
        }

        double avg = 0.0;
        double best = 1e9;
        for(size_t i = 0; i < cutPoints.size(); i++){
            double val = foldResults[i].bestValLoss;
            avg += val;
            if(val < best) best = val;
        }
        avg /= foldResults.size() ? foldResults.size() : 1;

        std::cout << "[SUMMARY] avg best val loss = " << std::fixed 
                  << std::setprecision(6) << avg << " | best fold loss = " << best << std::endl;



        // // split data into 80/20 train/validation split
        // int maxStartSequence = rawData.size() - sequenceLength;  
        // int splitStart = 0.8*maxStartSequence;
        
        // std::vector<PriceData> trainingData(rawData.begin(), rawData.begin() + splitStart);
        // std::vector<PriceData> validationData(rawData.begin() + splitStart, rawData.end());

        // Trainer trainer(
        //     numFeatures,
        //     hiddenSize,
        //     sequenceLength,
        //     batchSize, 
        //     learningRate,
        //     delta,
        //     windowSize,
        //     maxNorm,
        //     decayFactor,
        //     minLR,
        //     lrDecayMaxTries,
        //     trainingData,
        //     validationData
        // );

        // trainer.run(epochs, stoppingToleranceLoss, maxEpochsWithNoImprovement);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
