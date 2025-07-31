#include "inputs/parser.h"
#include "inputs/shape_inputs.h"
#include "model/lstm.h"
#include "model/ada_belief.h"
#include "model/huber_loss_function.h"
#include "model/dense.h"
#include "model/trainer.h"
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        // hyperparameters
        int numFeatures = 6;
        int hiddenSize = 32;  // dimension of lstm matrices
        int sequenceLength = 20; // number of days per sequence
        int batchSize = 10; // number of sequences per batch
        double learningRate = 1e-4;
        double windowSize = 256; // number of days in each scaler window
        double delta = 1.0; // huber loss delta value
        int epochs = 10;
        double stoppingToleranceLoss = 1e-3;
        int maxEpochsWithNoImprovement = 3;

        for(int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if(arg == "--epochs" && i+1 < argc) {
                epochs = std::stoi(argv[++i]);
            }
            else if(arg == "--early-stop-eps" && i+1 < argc) {
                stoppingToleranceLoss = std::stod(argv[++i]);
            }
            else if(arg == "--early-stop-patience" && i+1 < argc) {
                maxEpochsWithNoImprovement = std::stoi(argv[++i]);
            }
            else if(arg == "--help") {
                std::cout
                  << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  --epochs N                     Train up to N epochs (default 10)\n"
                  << "  --early-stop-eps X             Early-stop tol. (default 1e-3)\n"
                  << "  --early-stop-patience P        Early-stop patience (default 3)\n";
                return 0;
            }
        }

        // load data
        const std::string csvPath = "data/AAAU.csv";
        CSVLoader loader(csvPath);
        const std::vector<PriceData> rawData = loader.getData();
        std::cout << "parsed " << rawData.size() << " rows from CSV" << std::endl;

        // split data into 80/20 train/validation split
        int maxStartSequence = rawData.size() - sequenceLength;  
        int splitStart = 0.8*maxStartSequence;
        
        std::vector<PriceData> trainingData(rawData.begin(), rawData.begin() + (splitStart + sequenceLength));
        std::vector<PriceData> validationData(rawData.begin() + splitStart, rawData.end());

        Trainer trainer(
            numFeatures,
            hiddenSize,
            sequenceLength,
            batchSize, 
            learningRate,
            delta,
            windowSize,
            trainingData,
            validationData
        );

        trainer.run(epochs, stoppingToleranceLoss, maxEpochsWithNoImprovement);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
