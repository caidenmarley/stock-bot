#include "inputs/parser.h"
#include "inputs/shape_inputs.h"
#include "model/lstm.h"
#include "model/ada_belief.h"
#include "model/huber_loss_function.h"
#include "model/dense.h"
#include "model/trainer.h"
#include <iostream>

int main() {
    try {
        // load data
        const std::string csvPath = "data/AAAU.csv";
        CSVLoader loader(csvPath);
        const std::vector<PriceData> rawData = loader.getData();
        std::cout << "parsed " << rawData.size() << " rows from CSV" << std::endl;

        // hyperparameters
        int numFeatures = 6;
        int hiddenSize = 32;  // dimension of lstm matrices
        int sequenceLength = 20; // number of days per sequence
        int batchSize = 10; // number of sequences per batch
        double learningRate = 1e-4;
        double windowSize = 256;
        double delta = 1.0; // huber loss delta value
        int epochs = 10;

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

        trainer.run(epochs);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
