#include "inputs/parser.h"
#include "model/trainer.h"
#include "search/hyperparam_search.h"
#include <cstdio>
#include <cxxopts.hpp>
#include <fmt/base.h>
#include <vector>

auto main(int argc, char* argv[]) -> int {
    try {
        bool test;

        // hyperparameters
        int numFeatures = 6;
        int hiddenSize = 32;     // dimension of lstm matrices
        int sequenceLength = 20; // number of days per sequence
        int batchSize = 10;      // number of sequences per batch
        double learningRate = 1e-3;
        size_t windowSize = 256; // number of days in each scaler window
        double maxNorm = 0.001;
        double decayFactor = 0.5; // learning rate halves at each plateau
        double minLR = 1e-6;      // minimum learning rate
        int lrDecayMaxTries = 3;  // gives up after 3 decays
        double delta = 1.0;       // huber loss delta value
        int epochs;
        double stoppingToleranceLoss;
        int maxEpochsWithNoImprovement;

        cxxopts::Options options {"Stock Bot", ""};
        options.add_options()
			("e,epochs", "Train up to N epochs", cxxopts::value<int>()->default_value("30"))
			("early-stop-eps", "Early stop tol", cxxopts::value<double>()->default_value("1e-4"))
			("early-stop-patience", "Early stop patience", cxxopts::value<int>()->default_value("3"))
			("t,test", "Run hyperparameter search")
			("h,help", "Print usage");
        auto result = options.parse(argc, argv);

        if (result.contains("help")) {
            fmt::println("{}", options.help());
            return 0;
        }

        stoppingToleranceLoss = result["early-stop-eps"].as<double>();
        maxEpochsWithNoImprovement = result["early-stop-patience"].as<int>();
        epochs = result["epochs"].as<int>();
        test = result["test"].as<bool>();

        // load data
        const std::string csvPath = "data/AAAU.csv";
        CSVLoader loader(csvPath);
        const std::vector<PriceData> rawData = loader.getData();

        fmt::println("Parsed {} rows from CSV", rawData.size());

        // split data into 80/20 train/validation split
        int maxStartSequence = rawData.size() - sequenceLength;
        int splitStart = 0.8 * maxStartSequence;

        std::vector<PriceData> trainingData(
            rawData.begin(), rawData.begin() + (splitStart + sequenceLength)
        );
        std::vector<PriceData> validationData(
            rawData.begin() + splitStart, rawData.end()
        );

        if (test) {
            std::vector<HyperParam> params = {
                {"hiddenSize", ParamType::INTEGER, {16, 32, 64}},
                {"sequenceLength", ParamType::INTEGER, {10, 20, 40}},
                {"batchSize", ParamType::INTEGER, {16, 32, 64}},
                {"learningRate", ParamType::DOUBLE, {1e-3, 1e-4, 1e-5}},
                {"windowSize", ParamType::INTEGER, {64, 128, 256}},
                {"delta", ParamType::DOUBLE, {0.5, 1.0, 2.0}}
            };

            gridSearch(
                params,
                numFeatures,
                hiddenSize,
                sequenceLength,
                batchSize,
                learningRate,
                delta,
                windowSize,
                maxNorm,
                decayFactor,
                minLR,
                lrDecayMaxTries,
                trainingData,
                validationData,
                epochs,
                stoppingToleranceLoss,
                maxEpochsWithNoImprovement
            );
        } else {
            Trainer trainer(
                numFeatures,
                hiddenSize,
                sequenceLength,
                batchSize,
                learningRate,
                delta,
                windowSize,
                maxNorm,
                decayFactor,
                minLR,
                lrDecayMaxTries,
                trainingData,
                validationData
            );

            trainer.run(
                epochs, stoppingToleranceLoss, maxEpochsWithNoImprovement
            );
        }

    } catch (const std::exception& ex) {
        fmt::println(stderr, "Error: {}", ex.what());
        return 1;
    }
}
