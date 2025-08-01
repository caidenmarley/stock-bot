#include "search/hyperparam_search.h"
#include <unordered_map>
#include <iostream>

void gridSearch(
    const std::vector<HyperParam>& params,
    int numFeatures, 
    int hiddenSize, 
    int sequenceLength, 
    int batchSize, 
    double learningRate, 
    double delta,
    size_t windowSize, 
    const std::vector<PriceData>& rawTrainingData, 
    const std::vector<PriceData>& rawValidationData,
    int epochs,
    double stoppingToleranceLoss, 
    int maxEpochsWithNoImprovement
){
    // each element corresponds to the index into the "values" vector in the corresponding HyperParam struct
    std::vector<size_t> indexs(params.size(), 0);
    bool done = false;

    while(!done){
        // could covert to a map (tree) is ordering when logging becomes a hassle
        std::unordered_map<std::string, double> hashmap; // map each hyperparam name to its chosen value for this loop
        for(size_t i = 0; i < params.size(); ++i){
            double value = params[i].values[indexs[i]]; // look at current index to choose which value to use for this loop
            if(params[i].type == ParamType::INTEGER){
                // ensures no accidental fractions on integer types
                hashmap[params[i].name] = static_cast<int>(value);
            }else{
                hashmap[params[i].name] = value;
            }
        }

        // if value exists in map, use value in map, else use default value passed in
        int testHiddenSize = hashmap.count("hiddenSize") ? int(hashmap["hiddenSize"]) : hiddenSize;
        int testSequenceLength = hashmap.count("sequenceLength") ? int(hashmap["sequenceLength"]) : sequenceLength;
        int testBatchSize = hashmap.count("batchSize") ? int(hashmap["batchSize"]) : batchSize;
        double testLearningRate = hashmap.count("learningRate") ? hashmap["learningRate"] : learningRate; 
        double testDelta = hashmap.count("delta") ? hashmap["delta"] : delta; 
        size_t testWindowSize = hashmap.count("windowSize") ? size_t(hashmap["windowSize"]) : windowSize;
        
        Trainer trainer(
            numFeatures,
            testHiddenSize,
            testSequenceLength,
            testBatchSize,
            testLearningRate,
            testDelta,
            testWindowSize,
            rawTrainingData,
            rawValidationData
        );

        TrainingResult result = trainer.run(epochs, stoppingToleranceLoss, maxEpochsWithNoImprovement);

        std::cout << "---------------------------------------" << std::endl;
        for(auto& m: hashmap){
            std::cout << m.first << " = " << m.second << std::endl; 
        }
        std::cout << "bestValLoss = " << result.bestValLoss
                  << " at epoch " << result.epochOfBestValLoss
                  << ", total epochs = " << result.totalEpochs
                  << std::endl;

        // work from the outside in{X,Y,Z}, Z to X Z lsb, cycling through all possible values
        for(int i = int(params.size())-1; i >=0; --i){
            // check if position exists without overflow
            if(++indexs[i] < params[i].values.size()){
                // position exists, so try this one next
                break;
            }
            // if position didnt exist, reset and carry on to next position
            indexs[i] = 0;
            if(i == 0){
                // if first index has been fully checked then finished
                done = true;
            }
        }
    }
}