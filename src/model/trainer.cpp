#include "model/trainer.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <numeric>
#include <utility>

Trainer::Trainer(int numFeatures, int hiddenSize, int sequenceLength, int batchSize, double learningRate, double delta,
size_t windowSize, double maxNorm, double decayFactor, double minLR, int lrDecayMaxTries,
const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData): 
    lstm(numFeatures, hiddenSize, sequenceLength), outputLayer(hiddenSize), 
    optimiser(lstm.getParameterCount(), learningRate),  huberLoss(delta),
    sequenceLength(sequenceLength), batchSize(batchSize), learningRate(learningRate), windowSize(windowSize), 
    maxNorm(maxNorm), decayFactor(decayFactor), minLR(minLR), lrDecayMaxTries(lrDecayMaxTries),
    // Training has empty scaler
    trainingData(
        rawTrainingData, 
        numFeatures, 
        sequenceLength, 
        batchSize, 
        RollingWindowScaler(windowSize, numFeatures) // empty
    ), 
    // Validation has a temporary empty scaler to be rebuilt below with a preLoadedScaler
    validationData(
        rawValidationData, 
        numFeatures, 
        sequenceLength, 
        batchSize, 
        RollingWindowScaler(windowSize, numFeatures) // empty placeholder
    )
{
    RollingWindowScaler valScaler(windowSize, numFeatures);

    const size_t split = rawTrainingData.size(); // length of training data
    const size_t preLoadStart = (split > windowSize) ? (split - windowSize) : 0; // take windowSize worth of data
    for(size_t i = preLoadStart; i < split; ++i){
        // add windowSize number of data points before the split to the rolling window scalar
        valScaler.add(rawTrainingData[i]);
    }

    validationData = StockData(
        rawValidationData,
        numFeatures, 
        sequenceLength, 
        batchSize, 
        std::move(valScaler) // move pre loaded scaler
    );
}

TrainingResult Trainer::run(const int epochs, double stoppingToleranceLoss, int maxEpochsWithNoImprovement){
    double bestValLoss = 1000000;
    int bestEpoch = 0;
    int noImproveCount = 0;

    for(int epoch = 1; epoch <= epochs; epoch++){
        // TRAINING
        // reset trainingData position index, so batches start from the beginning
        trainingData.reset();
        double trainingLoss = 0.0;  // accumulator for training loss
        size_t trainingExamples = 0;

        // loop over all batches
        while(trainingData.hasAnotherBatch()){
            auto [inputBatch, targetBatch] = trainingData.nextBatch();
            int currentBatch = targetBatch.size();

            // reset gradients so next batch can make its own contribution to the weights
            lstm.zeroGrad();
            outputLayer.zeroGrad();

            // forward and backward pass over the current batch
            for(int i = 0; i < currentBatch; i++){
                lstm.reset();
                Eigen::VectorXd hiddenState;

                // forward pass over the batch through time = 0 to time = sequenceLength
                for(int j = 0; j < sequenceLength; j++){
                    // use a map to point at a slice of the already allocated batch tensor without creating new copies
                    // gets the feature vector for the ith sequence at time j
                    const double* ptr = inputBatch.data() + ((i*sequenceLength + j) * trainingData.getNumFeatures());
                    Eigen::Map<const Eigen::VectorXd> inputs(ptr, trainingData.getNumFeatures());

                    // single lstm step
                    hiddenState = lstm.forwardPass(inputs);
                }

                double targetPrediction = outputLayer.forward(hiddenState);

                double loss = huberLoss.forward(targetPrediction, targetBatch(i));
                Eigen::VectorXd gradient = huberLoss.backward();
                double dLdy = gradient(0);

                trainingLoss += loss;
                ++trainingExamples;

                // back pass through dense layer
                outputLayer.backward(hiddenState, dLdy);

                // back pass through lstm
                // Dense Layer:
                // y = Wh + b -> dL/dy = dLdy
                // dL/dh = dL/dy * dy/dh
                // dy/dh = W
                Eigen::VectorXd dLdhNext = outputLayer.W.transpose() * dLdy;
                // no next c as at end of sequence, so grad = 0
                Eigen::VectorXd dLdcNext = Eigen::VectorXd::Zero(hiddenState.size());   

                for(int j = sequenceLength; j > 0; j--){
                    // tie assigns the first var in the pair to the first var in the tie, then second to second
                    // dLdhNext = lstm.backwardPass().first, dLdcNext = lstm.backwardPass().second 
                    std::tie(dLdhNext, dLdcNext) = lstm.backwardPass(dLdhNext, dLdcNext);
                }
            }
            // scope for optimiser variables
            {
                Eigen::VectorXd params = lstm.getParametersVector();
                Eigen::VectorXd gradients = lstm.getGradientsVector();
                clipGlobalNorm(gradients, maxNorm);
                optimiser.update(params, gradients);
                lstm.setParametersVector(params);
            }

            // Dense layer: using SGD on W and b
            double norm = std::sqrt(outputLayer.dW.squaredNorm() + outputLayer.db * outputLayer.db);
            if (norm > maxNorm && norm > 0.0) {
                double scale = maxNorm / norm;
                outputLayer.dW *= scale;
                outputLayer.db *= scale;
            }
            outputLayer.W -= learningRate * outputLayer.dW;
            outputLayer.b -= learningRate * outputLayer.db;
        }

        double avgTrainingLoss = trainingExamples ? trainingLoss / double(trainingExamples) : 0.0;

        // VALIDATION - measures how the model does with the weights and bias it just worked out in training
        validationData.reset();
        double validationLoss = 0.0;
        size_t validationExamples = 0;

        // collect predictions & targets (TODO CAN BE REMOVED LATER)
        std::vector<double> valPreds;
        std::vector<double> valTargets;
        valPreds.reserve(10000);
        valTargets.reserve(10000);

        while(validationData.hasAnotherBatch()){
            auto [inputBatch, targetBatch] = validationData.nextBatch();
            int currentBatch = targetBatch.size();

            for(int i = 0; i < currentBatch; i++){
                lstm.reset();   // clear prev states
                Eigen::VectorXd hiddenState;
                for (int j = 0; j < sequenceLength; j++){
                    const double* ptr = inputBatch.data() + ((i*sequenceLength + j) * validationData.getNumFeatures());
                    Eigen::Map<const Eigen::VectorXd> inputs(ptr, validationData.getNumFeatures());

                    hiddenState = lstm.forwardPass(inputs);
                }

                double targetPrediction = outputLayer.forward(hiddenState);

                validationLoss += huberLoss.forward(targetPrediction, targetBatch(i));
                ++validationExamples;
                
                valPreds.push_back(targetPrediction);
                valTargets.push_back(targetBatch(i));
            }
        }

        double avgValidationLoss = validationExamples ? validationLoss / double(validationExamples) : 0.0;

        // --- extra validation metrics ---
        double se = 0.0, ae = 0.0;
        int correct = 0;
        const int nval = static_cast<int>(valPreds.size());
        for (int k = 0; k < nval; ++k) {
            double e = valTargets[k] - valPreds[k];
            se += e * e;
            ae += std::abs(e);
            if ((valPreds[k] >= 0.0) == (valTargets[k] >= 0.0)) ++correct;
        }
        double rmse = nval ? std::sqrt(se / nval) : 0.0;
        double mae  = nval ? (ae / nval) : 0.0;
        double da   = nval ? (static_cast<double>(correct) / nval) : 0.0;

        // has validation improved by at least the tolerance
        if(avgValidationLoss + stoppingToleranceLoss < bestValLoss){
            bestValLoss = avgValidationLoss;
            bestEpoch = epoch;
            noImproveCount = 0;
        }else{
            ++noImproveCount;
            if(noImproveCount >= maxEpochsWithNoImprovement){
                double opLR = optimiser.getLearningRate();
                if(lrDecayMaxTries > 0 && opLR > minLR){
                    opLR = std::max(minLR, opLR * decayFactor);
                    optimiser.setLearningRate(opLR);
                    learningRate = std::max(minLR ,learningRate * decayFactor); // for Dense layer
                    --lrDecayMaxTries;
                    noImproveCount = 0;
                    std::cout << "[LR-plateau] decayed LR to " << opLR
                      << " (dense lr " << learningRate << "), tries left " << lrDecayMaxTries << "\n";
                }else{
                    std::cout << "stopping early at epoch " << epoch << ", best val loss = " << bestValLoss << " at epoch " << bestEpoch << std::endl;
                    return {bestValLoss, bestEpoch, epoch};
                }
            }
        }

        // (TODO CAN BE REMOVED LATER)
        std::cout << "Epoch " << epoch
            << " | train: " << std::fixed << std::setprecision(6) << avgTrainingLoss
            << " | val: "   << std::fixed << std::setprecision(6) << avgValidationLoss
            << " | MAE: "   << std::fixed << std::setprecision(6) << mae
            << " | RMSE: "  << std::fixed << std::setprecision(6) << rmse
            << " | DA: "    << std::fixed << std::setprecision(3) << da
            << " | best val: " << std::fixed << std::setprecision(6)
            << bestValLoss << " (ep " << bestEpoch << ")"
            << std::endl;

        static bool wroteHeader = false;
        static std::ofstream csv("tests/results.csv", std::ios::app);

        if (csv && !wroteHeader) {
            csv << "epoch,train_loss,val_loss,mae,rmse,da,ada_lr,dense_lr\n";
            wroteHeader = true;
        }

        double adaLR   = optimiser.getLearningRate();
        double denseLR = learningRate;

        if (csv) {
            csv << epoch << ","
                << avgTrainingLoss << ","
                << avgValidationLoss << ","
                << mae << ","
                << rmse << ","
                << da << ","
                << adaLR << ","
                << denseLR << "\n";
            csv.flush();
        }
    }

    return {bestValLoss, bestEpoch, epochs};
}