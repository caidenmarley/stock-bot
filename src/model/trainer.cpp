#include "model/trainer.h"
#include <iostream>
#include <iomanip>

Trainer::Trainer(const int numFeatures, const int hiddenSize, const int sequenceLength, const int batchSize, const double learningRate, const double delta,
const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData): 
    lstm(numFeatures, hiddenSize, sequenceLength), outputLayer(hiddenSize), 
    optimiser(lstm.getParameterCount(), learningRate),  huberLoss(delta),
    sequenceLength(sequenceLength), batchSize(batchSize), learningRate(learningRate), 
    trainingData(rawTrainingData, sequenceLength, batchSize), validationData(rawValidationData, sequenceLength, batchSize){}

void Trainer::run(const int epochs){
    for(int epoch = 1; epoch <= epochs; epoch++){
        // TRAINING
        // reset trainingData position index, so batches start from the beginning
        trainingData.reset();
        double trainingLoss = 0.0;  // accumulator for training loss
        int trainingBatchCount = 0; // num batches processed

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
                    double* ptr = inputBatch.data() + ((i*sequenceLength + j) * trainingData.getNumFeatures());
                    Eigen::Map<Eigen::VectorXd> inputs(ptr, trainingData.getNumFeatures());

                    // single lstm step
                    hiddenState = lstm.forwardPass(inputs);
                }

                double targetPrediction = outputLayer.forward(hiddenState);

                double loss = huberLoss.forward(targetPrediction, targetBatch(i));
                Eigen::VectorXd gradient = huberLoss.backward();
                double dLdy = gradient(0);

                trainingLoss += loss;

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

                    // the external dLdhNext is 0 as you dont want the output layer gradient to affect anything other than
                    // the first step of the back pass, dLdh prev is still calculated internally in the backpass method
                    dLdhNext.setZero();
                }
            }
            // scope for optimiser variables
            {
                Eigen::VectorXd params = lstm.getParametersVector();
                Eigen::VectorXd gradients = lstm.getGradientsVector();
                optimiser.update(params, gradients);
                lstm.setParametersVector(params);
            }

            // Dense layer: using SGD on W and b
            outputLayer.W -= learningRate * outputLayer.dW;
            outputLayer.b -= learningRate * outputLayer.db;

            trainingBatchCount++;
        }

        double avgTrainingLoss{};
        if(trainingBatchCount > 0){
            avgTrainingLoss = trainingLoss/trainingBatchCount;
        }else{
            avgTrainingLoss = 0.0;
        }

        // VALIDATION - measures how the model does with the weights and bias it just worked out in training
        validationData.reset();
        double validationLoss{};
        int validationBatchCount{};

        while(validationData.hasAnotherBatch()){
            auto [inputBatch, targetBatch] = validationData.nextBatch();
            int currentBatch = targetBatch.size();

            for(int i = 0; i < currentBatch; i++){
                lstm.reset();   // clear prev states
                Eigen::VectorXd hiddenState;
                for (int j = 0; j < sequenceLength; j++){
                    double* ptr = inputBatch.data() + ((i*sequenceLength + j) * validationData.getNumFeatures());
                    Eigen::Map<Eigen::VectorXd> inputs(ptr, validationData.getNumFeatures());

                    hiddenState = lstm.forwardPass(inputs);
                }

                double targetPrediction = outputLayer.forward(hiddenState);

                validationLoss += huberLoss.forward(targetPrediction, targetBatch(i));
            }
            validationBatchCount++;
        }

        double avgValidationLoss{};
        if(validationBatchCount > 0){
            avgValidationLoss = validationLoss/validationBatchCount;
        }else{
            avgValidationLoss = 0.0;
        }

        std::cout << "Epoch " << epoch
            << " | train loss: " << std::fixed << std::setprecision(6)
            << avgTrainingLoss
            << " | val  loss: " << std::fixed << std::setprecision(6)
            << avgValidationLoss << "\n";
    }
}