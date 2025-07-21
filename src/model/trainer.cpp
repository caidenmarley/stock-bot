#include "model/trainer.h"

Trainer::Trainer(int numFeatures, int hiddenSize, int sequenceLength, int batchSize, double learningRate, double delta,
const std::vector<PriceData>& rawTrainingData, const std::vector<PriceData>& rawValidationData): 
    lstm(numFeatures, hiddenSize, sequenceLength), outputLayer(hiddenSize), 
    optimiser(lstm.getParameterCount(), learningRate),  huberLoss(delta),
    sequenceLength(sequenceLength), batchSize(batchSize), learningRate(learningRate), 
    trainingData(rawTrainingData, sequenceLength, batchSize), validationData(rawValidationData, sequenceLength, batchSize){}

void Trainer::run(int epochs){
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
                }
            }
        }
    }
}