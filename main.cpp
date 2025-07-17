#include "parser/parser.h"
#include "parser/shape_inputs.h"
#include "model/lstm.h"
#include "model/ada_belief.h"
#include "model/huber_loss_function.h"
#include "model/dense.h"
#include <iostream>

int main() {
    try {
        // --- Hyperparameters ---
        const std::string csvPath    = "data/AAAU.csv";
        const int sequenceLength     = 40;    // timesteps per sequence
        const int batchSize          = 20;    // sequences per batch
        const int hiddenSize         = 64;    // LSTM hidden layer size
        const int epochs             = 10;    // number of full passes
        const double learningRate    = 1e-3;  // optimizer learning rate
        const double huberDelta      = 1.0;   // Huber loss threshold

        // --- Load and Shape Data ---
        CSVLoader loader(csvPath);
        const auto& rawData = loader.getData();

        StockData stockData(rawData, sequenceLength, batchSize);
        int numFeatures = static_cast<int>(6);
        std::cout << "Data Loaded" << std::endl;
        // --- Model, Optimizer & Loss ---
        LSTMCell lstm(numFeatures, hiddenSize, sequenceLength);
        AdaBelief optimizer(lstm.getParameterCount(), learningRate);
        HuberLossFunction huber(huberDelta);
        Dense outLayer(hiddenSize);

        // --- Training Loop ---
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            stockData.reset();
            double epochLoss = 0.0;
            size_t batchCount = 0;

            while (stockData.hasAnotherBatch()) {
                auto [inputBatch, targetBatch] = stockData.nextBatch();
                int currentBatch = static_cast<int>(targetBatch.size());
                
                lstm.zeroGrad();
                outLayer.zeroGrad();

                // Process each sequence independently
                for (int i = 0; i < currentBatch; ++i) {
                    // Reset LSTM internal state for this sequence
                    lstm.reset();

                    Eigen::VectorXd h;
                    for (int t = 0; t < sequenceLength; ++t) {
                        // Map features for sequence i at timestep t
                        double* ptr = inputBatch.data() + ((i * sequenceLength + t) * numFeatures);
                        Eigen::Map<Eigen::VectorXd> x_t(ptr, numFeatures);
                        h = lstm.forwardPass(x_t);
                    }
                    double yPred = outLayer.forward(h);
                    

                    Eigen::VectorXd singleY(1), singleT(1);
                    singleY(0) = yPred;
                    singleT(0) = targetBatch(i);
                    double loss = huber.forward(singleY, singleT);
                    Eigen::VectorXd gradVec = huber.backward();
                    double dLdy = gradVec(0);
                    epochLoss += loss;

                    outLayer.backward(h, dLdy);

                    Eigen::VectorXd dh_next = outLayer.W.transpose() * dLdy;
                    Eigen::VectorXd dc_next = Eigen::VectorXd::Zero(hiddenSize);
                    for (int t = sequenceLength - 1; t >= 0; --t) {
                        lstm.backwardPass(dh_next, dc_next);
                        // after first step, only carry cell‐state gradient
                        dh_next.setZero();
                    }
                }
                //std::cout << "forward & backward done" << std::endl;

                // 6) update LSTM weights via AdaBelief
                {
                    auto params   = lstm.getParametersVector();
                    auto gradsVec = lstm.getGradientsVector();
                    optimizer.update(params, gradsVec);
                    lstm.setParametersVector(params);
                    //std::cout << "adabelief done" << std::endl;
                }

                // 7) update Dense layer with simple SGD
                outLayer.W -= learningRate * outLayer.dW;
                outLayer.b -= learningRate * outLayer.db;

                ++batchCount;
            }

            std::cout << "Epoch " << epoch
                      << " completed, avg loss = " << (epochLoss / batchCount)
                      << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
