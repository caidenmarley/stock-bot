#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>
#include "inputs/parser.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "inputs/rolling_window_scaler.h"

// prepares raw PriceData into tensors for LSTM training
class StockData {
  public:
    // builds the full dataset:
    // - rawData: full PriceData from parser (length = numDays)
    // - numTimesteps: how many days each sequence covers
    // - batchSize: how many sequences per training batch
    // - feeds "batchSize" sequences each with "numTimesteps" days into model
    StockData(const std::vector<PriceData>& rawData, size_t numTimestepsInp, size_t batchSizeInp, size_t windowSize)
        : numTimesteps(numTimestepsInp),
          numFeatures(6),
          batchSize(batchSizeInp),
          positionIndex(0),
          scaler(windowSize, numFeatures) {
        // Number of days available
        const size_t numDays = rawData.size();
        if (numDays < numTimesteps + 1) { // You need at least 1 day to compare with models predictions
            throw std::runtime_error("Not enough data: need at least numTimesteps+1 days");
        }

        // how many sliding window sequences can be made, leaving the final value as the target
        numSequences = numDays - numTimesteps;

        // build a [numDays x numFeatures] feature matrix for each day
        // Dynamic = row count determined at runtime
        // RowMajor = rows laid contiguously in memory (row by row)
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> feature_matrix(numDays, numFeatures);

        for (size_t day = 0; day < numDays; ++day) {
            // rolling window scaling per day
            scaler.add(rawData[day]);
            std::vector<double> scaled = scaler.scaledValuesPerDay();
            double* ptr = feature_matrix.data() + day * numFeatures;
            for(size_t i = 0; i < numFeatures; i++){
                ptr[i] = scaled[i];
            }
        }

        // inputs: shape [numSequences][numTimesteps][numFeatures]
        inputs = Eigen::Tensor<double, 3, Eigen::RowMajor>(
            // Eigen expects <long>
            static_cast<long>(numSequences),
            static_cast<long>(numTimesteps),
            static_cast<long>(numFeatures)
        );

        // targets: shape [numSequences], values to compare predictions with
        targets = Eigen::VectorXd(static_cast<long>(numSequences));

        // for each sequence, each block of data
        for (size_t seq = 0; seq < numSequences; ++seq) {
            // for each time step (day) for each sequence
            for (size_t t = 0; t < numTimesteps; ++t) {
                // puts the dates features into the corresponding days row in the inputs matrix
                for (size_t f = 0; f < numFeatures; ++f) {
                    // gets value to set
                    inputs(
                        static_cast<long>(seq),
                        static_cast<long>(t),
                        static_cast<long>(f)
                    ) = feature_matrix(static_cast<long>(seq + t), static_cast<long>(f)); // sets based of feature matrix
                }
            }
            // Compute the target: next-day return = (close[t+1] - close[t]) / close[t]
            size_t lastDay = seq + numTimesteps - 1;      // last day in window
            double closeT = rawData[lastDay].close;       // close on last day
            double closeTp1 = rawData[lastDay + 1].close; // close on day after last day

            // this is the % next day return value the model will try to predict
            targets(static_cast<long>(seq)) = (closeTp1 - closeT) / closeT;
        }
    }

    // true if there are more batches to fetch
    bool hasAnotherBatch() const {
        return positionIndex < numSequences;
    }

    // Returns the next batch as zero-copy tensor slices:
    // - first: [currentBatch, numTimesteps, numFeatures], inputs
    // - second: [currentBatch], targets with length currentBatch
    std::pair<Eigen::Tensor<double, 3, Eigen::RowMajor>, Eigen::VectorXd> nextBatch() {
        if (!hasAnotherBatch()) {
            throw std::out_of_range("No more batches; call reset() to start a new epoch");
        }

        // batch range
        const size_t start = positionIndex; // where in main matrix does batch begin
        const size_t end = std::min(positionIndex + batchSize, numSequences);
        const size_t currentBatch = end - start;

        // ptr to inputs.data() at start of sequence
        double* dataPtr = inputs.data() + start * numTimesteps * numFeatures;

        // treat existing memory as tensor without copying
        Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> inputBatch(
            dataPtr,
            static_cast<long>(currentBatch),
            static_cast<long>(numTimesteps),
            static_cast<long>(numFeatures)
        );

        // treat existing memory as vector without copying
        Eigen::VectorBlock<Eigen::VectorXd> targetBatch = targets.segment(static_cast<long>(start), static_cast<long>(currentBatch));

        // move index to start of next batch as current batch spans from start->end-1
        positionIndex = end;
        return {inputBatch, targetBatch};
    }

    // reset the postion index so batches start from the beginning again
    void reset() {
        positionIndex = 0;
    }

    size_t getNumFeatures() const{
        return this->numFeatures;
    }

  private:
    size_t numTimesteps;  // days per sequence window
    size_t numFeatures;   // feature count per day (6)
    size_t batchSize;     // sequences per batch
    size_t numSequences;  // total sliding window sequences
    size_t positionIndex; // how many sequences have been served

    RollingWindowScaler scaler;
    Eigen::Tensor<double, 3, Eigen::RowMajor> inputs; // [numSequences][numTimesteps][numFeatures]
    Eigen::VectorXd targets;                          // [numSequences]
};
