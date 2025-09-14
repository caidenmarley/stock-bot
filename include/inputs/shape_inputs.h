#pragma once

#include <vector>
#include "inputs/parser.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "inputs/rolling_window_scaler.h"

/**
 * Class that converts raw PriceData structs into tensors to be used in the model
 */
class StockData {
  public:
    /**
     * Converts the raw PriceData into feature matrix of size [numDays x numFeatures] for each day,
     * adding the scaled values to this matrix. Then splits this matrix into inputs [numSequences][sequenceLength][numFeatures]
     * and targets [numSequences], inputs being the values fed into the model and targets being the values the model is trying
     * to predict based on close prices. 
     * 
     * @param rawData Reference to the PriceData vector containing the rawData to be parsed into the tensors
     * @param numFeatures number of features per day
     * @param sequenceLength number of days each sequence contains
     * @param batchSize number of sequences per batch
     * @param windowSize size of the window used for the scaler
     */
    StockData(const std::vector<PriceData>& rawData, size_t numFeatures, size_t sequenceLength, size_t batchSize, size_t windowSize);

    // true if there are more batches to fetch
    bool hasAnotherBatch() const {
        return positionIndex < numSequences;
    }

    /**
     * Returns the next batch as a pair, zero copy tensor slices, 
     * splitting into inputs [currentBatch, sequenceLength, numFeatures], and targets [currentBatch]
     * 
     * @return std::pair, input batch and target batch
     */
    std::pair<
      Eigen::TensorMap< const Eigen::Tensor<double, 3, Eigen::RowMajor>>, 
      Eigen::Map<const Eigen::VectorXd>
    > nextBatch();

    // reset the postion index so batches start from the beginning again
    void reset() {
        positionIndex = 0;
    }

    size_t getNumFeatures() const{
        return this->numFeatures;
    }

  private:
    size_t sequenceLength;  // days per sequence window
    size_t numFeatures;   // feature count per day (6)
    size_t batchSize;     // sequences per batch
    size_t numSequences;  // total sliding window sequences
    size_t positionIndex; // how many sequences have been served

    RollingWindowScaler scaler;
    Eigen::Tensor<double, 3, Eigen::RowMajor> inputs; // [numSequences][sequenceLength][numFeatures]
    Eigen::VectorXd targets;                          // [numSequences]
};
