#pragma once

#include "inputs/parser.h"
#include "inputs/rolling_window_scaler.h"
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * Class that converts raw PriceData structs into tensors to be used in the model
 */
class StockData {
  public:
    /**
     * Converts the raw PriceData into feature matrix of size [numDays x numFeatures] for each day,
     * adding the scaled values to this matrix. Then splits this matrix into inputs [numWindows][sequenceLength][numFeatures]
     * and targets [numWindows], inputs being the values fed into the model and targets being the values the model is trying
     * to predict based on close prices. 
     * 
     * @param rawData Reference to the PriceData vector containing the rawData to be parsed into the tensors
     * @param numFeatures number of features per day
     * @param sequenceLength number of days each sequence contains
     * @param batchSize number of sequences per batch
     * @param windowSize size of the window used for the scaler
     */
    StockData(const std::vector<PriceData>& rawData, 
              int numFeatures, int sequenceLength, 
              int batchSize, RollingWindowScaler preLoadedScaler);

    // true if there are more batches to fetch
    bool hasAnotherBatch() const {
        return positionIndex < numWindows;
    }

    /**
     * Returns the next batch as a pair of zero copy tensor slices, 
     * splitting into inputs [currentBatch, sequenceLength, numFeatures], 
	 * and targets [currentBatch]. Uses the positionIndex variable to take 
	 * a contiguous slice
     * 
     * @return std::pair, input batch and target batch
     */
    std::pair<
        Eigen::TensorMap< const Eigen::Tensor<double, 3, Eigen::RowMajor>>, 
        Eigen::Map<const Eigen::VectorXd>
    > nextBatch();

	/**
	 * Returns a batch in a shuffled sequence order as a pair
	 * splitting into inputs [currentBatch, sequenceLength, numFeatures],
	 * and targets [currentBatch].
	 * 
	 * Intended for use in training only
	 * 
	 * @param order sequence indicies in the shuffled order (size numWindows)
	 * @param batchStart starting index into 'order' for this batch
	 * @param currentBatchSize size of current batch size (normally = batchSize, but last batch could be less than this)
	 * @return std::pair, input batch and target batch
	 */
    std::pair<
      	Eigen::TensorMap<const Eigen::Tensor<double,3,Eigen::RowMajor>>,
      	Eigen::Map<const Eigen::VectorXd>
	> nextBatchShuffled(
		const std::vector<int>& order,
		const int batchStart,
		const int currentBatchSize
	);
    

    // reset the postion index so batches start from the beginning again
    void reset() {
        positionIndex = 0;
    }

    int getNumFeatures() const{return this->numFeatures;}
    int getNumWindows() const{return this->numWindows;}

  private:
    int sequenceLength;  // days per sequence window
    int numFeatures;   // feature count per day (6)
    int batchSize;     // sequences per batch
    int numWindows;  // total sliding window sequences
    int positionIndex; // how many sequences have been served
    Eigen::Index elementsPerWindow; // sequenceLength * numFeatures as input shape is [numWindows][sequenceLength][numFeatures]

    RollingWindowScaler scaler;
    Eigen::Tensor<double, 3, Eigen::RowMajor> inputs; // [numWindows][sequenceLength][numFeatures]
    Eigen::VectorXd targets;                          // [numWindows]

	// space for one shuffled batch (reused every call of nextBatchShuffled)
	Eigen::Tensor<double, 3, Eigen::RowMajor> tempInputs;
	Eigen::VectorXd tempTargets;
};
