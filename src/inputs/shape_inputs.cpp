#include "inputs/shape_inputs.h"
#include <algorithm>
#include <stdexcept>

StockData::StockData(const std::vector<PriceData>& rawData, size_t numFeatures, size_t sequenceLength, size_t batchSize, size_t windowSize)
        : sequenceLength(sequenceLength),
        numFeatures(numFeatures),
        batchSize(batchSize),
        positionIndex(0),
        scaler(windowSize, numFeatures) {
    // Number of days available
    const size_t numDays = rawData.size();
    if (numDays < this->sequenceLength + 1) { // You need at least 1 day to compare with models predictions
        throw std::runtime_error("Not enough data: need at least sequenceLength+1 days");
    }

    // how many sliding window sequences can be made, leaving the final value as the target
    this->numSequences = numDays - this->sequenceLength;

    // build a [numDays x numFeatures] feature matrix to contain data for each day
    // Dynamic = row count determined at runtime
    // RowMajor = rows laid contiguously in memory (row by row)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> featureMatrix(numDays, numFeatures);

    for (size_t day = 0; day < numDays; ++day) {
        // rolling window scaling per day
        this->scaler.add(rawData[day]);
        std::vector<double> scaled = this->scaler.scaledValuesPerDay();
        double* ptr = featureMatrix.data() + day * this->numFeatures;
        for(size_t i = 0; i < this->numFeatures; i++){
            ptr[i] = scaled[i];
        }
    }

    // inputs: shape [numSequences][sequenceLength][numFeatures]
    inputs = Eigen::Tensor<double, 3, Eigen::RowMajor>(
        // Eigen expects <long>
        static_cast<long>(this->numSequences),
        static_cast<long>(this->sequenceLength),
        static_cast<long>(this->numFeatures)
    );

    // targets: shape [numSequences], values to compare predictions with
    targets = Eigen::VectorXd(static_cast<long>(this->numSequences));

    // for each sequence, each block of data
    for (size_t seq = 0; seq < this->numSequences; ++seq) {
        // for each time step (day) for each sequence
        for (size_t t = 0; t < this->sequenceLength; ++t) {
            // puts the dates features into the corresponding days row in the inputs matrix
            for (size_t f = 0; f < this->numFeatures; ++f) {
                // gets value to set
                inputs(
                    static_cast<long>(seq),
                    static_cast<long>(t),
                    static_cast<long>(f)
                ) = featureMatrix(static_cast<long>(seq + t), static_cast<long>(f)); // sets based of feature matrix
            }
        }
        // Compute the target: next-day return = (close[t+1] - close[t]) / close[t]
        size_t lastDay = seq + this->sequenceLength - 1;      // last day in window
        double closeT = rawData[lastDay].close;       // close on last day
        double closeTp1 = rawData[lastDay + 1].close; // close on day after last day

        // this is the % next day return value the model will try to predict
        targets(static_cast<long>(seq)) = (closeTp1 - closeT) / closeT;
    }
}

std::pair<Eigen::Tensor<double, 3, Eigen::RowMajor>, Eigen::VectorXd> StockData::nextBatch() {
    if (!hasAnotherBatch()) {
        throw std::out_of_range("No more batches; call reset() to start a new epoch");
    }

    // batch range
    const size_t start = positionIndex; // where in main matrix does batch begin
    const size_t end = std::min(positionIndex + batchSize, numSequences);
    const size_t currentBatch = end - start;

    // ptr to inputs.data() at start of sequence
    double* dataPtr = inputs.data() + start * sequenceLength * numFeatures;

    // treat existing memory as tensor without copying
    Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> inputBatch(
        dataPtr,
        static_cast<long>(currentBatch),
        static_cast<long>(this->sequenceLength),
        static_cast<long>(this->numFeatures)
    );

    // treat existing memory as vector without copying
    Eigen::VectorBlock<Eigen::VectorXd> targetBatch = targets.segment(static_cast<long>(start), static_cast<long>(currentBatch));

    // move index to start of next batch as current batch spans from start->end-1
    positionIndex = end;
    return {inputBatch, targetBatch};
}