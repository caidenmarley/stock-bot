#include "inputs/stock_data.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <cstring>
#include <cassert>

StockData::StockData(const std::vector<PriceData>& rawData, int numFeatures, int sequenceLength, int batchSize, RollingWindowScaler preLoadedScaler)
        : sequenceLength(sequenceLength),
        numFeatures(numFeatures),
        batchSize(batchSize),
        positionIndex(0),
        elementsPerWindow(static_cast<Eigen::Index>(sequenceLength) * static_cast<Eigen::Index>(numFeatures)),
        scaler(std::move(preLoadedScaler)) 
{
    // Initialise temp buffers for batch shuffling
    tempInputs = Eigen::Tensor<double, 3, Eigen::RowMajor>(
        static_cast<Eigen::Index>(batchSize),
        static_cast<Eigen::Index>(sequenceLength),
        static_cast<Eigen::Index>(numFeatures)
    );
    tempTargets = Eigen::VectorXd(static_cast<Eigen::Index>(batchSize));

    // Number of days available
    const int numDays = rawData.size();
    if (numDays < this->sequenceLength + 1) { // You need at least 1 day to compare with models predictions
        throw std::runtime_error("Not enough data: need at least sequenceLength+1 days");
    }

    // how many sliding window sequences can be made, leaving the final value as the target
    this->numWindows = numDays - this->sequenceLength;

    // build a [numDays x numFeatures] feature matrix to contain data for each day
    // Dynamic = row count determined at runtime
    // RowMajor = rows laid contiguously in memory (row by row)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> featureMatrix(numDays, numFeatures);

    for (int day = 0; day < numDays; ++day) {
        // rolling window scaling per day
        this->scaler.add(rawData[day]);
        std::vector<double> scaled = this->scaler.scaledValuesPerDay();
        double* rowPtr = featureMatrix.data() + day * this->numFeatures;
        for(int i = 0; i < this->numFeatures; i++){
            rowPtr[i] = scaled[i];
        }
    }

    // inputs: shape [numWindows][sequenceLength][numFeatures]
    inputs = Eigen::Tensor<double, 3, Eigen::RowMajor>(
        // Eigen::Index is what Eigen uses for indexing
        static_cast<Eigen::Index>(this->numWindows),
        static_cast<Eigen::Index>(this->sequenceLength),
        static_cast<Eigen::Index>(this->numFeatures)
    );

    // targets: shape [numWindows], values to compare predictions with
    targets = Eigen::VectorXd(static_cast<Eigen::Index>(this->numWindows));

    // for each sequence, each block of data
    for (int seq = 0; seq < this->numWindows; ++seq) {
        // for each time step (day) for each sequence
        for (int t = 0; t < this->sequenceLength; ++t) {
            // puts the dates features into the corresponding day's row in the inputs matrix
            for (int f = 0; f < this->numFeatures; ++f) {
                // gets value to set
                inputs(
                    static_cast<Eigen::Index>(seq),
                    static_cast<Eigen::Index>(t),
                    static_cast<Eigen::Index>(f)
                ) = featureMatrix(static_cast<Eigen::Index>(seq + t), static_cast<Eigen::Index>(f)); // sets based of feature matrix
            }
        }
        // Compute the target: next-day return = (close[t+1] - close[t]) / close[t]
        int lastDay = seq + this->sequenceLength - 1;      // last day in window
        double closeT = rawData[lastDay].close;       // close on last day
        double closeTp1 = rawData[lastDay + 1].close; // close on day after last day

        // this is the % next day return value the model will try to predict
        targets(static_cast<Eigen::Index>(seq)) = (closeTp1 - closeT) / closeT;
    }
}

std::pair<
    Eigen::TensorMap< const Eigen::Tensor<double, 3, Eigen::RowMajor>>, 
    Eigen::Map<const Eigen::VectorXd>
>
StockData::nextBatch() {
    if (!hasAnotherBatch()) {
        throw std::out_of_range("No more batches; call reset() to start a new epoch");
    }

    // batch range
    const int start = positionIndex; // where in main matrix does batch begin
    const int end = std::min(positionIndex + batchSize, numWindows);
    const int currentBatch = end - start;

    // ptr to inputs.data() at start of sequence
    double* inputsPtr = inputs.data() + start * elementsPerWindow;

    // treat existing memory as tensor without copying
    Eigen::TensorMap<const Eigen::Tensor<double, 3, Eigen::RowMajor>> inputBatch(
        inputsPtr,
        static_cast<Eigen::Index>(currentBatch),
        static_cast<Eigen::Index>(this->sequenceLength),
        static_cast<Eigen::Index>(this->numFeatures)
    );

    double* targetsPtr = targets.data() + static_cast<Eigen::Index>(start);

    Eigen::Map<const Eigen::VectorXd> targetBatch(targetsPtr, static_cast<Eigen::Index>(currentBatch));

    // move index to start of next batch as current batch spans from start->end-1
    positionIndex = end;
    return {inputBatch, targetBatch};
}

std::pair<
    Eigen::TensorMap<const Eigen::Tensor<double,3,Eigen::RowMajor>>,
    Eigen::Map<const Eigen::VectorXd>
> StockData::nextBatchShuffled(
    const std::vector<int>& order,
    const int batchStart,
    const int currentBatchSize // last batch could be different than batchSize
){
    assert(currentBatchSize > 0);
    assert(batchStart >= 0);
    assert(batchStart + currentBatchSize <= static_cast<int>(order.size()));

    for(int i = 0; i < currentBatchSize; ++i){
        const int inputsIndex = order[batchStart + i];
        // ptr arithmetic with Eigen::Index to ensure arithmetic is done with same type eigen uses
        const double* inputsPtr = inputs.data() + static_cast<Eigen::Index>(inputsIndex) * elementsPerWindow;
        double* tempInputsPtr = tempInputs.data() + static_cast<Eigen::Index>(i) * elementsPerWindow;
    
        // copy inputs at specified index by the vector to the tempInputs buffer
        std::memcpy(tempInputsPtr, inputsPtr, sizeof(double) * static_cast<size_t>(elementsPerWindow));
        tempTargets(static_cast<Eigen::Index>(i)) = targets(static_cast<Eigen::Index>(inputsIndex));
    }

    Eigen::TensorMap<const Eigen::Tensor<double,3,Eigen::RowMajor>> inputsView(
        tempInputs.data(), 
        static_cast<Eigen::Index>(currentBatchSize),
        static_cast<Eigen::Index>(sequenceLength),
        static_cast<Eigen::Index>(numFeatures)
    );

    Eigen::Map<const Eigen::VectorXd> targetsView(
        tempTargets.data(),
        static_cast<Eigen::Index>(currentBatchSize)
    );

    return { inputsView, targetsView };
}