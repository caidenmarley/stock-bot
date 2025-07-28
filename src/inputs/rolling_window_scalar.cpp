#include "inputs/rolling_window_scaler.h"

RollingWindowScaler::RollingWindowScaler(size_t windowSize, size_t numFeatures)
    : windowSize(windowSize), numFeatures(numFeatures),
    rawValues(numFeatures), sums(numFeatures, 0.0), sumsOfSquares(numFeatures, 0.0){}

// adds value to values window, removes value from queue if queue is bigger than windowSize
void RollingWindowScaler::add(const PriceData& data){
    std::vector<double> values{
        data.open,
        data.high,
        data.low,
        data.close,
        data.adjClose,
        static_cast<double>(data.volume)
    };

    for(size_t i = 0; i < numFeatures; i++){
        double val = values[i];
        this->rawValues[i].push_back(val);
        this->sums[i] += val;
        this->sumsOfSquares[i] += val*val;

        if(rawValues[i].size() > windowSize){
            double removedVal = this->rawValues[i].front();
            this->rawValues[i].pop_front();
            sums[i] -= removedVal;
            sumsOfSquares[i] -= removedVal*removedVal;
        }
    }
}

// returns the scaled values for the day
std::vector<double> RollingWindowScaler::scaledValuesPerDay() const{
    std::vector<double> scaled(numFeatures, 0.0);
    if(rawValues[0].size() < windowSize){
        // not enought data
        return scaled;  // 0
    }

    for(size_t i = 0; i < numFeatures; i++){
        double stdDev = calcStdDev(i);
        if(stdDev > 0.0){
            // prevent / 0
            scaled[i] = (rawValues[i].back() - calcMean(i)) / stdDev;
        }
    }
    return scaled;
}

