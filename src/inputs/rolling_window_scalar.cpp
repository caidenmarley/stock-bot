#include "inputs/rolling_window_scaler.h"
#include <cmath>

// TODO could maybe do some sort of ring buffer with contiguous memory for fewer cache misses

RollingWindowScaler::RollingWindowScaler(size_t windowSize, size_t numFeatures)
    : windowSize(windowSize), numFeatures(numFeatures),
    rawValues(numFeatures), sums(numFeatures, 0.0), sumsOfSquares(numFeatures, 0.0){}


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

std::vector<double> RollingWindowScaler::scaledValuesPerDay() const{
    std::vector<double> scaled(numFeatures, 0.0);

    for(size_t i = 0; i < numFeatures; i++){
        size_t n = rawValues[i].size(); 

        if(n == 0){
            // for the first value as there is no data to compare
            scaled[i] = 0.0;
            continue;
        }

        double mean = sums[i] / static_cast<double>(n);
        double meanS = sumsOfSquares[i] / static_cast<double>(n);
        double var = meanS - mean*mean;
        double stdDev;
        if(var > 0.0){
            stdDev = std::sqrt(var);
        }else{
            stdDev = 0.0;
        }

        if(stdDev > 0.0){
            // prevent / 0
            scaled[i] = (rawValues[i].back() - mean) / stdDev;
        }else{
            scaled[i] = 0.0;
        }
    }
    return scaled;
}

void RollingWindowScaler::reset(){
    for(size_t i = 0; i < numFeatures; i++){
        rawValues[i].clear();
        sums[i] = 0.0;
        sumsOfSquares[i] = 0.0;
    }
}

size_t RollingWindowScaler::getWindowSize() const{
    return this->windowSize;
}

size_t RollingWindowScaler::getNumFeatures() const{
    return this->numFeatures;
}

