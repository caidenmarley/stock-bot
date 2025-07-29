#pragma once

#include <vector>
#include <stddef.h>
#include <deque>
#include "inputs/parser.h"

// Scales each day based on the last "windowSize" days of history
class RollingWindowScaler{
public:
    RollingWindowScaler(size_t windowSize, size_t numFeatures);
    void add(const PriceData& data); // adds values to rawValues vec
    std::vector<double> scaledValuesPerDay() const; // gets scaled values for the day
    void reset();
    size_t getWindowSize() const;
    size_t getNumFeatures() const;
private:
    size_t windowSize;
    size_t numFeatures;
    std::vector<std::deque<double>> rawValues; // stores up to windowSize values
    // both of size numFeatures
    std::vector<double> sums; // running sum for each feature
    std::vector<double> sumsOfSquares; // running sum of squares for each feature
};