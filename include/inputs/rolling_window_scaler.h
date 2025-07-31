#pragma once

#include <vector>
#include <stddef.h>
#include <deque>
#include "inputs/parser.h"

/**
 * Class which scales each input feature based on the mean and standard deviation of the
 * past "windowSize" days, processes one day at a time so for the days less than window size
 * just uses all avaliable previous days mean and stddev
 */
class RollingWindowScaler{
public:
    RollingWindowScaler(size_t windowSize, size_t numFeatures);
    /**
     * adds raw value from PriceData struct to rawValues vector keeping a running tally of the sum
     * and the sum of squares to calc stddev and mean. Removes value from front of the queue if the
     * size of the queue is bigger than "windowSize"
     * 
     * @param data the raw PriceData struct from parser
     */
    void add(const PriceData& data);
    /**
     * Returns the scaled values for the most recent day added to the queue (back of the rawValues queue)
     * scaled through scaled = feature - mean / stdDev, normalising the data based on the data seen so far
     * 
     * @return vector containing all the scaled features for the most recently added day
     */
    std::vector<double> scaledValuesPerDay() const;
    /**
     * Clears the rawValues vector and its queues, and sets the sum tallys to 0
     */
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