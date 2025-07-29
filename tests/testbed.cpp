#include "inputs/parser.h"
#include "inputs/rolling_window_scaler.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main(){
    const std::size_t windowSize = 3;
    const std::size_t numFeatures = 6;

    RollingWindowScaler scaler(windowSize, numFeatures);

    std::vector<PriceData> data;
    for (int day = 1; day <= 5; ++day) {
        PriceData p;
        p.open     = day;
        p.high     = day;
        p.low      = day;
        p.close    = day;
        p.adjClose = day;
        p.volume   = static_cast<uint64_t>(day);
        data.push_back(p);
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Day  \t Scaled[0]  (all features identical)\n";
    std::cout << "------------------------------------\n";
    for (std::size_t i = 0; i < data.size(); ++i) {
        scaler.add(data[i]);
        auto scaled = scaler.scaledValuesPerDay();
        std::cout << "Day " << (i+1) << "\t" << std::setw(8) << scaled[0] << "\n";
    }

    scaler.reset();
    std::cout << "\nAfter reset, feeding day=10:\n";
    PriceData p;
    p.open = p.high = p.low = p.close = p.adjClose = 10;
    p.volume = 10;
    scaler.add(p);
    auto scaled = scaler.scaledValuesPerDay();
    std::cout << "Scaled after reset (should be 0): " << scaled[0] << "\n";

}