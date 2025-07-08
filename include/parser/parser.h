#pragma once

#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#define DATASET_SIZE 1000000

// one row of price data from the csv
struct PriceData {
    double open, high, low, close, adjClose; // Columns on csv
    uint64_t volume;                         // larger volume
    // store date as string_view into the buffer.
    std::string_view date; // avoids copying or allocating new strings per row
};

// class for loading and parsing csv files into PriceData vector
class CSVLoader {
  public:
    CSVLoader(const std::string& filename);
    const std::vector<PriceData>& getData() const;

  private:
    std::vector<char> buffer;    // Raw file data buffer
    std::vector<PriceData> data; // Parsed data rows

    void readFile(const std::string& filename); // load file to memory
    void parseBuffer();                         // // parse the buffer into structs
};
