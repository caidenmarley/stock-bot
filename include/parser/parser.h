#pragma once

#include <charconv>
#include <string_view>
#include <vector>
#include <stdexcept>

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
    template<typename T>
    static inline void parseNext(char*& ptr, char* end, T& dest) {
        if (ptr >= end){
          throw std::runtime_error("Unexpected end of buffer");
        } 
        if (*ptr == ',') ++ptr;
        char* start = ptr;
        while (ptr < end && *ptr != ',' && *ptr != '\n') ++ptr;
        // conv string into num type without copying or allocating
        auto res = std::from_chars(start, ptr, dest);
        if (res.ec != std::errc()) {
            dest = T{}; // fallback zero
        }
    }
};
