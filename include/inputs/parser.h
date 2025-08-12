#pragma once

#include <charconv>
#include <stdexcept>
#include <string_view>
#include <vector>

#define DATASET_SIZE 1000000

/**
 * Struct containing all the raw price data parsed from the csv
 */
struct PriceData {
    double open, high, low, close, adjClose; // Columns on csv
    uint64_t volume;                         // larger volume
    std::string_view date; // avoids copying or allocating new strings per row
};

/**
 * Class for loading and parsing csv files into PriceData vector
 */
class CSVLoader {
  public:
    /**
     * Loads and parses the csv into a vector of PriceData structs
     * 
     * @param filename path to the csv file to be parsed
     */
    CSVLoader(const std::string& filename);

    /**
     * Gets the PriceData vector containing all the data from the csv
     * 
     * @return PriceData vector
     */
    const std::vector<PriceData>& getData() const;

  private:
    std::vector<char> buffer;    // Raw file data buffer
    std::vector<PriceData> data; // Parsed data rows

    /**
     * Reads the file from the path and parses binary into the char buffer
     * 
     * @param filename path to the csv file to be parsed
     */
    void readFile(const std::string& filename);
    /**
     * Parses the raw binary buffer into the PriceData vector
     */
    void parseBuffer();         
    
    /**
     * Inline function to scan until the next comma or newline then calls from_chars to convert
     * that substring into "dest" without any extra allocations
     * 
     * @tparam T any type that from_cahrs can convert to, (e.g. double)
     * @param ptr reference to the ptr to the start of the substring being converted
     * @param end ptr to one past the end of the substring being converted (comma or newline)
     * @param dest Reference to the variable that recieves the parsed value
     */
    template <typename T>
    static inline void parseNext(char*& ptr, char* end, T& dest) {
        if (ptr >= end) {
            throw std::runtime_error("Unexpected end of buffer");
        }
        if (*ptr == ',') ++ptr;
        char* start = ptr;
        while (ptr < end && *ptr != ',' && *ptr != '\n' && *ptr != '\r'){
            ++ptr;
        } 
        // conv string into num type without copying or allocating
        auto res = std::from_chars(start, ptr, dest);
        if (res.ec != std::errc()) {
            dest = T {}; // fallback zero
        }
    }
};
