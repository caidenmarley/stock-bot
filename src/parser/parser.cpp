#include "inputs/parser.h"
#include <chrono>
#include <fstream>
#include <iostream>

// Loads and parses the CSV file while timing
CSVLoader::CSVLoader(const std::string& filename) {
    auto start = std::chrono::high_resolution_clock::now();

    readFile(filename); // load file to memory
    parseBuffer();      // parse the buffer into structs

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Parsed " << data.size() << " rows in " << elapsed.count() << " seconds.\n";
}

// getter for parsed data
const std::vector<PriceData>& CSVLoader::getData() const {
    return data;
}

// reads file into buffer
void CSVLoader::readFile(const std::string& filename) {
    // open file in binary with the pointer straight to the end of the file
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streamsize size = file.tellg(); // get file size (pointer is at end)
    file.seekg(0, std::ios::beg);        // reset to beginning

    buffer.resize(size); // allocate buffer with file size

    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }

    file.close();
}

// parse csv buffer into PriceData
void CSVLoader::parseBuffer() {
    char* ptr = buffer.data();                 // ptr to start of buffer
    char* end = buffer.data() + buffer.size(); // ptr to end of buffer


    /*THIS COULD BE DONE WITH std::memchr or std::find*/
    // skip header
    while (ptr < end && *ptr != '\n') {
        ++ptr;
    }

    // move ahead of \n
    if (ptr < end) {
        ++ptr;
    }

    // preallocate large chunk to avoid repeated reallocations
    data.reserve(DATASET_SIZE); // TODO change if you know expected row count.

    while (ptr < end) {
        PriceData row {};

        // parse date
        char* fieldStart = ptr;

        // moves ptr forward until it finds comma or newline
        while (ptr < end && *ptr != ',' && *ptr != '\n') {
            ++ptr;
        }
        // creates string view of the date
        row.date = std::string_view(fieldStart, ptr - fieldStart);

        parseNext(ptr, end, row.open);
        parseNext(ptr, end, row.high);
        parseNext(ptr, end, row.low);
        parseNext(ptr, end, row.close);
        parseNext(ptr, end, row.adjClose);
        parseNext(ptr, end, row.volume);

        // move to next line
        while (ptr < end && *ptr != '\n') {
            ++ptr;
        }

        // move past newline
        if (ptr < end) {
            ++ptr;
        }

        data.push_back(row);
    }

    // TODO not sure if needed
    data.shrink_to_fit(); // reclaim any over used memory
}
