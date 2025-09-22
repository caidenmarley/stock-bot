#include "inputs/parser.h"
#include <chrono>
#include <fstream>
#include <iostream>

// TODO stream based csv parsing remove need for buffer parse just into priceData structs

CSVLoader::CSVLoader(const std::string& filename) {
    auto start = std::chrono::high_resolution_clock::now();

    readFile(filename); // load file to memory
    parseBuffer();      // parse the buffer into structs

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Parsed " << data.size() << " rows in " << elapsed.count() << " seconds.\n";
}

const std::vector<PriceData>& CSVLoader::getData() const {
    return data;
}

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

void CSVLoader::parseBuffer() {
    char* ptr = buffer.data();                 // ptr to start of buffer
    const char* end = buffer.data() + buffer.size(); // ptr to end of buffer


    /*TODO THIS COULD BE DONE WITH std::memchr or std::find*/
    // skip header
    while (ptr < end && *ptr != '\n' && *ptr != '\r') {
        ++ptr;
    }

    // move ahead of \n
    if (ptr < end) {
        char eol = *ptr++;
        if(ptr < end && ((eol == '\r' && *ptr == '\n') || (eol == '\n' && *ptr == '\r'))){
            // protects against \r\n or \n\r
            ptr++;
        }
    }

    // preallocate large chunk to avoid repeated reallocations
    data.reserve(DATASET_SIZE); // TODO change if you know expected row count.

    while (ptr < end) {
        PriceData row {};

        // parse date
        char* fieldStart = ptr;

        // moves ptr forward until it finds comma or newline
        while (ptr < end && *ptr != ',' && *ptr != '\n' && *ptr != '\r') {
            ++ptr;
        }

        // creates string view of the date
        row.date = std::string_view(fieldStart, ptr - fieldStart);

        // if there is a blank line, skip it
        if(row.date.size() == 0){
            if (ptr < end && (*ptr == '\n' || *ptr == '\r')) {
                char eol = *ptr++;
                if (ptr < end && ((eol == '\r' && *ptr == '\n') || (eol == '\n' && *ptr == '\r'))) {
                    ++ptr;
                }
            }
            continue; // try the next line
        }

        parseNext(ptr, end, row.open);
        parseNext(ptr, end, row.high);
        parseNext(ptr, end, row.low);
        parseNext(ptr, end, row.close);
        parseNext(ptr, end, row.adjClose);
        parseNext(ptr, end, row.volume);

        // move to next line
        while (ptr < end && *ptr != '\n' && *ptr != '\r') {
            ++ptr;
        }

        // move past newline
        if (ptr < end) {
            char eol = *ptr++;
            if (ptr < end && ((eol == '\r' && *ptr == '\n') || (eol == '\n' && *ptr == '\r'))) {
                ++ptr;
            }
        }

        data.push_back(row);
    }

    // TODO not sure if needed
    data.shrink_to_fit(); // reclaim any over used memory
}
