#include "parser/parser.h"

// Constructor: Loads and parses the CSV file while timing the operation.
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
        std::cerr << "Failed to open file: " << filename << "\n";
        std::exit(1);
    }

    std::streamsize size = file.tellg(); // get file size (pointer is at end)
    file.seekg(0, std::ios::beg);        // reset to beginning

    // allocate buffer with file size
    buffer.resize(size + 1); // +1 for null terminator.

    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read file.\n";
        std::exit(1);
    }

    buffer[size] = '\0'; // null term

    file.close();
}

// parse csv buffer into PriceData
void CSVLoader::parseBuffer() {
    char* ptr = buffer.data();                 // ptr to start of buffer
    char* end = buffer.data() + buffer.size(); // ptr to end of buffer

    // skip header
    while (ptr < end && *ptr != '\n') {
        ++ptr;
    }

    // move ahead of \n
    if (ptr < end) {
        ++ptr;
    }

    // preallocate large chunk to avoid repeated reallocations
    data.reserve(DATASET_SIZE); // change if you know expected row count.

    while (ptr < end) {
        PriceData row {};

        // parse date
        char* fieldStart = ptr;

        // moves ptr forward until it finds comma or newline
        while (*ptr != ',' && *ptr != '\n' && ptr < end) {
            ++ptr;
        }
        // creates string view of the date
        row.date = std::string_view(fieldStart, ptr - fieldStart);

        // inline parsing macro to avoid repetitive code and function calls on stack
#define NEXT_FIELD_DOUBLE(dest)                                       \
    if (*ptr == ',') ++ptr;                                           \
    {                                                                 \
        char* fieldStart = ptr;                                       \
        while (*ptr != ',' && *ptr != '\n' && ptr < end)              \
            ++ptr;                                                    \
        /* conv string into num type without copying or allocating */ \
        auto res = std::from_chars(fieldStart, ptr, dest);            \
        if (res.ec != std::errc()) {                                  \
            dest = 0.0; /* fallback value if error */                 \
        }                                                             \
    }

#define NEXT_FIELD_UINT(dest)                              \
    if (*ptr == ',') ++ptr;                                \
    {                                                      \
        char* fieldStart = ptr;                            \
        while (*ptr != ',' && *ptr != '\n' && ptr < end)   \
            ++ptr;                                         \
        auto res = std::from_chars(fieldStart, ptr, dest); \
        if (res.ec != std::errc()) {                       \
            dest = 0; /* fallback value if error */        \
        }                                                  \
    }

        NEXT_FIELD_DOUBLE(row.open);
        NEXT_FIELD_DOUBLE(row.high);
        NEXT_FIELD_DOUBLE(row.low);
        NEXT_FIELD_DOUBLE(row.close);
        NEXT_FIELD_DOUBLE(row.adjClose);
        NEXT_FIELD_UINT(row.volume);

        // remove macro definitions
#undef NEXT_FIELD_DOUBLE
#undef NEXT_FIELD_UINT

        // move to next line
        while (*ptr != '\n' && ptr < end) {
            ++ptr;
        }

        // move past newline
        if (ptr < end) {
            ++ptr;
        }

        data.push_back(row);
    }

    data.shrink_to_fit(); // reclaim any over used memory
}
