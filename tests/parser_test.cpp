#include "inputs/parser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <iomanip>

// Test helper: write CSV content to a temporary file
void writeTestFile(const std::string& filename, const std::string& content) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create test file: " + filename);
    }
    file.write(content.c_str(), content.size());
    file.close();
}

// Helper to clean up test files
void cleanupTestFile(const std::string& filename) {
    if (std::remove(filename.c_str()) != 0) {
        std::cerr << "Warning: Failed to delete test file: " << filename << "\n";
    }
}

// Test 1: Header row is skipped
void test_header_skipped() {
    std::cout << "\n=== Test 1: Header Row Skipped ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\n"
                      "2024-01-01,100.0,101.0,99.0,100.5,100.5,1000000\n";
    
    writeTestFile("/tmp/test_header.csv", csv);
    
    try {
        CSVLoader loader("/tmp/test_header.csv");
        const auto& data = loader.getData();
        
        assert(data.size() == 1);
        assert(std::string(data[0].date) == "2024-01-01");
        assert(data[0].open == 100.0);
        std::cout << "✓ PASSED: Header correctly skipped, 1 data row parsed\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_header.csv");
}

// Test 2: Normal rows parse correctly
void test_normal_row_parsing() {
    std::cout << "\n=== Test 2: Normal Row Parsing ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\n"
                      "2024-01-01,100.5,101.5,99.5,100.0,100.0,1500000\n"
                      "2024-01-02,100.0,102.0,99.0,101.0,101.0,2000000\n";
    
    writeTestFile("/tmp/test_normal.csv", csv);
    
    try {
        CSVLoader loader("/tmp/test_normal.csv");
        const auto& data = loader.getData();
        
        assert(data.size() == 2);
        
        // Row 1
        assert(std::string(data[0].date) == "2024-01-01");
        assert(data[0].open == 100.5);
        assert(data[0].high == 101.5);
        assert(data[0].low == 99.5);
        assert(data[0].close == 100.0);
        assert(data[0].adjClose == 100.0);
        assert(data[0].volume == 1500000);
        
        // Row 2
        assert(std::string(data[1].date) == "2024-01-02");
        assert(data[1].close == 101.0);
        assert(data[1].volume == 2000000);
        
        std::cout << "✓ PASSED: 2 rows parsed with correct values\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_normal.csv");
}

// Test 3: Blank lines are handled
void test_blank_lines_skipped() {
    std::cout << "\n=== Test 3: Blank Lines Handled ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\n"
                      "2024-01-01,100.0,101.0,99.0,100.5,100.5,1000000\n"
                      "\n"  // blank line
                      "2024-01-02,100.0,102.0,99.0,101.0,101.0,2000000\n";
    
    writeTestFile("/tmp/test_blank.csv", csv);
    
    try {
        CSVLoader loader("/tmp/test_blank.csv");
        const auto& data = loader.getData();
        
        // Should skip blank line
        assert(data.size() == 2);
        assert(std::string(data[0].date) == "2024-01-01");
        assert(std::string(data[1].date) == "2024-01-02");
        
        std::cout << "✓ PASSED: Blank line correctly skipped, 2 data rows parsed\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_blank.csv");
}

// Test 4: Unix line endings work
void test_unix_line_endings() {
    std::cout << "\n=== Test 4: Unix Line Endings (LF) ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\n"
                      "2024-01-01,100.0,101.0,99.0,100.5,100.5,1000000\n"
                      "2024-01-02,100.0,102.0,99.0,101.0,101.0,2000000\n";
    
    writeTestFile("/tmp/test_unix.csv", csv);
    
    try {
        CSVLoader loader("/tmp/test_unix.csv");
        const auto& data = loader.getData();
        
        assert(data.size() == 2);
        assert(std::string(data[0].date) == "2024-01-01");
        assert(std::string(data[1].date) == "2024-01-02");
        
        std::cout << "✓ PASSED: Unix line endings handled correctly\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_unix.csv");
}

// Test 5: Windows CRLF line endings work
void test_windows_line_endings() {
    std::cout << "\n=== Test 5: Windows Line Endings (CRLF) ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\r\n"
                      "2024-01-01,100.0,101.0,99.0,100.5,100.5,1000000\r\n"
                      "2024-01-02,100.0,102.0,99.0,101.0,101.0,2000000\r\n";
    
    writeTestFile("/tmp/test_crlf.csv", csv);
    
    try {
        CSVLoader loader("/tmp/test_crlf.csv");
        const auto& data = loader.getData();
        
        assert(data.size() == 2);
        assert(std::string(data[0].date) == "2024-01-01");
        assert(std::string(data[1].date) == "2024-01-02");
        
        std::cout << "✓ PASSED: Windows CRLF line endings handled correctly\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_crlf.csv");
}

// Test 6: Large uint64 volume parses correctly
void test_large_volume_parsing() {
    std::cout << "\n=== Test 6: Large uint64 Volume Parsing ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\n"
                      "2024-01-01,100.0,101.0,99.0,100.5,100.5,18446744073709551615\n";  // max uint64
    
    writeTestFile("/tmp/test_volume.csv", csv);
    
    try {
        CSVLoader loader("/tmp/test_volume.csv");
        const auto& data = loader.getData();
        
        assert(data.size() == 1);
        assert(data[0].volume == 18446744073709551615ULL);
        
        std::cout << "✓ PASSED: Large uint64 volume parsed correctly (max uint64)\n";
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_volume.csv");
}

// Test 7: Malformed numeric data throws exception
void test_malformed_data_exception() {
    std::cout << "\n=== Test 7: Malformed Numeric Data Exception ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\n"
                      "2024-01-01,NOT_A_NUMBER,101.0,99.0,100.5,100.5,1000000\n";
    
    writeTestFile("/tmp/test_malformed.csv", csv);
    
    try {
        CSVLoader loader("/tmp/test_malformed.csv");
        std::cout << "✗ FAILED: Expected exception for malformed data, but none was thrown\n";
    } catch (const std::exception& e) {
        std::cout << "✓ PASSED: Exception thrown for malformed data: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_malformed.csv");
}

// Test 8: Date string_view lifetime verification
void test_stringview_lifetime() {
    std::cout << "\n=== Test 8: Date string_view Lifetime Verification ===\n";
    
    std::string csv = "Date,Open,High,Low,Close,AdjClose,Volume\n"
                      "2024-01-01,100.0,101.0,99.0,100.5,100.5,1000000\n"
                      "2024-01-02,100.0,102.0,99.0,101.0,101.0,2000000\n";
    
    writeTestFile("/tmp/test_lifetime.csv", csv);
    
    try {
        std::vector<std::string> captured_dates;
        
        {
            CSVLoader loader("/tmp/test_lifetime.csv");
            const auto& data = loader.getData();
            
            // Capture dates while CSVLoader is alive
            for (const auto& row : data) {
                captured_dates.push_back(std::string(row.date));
            }
            
            assert(captured_dates.size() == 2);
            assert(captured_dates[0] == "2024-01-01");
            assert(captured_dates[1] == "2024-01-02");
            
            std::cout << "✓ PASSED: Date string_view values remain valid while CSVLoader is alive\n";
        } // CSVLoader destroyed here
        
        // Verify captured strings are still valid (they are std::string copies)
        assert(captured_dates[0] == "2024-01-01");
        std::cout << "✓ PASSED: Captured date strings remain valid after CSVLoader destroyed\n";
        
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << "\n";
    }
    cleanupTestFile("/tmp/test_lifetime.csv");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "     PARSER (CSVLoader) TEST SUITE      \n";
    std::cout << "========================================\n";
    
    test_header_skipped();
    test_normal_row_parsing();
    test_blank_lines_skipped();
    test_unix_line_endings();
    test_windows_line_endings();
    test_large_volume_parsing();
    test_malformed_data_exception();
    test_stringview_lifetime();
    
    std::cout << "\n========================================\n";
    std::cout << "     ALL PARSER TESTS COMPLETED        \n";
    std::cout << "========================================\n";
    
    return 0;
}
