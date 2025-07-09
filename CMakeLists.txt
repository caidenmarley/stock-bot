cmake_minimum_required(VERSION 3.20)
project(stock_bot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable warnings
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Include directories
include_directories(include)

# Gather source files
file(GLOB_RECURSE SRC_FILES
    src/*.cpp
)

file(GLOB_RECURSE TEST_FILES
    tests/*.cpp
)

# Create main executable
add_executable(stock_bot main.cpp ${SRC_FILES})

# Link libraries if needed in future (e.g., CURL for scraping, SQLite for caching, etc.)
# find_package(CURL REQUIRED)
# target_link_libraries(stock_predictor PRIVATE CURL::libcurl)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(EIGEN_USE_GPU OFF)  # if cpu only
include_directories(${EIGEN3_INCLUDE_DIR})

