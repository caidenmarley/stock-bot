cmake_minimum_required(VERSION 3.20)
project(stock_bot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# enable warnings
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

include_directories(include)

file(GLOB_RECURSE SRC_FILES
    src/*.cpp
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# create main exe, cmake --build build --target stock_bot
add_executable(stock_bot main.cpp ${SRC_FILES})
# create testbed exe, cmake --build build --target testbed
add_executable(testbed tests/testbed.cpp ${SRC_FILES})
