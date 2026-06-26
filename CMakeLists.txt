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
# create parser_test exe, cmake --build build --target parser_test
add_executable(parser_test tests/parser_test.cpp ${SRC_FILES})
# create rolling_window_scaler_test exe, cmake --build build --target rolling_window_scaler_test
add_executable(rolling_window_scaler_test tests/rolling_window_scaler_test.cpp ${SRC_FILES})
# create stock_data_test exe, cmake --build build --target stock_data_test
add_executable(stock_data_test tests/stock_data_test.cpp ${SRC_FILES})
# create loss_metrics_test exe, cmake --build build --target loss_metrics_test
add_executable(loss_metrics_test tests/loss_metrics_test.cpp ${SRC_FILES})
# create dense_gradient_test exe, cmake --build build --target dense_gradient_test
add_executable(dense_gradient_test tests/dense_gradient_test.cpp ${SRC_FILES})
