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
enable_testing()

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
# create lstm_parameter_order_test exe, cmake --build build --target lstm_parameter_order_test
add_executable(lstm_parameter_order_test tests/lstm_parameter_order_test.cpp ${SRC_FILES})
# create lstm_gradient_test exe, cmake --build build --target lstm_gradient_test
add_executable(lstm_gradient_test tests/lstm_gradient_test.cpp ${SRC_FILES})
# create time_series_validation_test exe, cmake --build build --target time_series_validation_test
add_executable(time_series_validation_test tests/time_series_validation_test.cpp ${SRC_FILES})
# create reproducibility_test exe, cmake --build build --target reproducibility_test
add_executable(reproducibility_test tests/reproducibility_test.cpp ${SRC_FILES})
# create integration_validation_test exe, cmake --build build --target integration_validation_test
add_executable(integration_validation_test tests/integration_validation_test.cpp ${SRC_FILES})
# create end_to_end_determinism_test exe, cmake --build build --target end_to_end_determinism_test
add_executable(end_to_end_determinism_test tests/end_to_end_determinism_test.cpp ${SRC_FILES})
# create results_file_hygiene_test exe, cmake --build build --target results_file_hygiene_test
add_executable(results_file_hygiene_test tests/results_file_hygiene_test.cpp ${SRC_FILES})
# create adabelief_test exe, cmake --build build --target adabelief_test
add_executable(adabelief_test tests/adabelief_test.cpp ${SRC_FILES})
# create trainer_behavior_test exe, cmake --build build --target trainer_behavior_test
add_executable(trainer_behavior_test tests/trainer_behavior_test.cpp ${SRC_FILES})

add_test(NAME testbed COMMAND testbed)
add_test(NAME parser_test COMMAND parser_test)
add_test(NAME rolling_window_scaler_test COMMAND rolling_window_scaler_test)
add_test(NAME stock_data_test COMMAND stock_data_test)
add_test(NAME loss_metrics_test COMMAND loss_metrics_test)
add_test(NAME dense_gradient_test COMMAND dense_gradient_test)
add_test(NAME lstm_parameter_order_test COMMAND lstm_parameter_order_test)
add_test(NAME lstm_gradient_test COMMAND lstm_gradient_test)
add_test(NAME time_series_validation_test COMMAND time_series_validation_test)
add_test(NAME reproducibility_test COMMAND reproducibility_test)
add_test(NAME integration_validation_test COMMAND integration_validation_test)
add_test(NAME end_to_end_determinism_test COMMAND end_to_end_determinism_test)
add_test(NAME results_file_hygiene_test COMMAND results_file_hygiene_test)
add_test(NAME adabelief_test COMMAND adabelief_test)
add_test(NAME trainer_behavior_test COMMAND trainer_behavior_test)

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS
        testbed
        parser_test
        rolling_window_scaler_test
        stock_data_test
        loss_metrics_test
        dense_gradient_test
        lstm_parameter_order_test
        lstm_gradient_test
        time_series_validation_test
        reproducibility_test
        integration_validation_test
        end_to_end_determinism_test
        results_file_hygiene_test
        adabelief_test
        trainer_behavior_test
)
