# BUILD.md

## Dependencies

Required tooling:
- CMake >= 3.20
- C++20 compiler (project has been built with g++ 11.4.0)
- Eigen3 >= 3.3 (project has used Eigen3 3.4.0)

## Configure

From repository root:

```bash
cd /home/caidenmarley/stock-bot/build
cmake ..
```

## Build

Main executable:

```bash
cd /home/caidenmarley/stock-bot/build
cmake --build . --target stock_bot
```

Core test executables currently used in recovery milestones:

```bash
cd /home/caidenmarley/stock-bot/build
cmake --build . --target testbed
cmake --build . --target parser_test
cmake --build . --target rolling_window_scaler_test
cmake --build . --target stock_data_test
cmake --build . --target loss_metrics_test
cmake --build . --target dense_gradient_test
cmake --build . --target lstm_parameter_order_test
cmake --build . --target lstm_gradient_test
cmake --build . --target time_series_validation_test
```

Build and run all registered tests in one command:

```bash
cd /home/caidenmarley/stock-bot/build
cmake --build . --target run_tests
```

## Run

From repository root:

```bash
cd /home/caidenmarley/stock-bot
./build/stock_bot
./build/testbed
./build/parser_test
./build/rolling_window_scaler_test
./build/stock_data_test
./build/loss_metrics_test
./build/dense_gradient_test
./build/lstm_parameter_order_test
./build/lstm_gradient_test
./build/time_series_validation_test
```

Run the full suite through CTest (recommended):

```bash
cd /home/caidenmarley/stock-bot/build
ctest --output-on-failure

# or from repository root
ctest --test-dir build --output-on-failure
```

## Generated Results Files

The training flow writes generated output to:
- `tests/results.csv` (written/appended by trainer flow, truncated in main startup path)

There is also hyperparameter-search output logic for:
- `data/results.csv`

These are generated artifacts and should not usually be committed as source changes.
