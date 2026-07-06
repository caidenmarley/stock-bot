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
cmake --build . --target benchmark_metrics_test
cmake --build . --target dense_gradient_test
cmake --build . --target dense_seed_test
cmake --build . --target lstm_parameter_order_test
cmake --build . --target lstm_gradient_test
cmake --build . --target time_series_validation_test
cmake --build . --target integration_validation_test
cmake --build . --target end_to_end_determinism_test
cmake --build . --target results_file_hygiene_test
cmake --build . --target adabelief_test
cmake --build . --target trainer_behavior_test
cmake --build . --target cli_smoke_test
cmake --build . --target cli_invalid_args_test
cmake --build . --target hyperparam_search_test
cmake --build . --target seed_plumbing_test
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
./build/benchmark_metrics_test
./build/dense_gradient_test
./build/dense_seed_test
./build/lstm_parameter_order_test
./build/lstm_gradient_test
./build/time_series_validation_test
./build/integration_validation_test
./build/end_to_end_determinism_test
./build/results_file_hygiene_test
./build/adabelief_test
./build/trainer_behavior_test
./build/cli_smoke_test
./build/cli_invalid_args_test
./build/hyperparam_search_test
./build/seed_plumbing_test

# safe short stock_bot CLI smoke runs
./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1 --no-results
./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1 --results-file build/results/cli_smoke_results.csv

# feature-count ablation modes
./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1 --feature-count 6 --no-results
./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1 --feature-count 13 --no-results
./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1 --feature-ablation --ablation-report build/test_outputs/feature_ablation_v1.csv --no-results

# explicit dataset selection
./build/stock_bot --data-path data/AAAU.csv --epochs 1 --seed 0 --early-stop-patience 1 --feature-count 13 --no-results
./build/stock_bot --data-dir data --epochs 1 --seed 0 --early-stop-patience 1 --feature-count 13 --no-results
./build/stock_bot --feature-ablation --data-dir data --ablation-report build/results/multi_ticker_feature_ablation.csv --epochs 1 --seed 0 --early-stop-patience 1 --no-results

# invalid flags/values should fail non-zero
./build/stock_bot --definitely-invalid-option
```

Run the full suite through CTest (recommended):

```bash
cd /home/caidenmarley/stock-bot/build
ctest --output-on-failure

# or from repository root
ctest --test-dir build --output-on-failure
```

## Generated Results Files

Trainer-generated metrics output is now explicit and configurable:
- `Trainer` takes an optional `resultsFilePath`.
- Empty path disables trainer CSV output.
- `main.cpp` defaults to `build/results/trainer_results.csv`.
- `main.cpp` options:
	- `--results-file PATH` to select an explicit output file
	- `--no-results` to disable trainer CSV output

Hyperparameter-search output now defaults to:
- `build/results/hyperparam_search_results.csv`

You can call `gridSearch(...)` with an explicit output path to keep results build-local (for example under `build/test_outputs/`).

`randomSearch(...)` is currently declared but intentionally not implemented; it throws a clear `std::logic_error` when called.

These are generated artifacts and should not usually be committed as source changes.
Use build-local paths (for example under `build/results/` or `build/test_outputs/`) to avoid dirty source-tree files during tests and audits.

For `--data-dir` runs, CSV files are discovered deterministically (sorted by filename).
If a file is malformed or too short for the configured sequence/fold setup, the run prints a warning and continues with the remaining files.
