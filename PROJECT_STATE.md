# PROJECT_STATE.md

## Overview

This is a C++ stock price prediction system built from first principles for learning ML. The current implementation focuses on a **numerical LSTM-based stock-price model** that predicts next-day returns from historical OHLCV (Open, High, Low, Close, Volume) data.

**Philosophy**: Deep understanding and correctness first. The project avoids black-box ML frameworks—all neural network, loss functions, and optimizer logic are implemented from scratch using Eigen for linear algebra.

---

## Current Training Flow

```
1. CSV Load (data/AAAU.csv)
   └─> CSVLoader::readFile() → parses OHLCV data into PriceData vector

2. Rolling-Window Feature Scaling
   └─> RollingWindowScaler
       ├─ Maintains a deque of up to 256 values per feature (historical window)
       ├─ Computes running mean and stddev for the window
       └─ Returns scaled values for the most recently added day: (day_value - window_mean) / window_stddev

3. Sequence Construction (StockData)
   └─> Converts raw price data into:
       ├─ Input tensors: [numWindows, sequenceLength=15, numFeatures=6]
       ├─ Target vectors: [numWindows] (next-day return labels)
       └─ Batching: [batchSize=10, sequenceLength, numFeatures]

4. Rolling Validation Split (3 folds at 60%, 70%, 80% splits)
   └─> For each fold:
       ├─ Training data: days 0 to split point (batched in shuffled sequence order)
       ├─ Validation data: days split point to end (batched in sequential order)
       └─ Time-series temporal order preserved (no test data shuffling)

5. Training Loop (Trainer::run)
   └─> For each epoch:
       ├─ Trainer processes training batches:
       │   ├─ LSTM forward pass over sequences
       │   ├─ Dense output layer forward (y = Wh + b)
       │   ├─ Huber loss computation
       │   ├─ Backward pass through dense layer
       │   ├─ LSTM backward pass (unrolled BPTT)
       │   ├─ Gradient accumulation and clipping (maxNorm=1.0)
       │   └─ AdaBelief optimizer update
       │
       ├─ Validation:
       │   ├─ Process validation batches (no gradient computation)
       │   ├─ Compute average validation loss
       │   ├─ Track best validation loss and epoch
       │   └─ Early stopping: halt if no improvement for 3 epochs
       │
       └─ Learning rate decay:
           └─ If no improvement detected, halve learning rate (up to 3 times)

6. Output
   └─> For each fold: best validation loss and epoch count
       └─> Summary: average and best fold validation loss
```

---

## Major Modules

### Input Pipeline (`inputs/`)

#### **CSVLoader** (`parser.h/cpp`)
- **Responsibility**: Load and parse OHLCV CSV files into `PriceData` structs
- **Key Design**:
  - Uses a contiguous buffer (`std::vector<char>`) for the entire file
  - Parses using `std::from_chars` (fast, no exceptions)
  - Stores dates as `std::string_view` to avoid allocations per row
  - Skips header row; assumes 6 numeric columns + volume
- **Assumptions**:
  - CSV format: `Date, Open, High, Low, Close, AdjClose, Volume`
  - Line endings: handles both Unix and Windows
  - Numeric parsing: base-10 floats and uint64 for volume

#### **RollingWindowScaler** (`rolling_window_scaler.h/cpp`)
- **Responsibility**: Scale each feature independently using a rolling window
- **Key Design**:
  - Maintains a deque of up to `windowSize` (256) historical values per feature
  - Computes running mean and stddev over the window
  - Handles partial windows at the start (uses available data only)
  - Feature order: open, high, low, close, adjClose, volume
- **Output**: `scaledValue = (current_day_raw - window_mean) / window_stddev`
  - Note: Scales the most recently added day using statistics from the rolling window
  - Acceptable for next-day prediction models but assumptions should be tested for leakage
- **Assumptions**:
  - No future data in the window (rolling window uses only historical data up to current day)
  - Each feature scaled independently (correct for time-series)

#### **StockData** (`stock_data.h/cpp`)
- **Responsibility**: Convert scaled daily data into sequences and batches for LSTM
- **Key Design**:
  - Combines raw data + rolling scaler
  - Creates sliding windows: each window is `sequenceLength` days (15)
  - Target: next-day return, computed from close prices
  - Provides two batching modes:
    - `nextBatch()`: Sequential slices of size `batchSize` (10) – used for validation
    - `nextBatchShuffled()`: Shuffled sequence order – used for training
- **Critical Behavior**:
  - Validation maintains temporal order (time-series integrity)
  - Training shuffles within the dataset (allows model to see patterns in random order)
  - Returns tensor maps (zero-copy views into underlying storage)
  - Batch iteration is deterministic given a seed

---

### Model Architecture (`model/`)

#### **LSTMCell** (`lstm.h/cpp`)
- **Responsibility**: Implement LSTM forward and backward passes
- **Key Design**:
  - Standard LSTM equations:
    - Forget gate: `f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)` (bias init to 1.0)
    - Input gate: `i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)`
    - Cell gate: `C'_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)`
    - Output gate: `o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)`
    - Cell state: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ C'_t`
    - Hidden state: `h_t = o_t ⊙ tanh(C_t)`
  - Xavier weight initialization
  - Unrolled BPTT (backward pass unrolls through entire sequence)
- **Parameter Vector**:
  - All weights and biases flattened into a single vector for optimizer
  - Order matters: must match between `getParametersVector()` and `setParametersVector()`
- **State Management**:
  - Hidden and cell states reset per sequence
  - Gradients accumulated during BPTT

#### **Dense** (`dense.h/cpp`)
- **Responsibility**: Output layer (linear regression for next-day return)
- **Key Design**:
  - Simple: `output = W @ h + b` (1×hidden_size matrix × hidden_size vector)
  - Weights initialized to ±0.01 random
  - Accumulates gradients: `dW += (dLdy) * h^T`, `db += dLdy`
  - Used by HuberLossFunction during backward pass

#### **HuberLossFunction** (`huber_loss_function.h/cpp`)
- **Responsibility**: Robust loss for next-day return prediction
- **Key Design**:
  - Huber loss: quadratic for |error| ≤ delta, linear for |error| > delta
  - Reduces impact of outlier returns
  - Forward: computes loss for batch of predictions vs. targets
  - Backward: computes `dL/dy_pred` (gradient w.r.t. predictions)
- **Parameters**: delta = 1.0 (tunable hyperparameter)

#### **AdaBelief** (`ada_belief.h/cpp`)
- **Responsibility**: Adaptive optimizer for training
- **Key Design**:
  - Similar to Adam but adapts to variance of gradients
  - Maintains exponential moving averages:
    - `m_t` = EMA of gradients
    - `s_t` = EMA of `(g_t - m_t)^2` (belief in step size)
  - Update: `param -= lr * m_t / (sqrt(s_t) + epsilon)`
  - Bias correction for early training steps
- **Parameters**:
  - Learning rate (decayed during training)
  - Beta1 = 0.9, Beta2 = 0.999 (standard defaults)
  - Epsilon = 1e-8

#### **Trainer** (`trainer.h/cpp`)
- **Responsibility**: Orchestrate training and validation loops
- **Key Design**:
  - Manages LSTM, Dense layer, HuberLoss, and AdaBelief
  - Preloads validation scaler using training data tail to provide context while avoiding future leakage
    - Note: This design is intentional but should be reviewed/tested to ensure no data leakage occurs
  - Per epoch:
    1. Process training batches in shuffled order (forward, loss, backward, update)
    2. Process validation batches in sequential order (forward, loss only; no updates)
    3. Check early stopping and learning rate decay
  - Gradient clipping: rescales `||grad|| → maxNorm` if exceeds threshold
  - Learning rate decay: halves LR if validation loss plateaus (max 3 times)
- **Output**: Best validation loss, epoch number, total epochs

---

### Metrics and Evaluation (`model/metrics.h/cpp`)

#### **Prediction-to-Position Conversion**
- `predictionsToPositions()` – Converts model predictions to binary trading positions:
  - Position = 1.0 (long) if prediction > threshold
  - Position = 0.0 (flat/cash) otherwise
  - **Status**: Implemented
- `predictionsToScaledPositions()` – Declared but implementation not found; may not be implemented

#### **Daily PnL & Turnover**
- `calcSharpeAndTurnover()` – Computes metrics for positions vs. actual returns:
  - Gross Return: `position[t] * actual_return[t]`
  - Net Return: Gross - transaction costs
  - Turnover: `|position[t] - position[t-1]|` (position changes)
  - Sharpe Ratio: `mean_daily_return / std_daily_return * sqrt(252)` (annualized)
  - **Status**: Implemented

#### **Key Limitation**:
- These metrics are for **evaluation only**—not proof of profitability
- Transaction costs are assumed but may not reflect real-world slippage
- Sharpe ratio over validation period is a weak signal (may reflect overfitting)

---

### Hyperparameter Search (`search/`)

#### **gridSearch** function
- **Status**: Implemented
- Exhaustive search over all parameter combinations
- Writes results to `data/results.csv` (note: separate from Trainer's `tests/results.csv`)
- Not yet integrated into main training flow

#### **randomSearch** function
- **Status**: Declared in header; implementation not found (may not be implemented)
- Intended to sample N random parameter combinations
- Not yet integrated into main training flow

---

## Build System

**CMake** with C++20 standard:
- `add_executable(stock_bot main.cpp ${SRC_FILES})` → main executable
- `add_executable(testbed tests/testbed.cpp ${SRC_FILES})` → testbed executable
- Requires Eigen3 3.3+ (linear algebra library)
- Compiler flags: `-Wall -Wextra -Wpedantic -Werror` (strict warnings as errors)

### Milestone 2: Build Verification - PASSED ✅

**Configuration Date**: June 26, 2026

**Build Commands Used**:
```bash
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target stock_bot
cmake --build . --target testbed
```

**Configuration Result**: ✅ **PASSED**
- CMake version: 3.22.1 (requires ≥ 3.20)
- Configuration completed successfully with no errors or warnings

**Compilation Result**: ✅ **PASSED**
- `stock_bot` target: Built successfully
- `testbed` target: Built successfully
- No compiler warnings or errors on either target
- Compiler: g++ 11.4.0 (supports C++20)

**Dependencies**:
- ✅ Eigen3 3.4.0 installed (requires ≥ 3.3)
- ✅ CMake 3.22.1 installed (requires ≥ 3.20)
- ✅ C++20 compiler available

**Executables**:
- Main executable: `/home/caidenmarley/stock-bot/build/stock_bot` (1.6M, executable)
- Test executable: `/home/caidenmarley/stock-bot/build/testbed` (1.6M, executable)

**Recommended Smoke Test Command** (Milestone 3):
```bash
./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1
```

**Notes**:
- Build is clean with no warnings despite `-Werror` flag
- Both executables are linked and ready to run
- No build errors or configuration issues detected

---

## What Appears Complete

1. ✅ **CSV parsing** – CSVLoader handles OHLCV data efficiently
2. ✅ **Rolling-window scaling** – RollingWindowScaler works correctly for independent features
3. ✅ **Sequence construction** – StockData converts data into LSTM-ready batches
4. ✅ **LSTM forward pass** – Standard equations, Xavier init
5. ⚠️ **LSTM backward pass** – Tiny deterministic finite-difference BPTT check passed in Milestone 10; broader coverage is still pending.
6. ⚠️ **Dense output layer** – Finite-difference gradient check passed for a deterministic case in Milestone 8; not exhaustively proven.
7. ⚠️ **Huber loss** – Forward/backward behavior covered by Milestone 7 tests; not exhaustively proven.
8. ✅ **AdaBelief optimizer** – Parameter updates with bias correction
9. ✅ **Training loop** – Epoch iteration, batch processing, validation
10. ✅ **Learning rate decay** – Halves LR on validation plateau
13. ✅ **Gradient clipping** – Prevents exploding gradients (implementation present)
12. ✅ **Early stopping** – Halts if no improvement for N epochs
13. ✅ **Rolling validation folds** – 3 time-ordered splits (60%, 70%, 80%)
15. ✅ **Build system** – CMake configuration with Eigen3 dependency (Milestone 2: Build Verification PASSED on June 26, 2026)
16. ✅ **Reproducibility** – Seed setting via `LSTMCell::setGlobalInitSeed()` (implementation present but full reproducibility not yet verified)

---

## What Appears Incomplete

1. ⚠️ **Rolling scaler coverage is improved but not exhaustive**
  - Core rolling behavior is now covered by dedicated tests (Milestone 5)
  - Remaining gaps: stress/performance on long streams and floating-point stability under extreme magnitudes
  - Validation-split leakage checks still depend on Trainer/StockData integration tests (Milestone 12)

2. ⚠️ **StockData coverage is improved but not exhaustive**
  - Core sequence construction, batching, reset/error behavior, target alignment, and shuffled ordering are now covered (Milestone 6)
  - Remaining gaps: deeper numeric validation of scaled input tensor values and stress/performance behavior on large datasets
  - Integration leakage guarantees across train/validation folds still require Milestone 12 validation tests

3. ⚠️ **ML correctness validation** – LSTM checks still incomplete
  - Dense layer finite-difference gradient check now passes (Milestone 8)
  - LSTM finite-difference gradient check now passes for a tiny deterministic case (Milestone 10)
  - Broader LSTM gradient coverage (more shapes/sequences/loss setups) is still pending
  - Milestone 9 parameter-order consistency checks now pass after fixing a test issue in the original single-index mutation assertion

4. ⚠️ **Loss/metrics coverage is improved but not exhaustive**
  - Huber forward/backward behavior and core metrics pipeline are now covered (Milestone 7)
  - `predictionsToScaledPositions` is declared in headers but not implemented in source
  - Trading realism (slippage/model assumptions) still needs review in Milestone 12

5. ⚠️ **Data leakage checks** – Validation scaler usage not verified
   - Does validation data use its own rolling scaler?
   - Does validation scaler peek at future validation data?
   - No tests confirm time-series integrity

6. ⚠️ **Integration tests** – No end-to-end test with real data
   - Only manual execution of `./stock_bot`
   - No CI/CD or automated test suite

7. ⚠️ **Hyperparameter search** – Exists but not connected to main flow
   - Functions in `search/hyperparam_search.h` defined but not called
   - No integration with `main.cpp`

8. ⚠️ **Multi-model ensemble** – Long-term goal, not implemented
   - Sentiment model: planned but not started
   - Source-reliability model: planned but not started
   - Ensemble layer: planned but not started

9. ⚠️ **Web scraping** – Planned for sentiment data, not started

---

## Highest-Risk Areas

### 1. **Time-Series Data Leakage** (Critical)
- **Risk**: Validation data might use future information in scaling or batching
- **Current Status**: 
  - Training and validation splits are correctly ordered
  - Trainer preloads validation scaler using training data tail (intentional design to provide context)
  - Rolling scaler is applied independently to each fold
  - Need to verify: Preloading strategy prevents future leakage while preserving scaler context
- **Mitigation**: Add explicit tests for scaler state isolation and preloading correctness

### 2. **LSTM Gradient Correctness** (Critical)
- **Risk**: BPTT implementation might have sign errors or off-by-one bugs
- **Current Status**: 
  - Forward pass follows standard LSTM equations
  - Tiny deterministic numerical gradient check passed in Milestone 10
  - Broader-case coverage (different shapes/sequences/loss setups) is still pending
  - Parameter vector ordering checks passed for covered cases in Milestone 9
- **Mitigation**: Expand LSTM gradient checks to additional configurations before considering BPTT broadly verified

### 3. **Parameter Vector Ordering** (High)
- **Risk**: `getParametersVector()`, `setParametersVector()`, `getGradientsVector()` must be perfectly aligned
- **Current Status**: 
  - Observed ordering from implementation appears to be: `Wf, Uf, bf, Wi, Ui, bi, Wc, Uc, bc, Wo, Uo, bo`
  - Milestone 9 test suite now passes all checks (6/6)
  - Original failure was caused by a test bug (reference aliasing in the old single-index assertion), not a confirmed production ordering bug
- **Mitigation**: Keep ordering assertions in regression tests and continue validating through broader LSTM gradient cases

### 4. **Dense Layer Gradient Accumulation** (High)
- **Risk**: Gradients might be summed instead of averaged, or vice versa
- **Current Status**: 
  - `Dense::backward()` accumulates via `dW += dLdy * h^T` (implemented)
  - Optimizer handles learning rate scaling
  - Milestone 8 finite-difference check passed for a deterministic case
- **Mitigation**: Add additional multi-shape/multi-batch dense checks if deeper accumulation coverage is needed

### 5. **Validation Loss Interpretation** (High)
- **Risk**: Low validation loss does not guarantee profitability or realistic trading performance
- **Current Status**: 
  - Huber loss forward/backward and core metrics formulas are covered by Milestone 7 tests
  - Trading metrics (Sharpe, PnL, turnover) are implemented, but realism/market assumptions are not fully validated
  - Transaction cost model is simple; slippage not simulated
- **Mitigation**: Document limitations and review cost assumptions in Milestone 12

### 6. **Eigen Tensor Memory Layout** (Medium)
- **Risk**: Row-major vs. column-major assumptions could cause subtle indexing bugs
- **Current Status**: 
  - StockData uses `Eigen::RowMajor` for consistency
  - No explicit documentation of memory layout expectations
- **Mitigation**: Add inline comments and a consistency test

### 7. **Random Seed and Reproducibility** (Medium)
- **Risk**: Seeding only affects LSTM weight init; other randomness might persist
- **Current Status**: 
  - `LSTMCell::setGlobalInitSeed()` controls Xavier initialization
  - AdaBelief determinism not verified
  - No full reproducibility test (same seed → exact same results)
- **Mitigation**: Add end-to-end reproducibility test

---

## Recommended Next Steps (Milestone Sequence)

Follow the **Codebase Recovery** milestones in order:

### **Milestone 2: Build Verification** ✅ COMPLETE (June 26, 2026)
- [x] Confirmed current build workflow:
  - [x] Ran `cmake ..` – configuration successful
  - [x] Ran `cmake --build . --target stock_bot` – compiled successfully, no warnings
  - [x] Ran `cmake --build . --target testbed` – compiled successfully, no warnings
- [x] Documented build dependencies:
  - CMake 3.22.1 (requires ≥ 3.20) ✅
  - g++ 11.4.0 (C++20 support) ✅
  - Eigen3 3.4.0 (requires ≥ 3.3) ✅
- [x] Executables verified: stock_bot and testbed both present and executable (1.6M each)

### **Milestone 3: Minimal Run Verification** ✅ COMPLETE (June 26, 2026)
- [x] Ran smoke test with tiny configuration
- [x] Verified executables complete without crash
- [x] Verified output is reasonable (loss values, fold results, metrics)
- [x] Documented sample outputs

**Commands Executed**:
```bash
./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1
./build/testbed
```

**stock_bot Smoke Test Result**: ✅ **SUCCESS**

Command executed successfully in single epoch with seed 0. Output summary:
```
Parsed 410 rows in 0.000698641 seconds from data/AAAU.csv
3 rolling validation folds executed:

FOLD 1: train=[0,237] val=[237,410]
  - Epoch 1: train_loss=0.000043 | val_loss=0.000077
  - MAE: 0.008328, RMSE: 0.012436, Directional Accuracy: 0.386
  - PnL: Sharpe=-1.89, AvgTurnover=0.025

FOLD 2: train=[0,276] val=[276,410]
  - Epoch 1: train_loss=0.000041 | val_loss=0.000076
  - MAE: 0.007786, RMSE: 0.012292, Directional Accuracy: 0.546
  - PnL: Sharpe=1.13, AvgTurnover=0.034

FOLD 3: train=[0,316] val=[316,410]
  - Epoch 1: train_loss=0.000042 | val_loss=0.000112
  - MAE: 0.009406, RMSE: 0.014977, Directional Accuracy: 0.658
  - PnL: Sharpe=0.04, AvgTurnover=0.076

SUMMARY: avg best val loss = 0.000088 | best fold loss = 0.000076
```

**Testbed Result**: ✅ **SUCCESS**

Testbed (rolling scaler synthetic test) output:
```
Day      Scaled[0]  (all features identical)
------------------------------------
Day 1     0.0000
Day 2     1.0000
Day 3     1.2247
Day 4     1.2247
Day 5     1.2247

After reset, feeding day=10:
Scaled after reset (should be 0): 0.0000
```

**Runtime Behavior Assessment**: ✅ **REASONABLE**
- ✅ Both executables complete without crashes or hangs
- ✅ CSV parsing works (410 rows parsed in 0.7ms)
- ✅ Training loop executes successfully
- ✅ Loss values are positive and in reasonable range (0.00004-0.00011)
- ✅ Validation loss slightly higher than training loss (expected)
- ✅ Metrics output (MAE, RMSE, Sharpe, Turnover) all present and numeric
- ✅ Rolling validation folds executed with time-ordered splits; validation ranges are nested/overlapping by design and should be reviewed later
- ✅ Testbed correctly validates rolling scaler behavior (0 for constant values, proper scaling)
- ⚠️ Note: Sharpe ratios vary widely (-1.89 to 1.13), consistent with very small returns and short 94-day validation windows
- ⚠️ Note: Directional accuracy is 38-65%, which is modest but not unexpected for 1 epoch on small dataset

**Pipeline Smoke Test**: ✅ **PASSED**
- CSV file loads successfully (44K file with 410 rows)
- End-to-end execution completes without crashes
- Data flows through entire pipeline: parse → scale → batch → train → validate
- **Caveat**: Parser correctness, target alignment, time-series leakage, and data integrity still require dedicated unit tests (Milestones 4–7, 12)

**Reproducibility**: ⚠️ **SEED ACCEPTED, NEEDS VERIFICATION**
- `--seed 0` flag was accepted and single run completed
- Deterministic behavior suggests seeding works, but full reproducibility requires repeated runs with same seed to confirm byte-for-byte identical output

### **Milestone 4: Parser Tests** ✅ COMPLETE (June 26, 2026)

**Files Changed** (intentionally modified: 3 files):

- `tests/parser_test.cpp` (NEW) – Comprehensive parser test suite (8 test cases, 280 lines)
- `CMakeLists.txt` (UPDATED) – Added parser_test executable target
- `PROJECT_STATE.md` (UPDATED) – Documented Milestone 4 results

**Production code not modified**: No changes to `src/` or `include/` directories

**Build/Run Commands**:
```bash
# Configure and build (one-time from repo root)
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target parser_test

# Run tests (from repo root)
cd /home/caidenmarley/stock-bot
./build/parser_test

# Alternative: Run tests (from build directory)
cd /home/caidenmarley/stock-bot/build
./parser_test
```

**Test Results**: ✅ **ALL TESTS PASSED**

- ✅ **Test 1: Header Row Skipped** – CSV header correctly skipped, first data row parsed
- ✅ **Test 2: Normal Row Parsing** – All 6 columns + volume parse to correct numeric values (double/uint64)
- ✅ **Test 3: Blank Lines Handled** – Empty lines skipped gracefully, data rows still parsed correctly
- ✅ **Test 4: Unix Line Endings (LF)** – CSV with \n line endings parsed correctly
- ✅ **Test 5: Windows Line Endings (CRLF)** – CSV with \r\n line endings parsed correctly
- ✅ **Test 6: Large uint64 Volume Parsing** – Max uint64 (18446744073709551615) parsed correctly
- ✅ **Test 7: Malformed Numeric Data Exception** – Non-numeric field throws `std::runtime_error` with "from chars failed" message
- ✅ **Test 8: Date string_view Lifetime Verification** – Date string_view remains valid while CSVLoader alive; captured as std::string for persistence

**Compilation**: ✅ **CLEAN**
- No warnings or errors
- Parser_test executable linked successfully (1.6M)
- All dependencies resolved (Eigen3, std libraries)

**Parser Test Suite Coverage**:
- ✅ CSV parsing handles all tested edge cases correctly
- ✅ Header skipping works reliably
- ✅ Line ending handling covers both Unix (\n) and Windows (\r\n)
- ✅ Numeric parsing (floats, uint64) passes all test cases
- ✅ Error handling for malformed data throws exceptions as designed
- ✅ Date `std::string_view` values remain valid while CSVLoader is alive
- ✅ Blank line handling is implemented and works as tested

**No parser bugs detected by current tests** – All tested behaviors match expected design

**Parser correctness is improved by tests but not exhaustively proven** – Additional edge cases (very large files, Unicode, special characters, etc.) remain untested

**Risks Addressed**:
- `std::string_view` lifetime – Date references remain valid while CSVLoader object is alive; when copied to std::string as shown in Test 8, remain valid indefinitely
- CSV parsing coverage – Numeric types, line endings, and currently-tested edge cases all handled correctly
- Malformed data handling – Exceptions thrown properly with clear error messages

### **Milestone 5: Rolling Scaler Tests** ✅ COMPLETE (June 26, 2026)

**Files Changed** (intentionally modified: 3 files):

- `tests/rolling_window_scaler_test.cpp` (NEW) – Dedicated rolling scaler test suite (8 tests)
- `CMakeLists.txt` (UPDATED) – Added `rolling_window_scaler_test` executable target
- `PROJECT_STATE.md` (UPDATED) – Documented Milestone 5 results

**Production code not modified**: No changes to `src/` or `include/` directories

**Build/Run Commands**:
```bash
# Configure and build
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target rolling_window_scaler_test

# Run tests from repo root
cd /home/caidenmarley/stock-bot
./build/rolling_window_scaler_test
```

**Test Results**: ✅ **8/8 PASSED**

- ✅ **Test 1: First value after reset returns 0** when stddev is 0
- ✅ **Test 2: Constant feature values scale to 0**
- ✅ **Test 3: Growing-window mean/stddev correctness** verified with explicit expected values
- ✅ **Test 4: Window drop-off correctness** verified for window size 3 with values 1,2,3,4
  - Day 1: 0
  - Day 2: 1.0
  - Day 3: ~1.224744871
  - Day 4: ~1.224744871 (active window [2,3,4])
- ✅ **Test 5: Per-feature independence** verified
- ✅ **Test 6: Volume feature inclusion/scaling** verified
- ✅ **Test 7: `reset()` clears internal state** verified
- ✅ **Test 8: Current day uses available rolling window only** (no future values)

**No scaler bugs detected by current tests** for covered behaviors.

**Remaining scaler risks (unresolved)**:
- Not exhaustively proven under extreme numeric ranges or very long sequences
- Cross-module leakage behavior (Trainer validation preloading) remains a Milestone 12 concern

### **Milestone 6: StockData Tests** ✅ COMPLETE (June 26, 2026)

**Files Changed** (intentionally modified: 3 files):

- `tests/stock_data_test.cpp` (NEW) – Dedicated StockData test suite (9 tests)
- `CMakeLists.txt` (UPDATED) – Added `stock_data_test` executable target
- `PROJECT_STATE.md` (UPDATED) – Documented Milestone 6 results

**Production code not modified**: No changes to `src/` or `include/` directories

**Build/Run Commands**:
```bash
# Configure and build
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target stock_data_test

# Run tests from repo root
cd /home/caidenmarley/stock-bot
./build/stock_data_test
```

**Test Results**: ✅ **9/9 PASSED**

- ✅ **Test 1: Constructor rejects too-small datasets** (`numDays < sequenceLength + 1`)
- ✅ **Test 2: `numWindows = numDays - sequenceLength`**
- ✅ **Test 3: `nextBatch()` returns expected batch size and expected targets**
- ✅ **Test 4: Final partial batch works** when `numWindows` is not divisible by `batchSize`
- ✅ **Test 5: `reset()` restarts batching from the beginning**
- ✅ **Test 6: `nextBatch()` throws** after all batches are consumed
- ✅ **Test 7: Target alignment formula is correct**:
  - `target(seq) = (close[seq + sequenceLength] - close[seq + sequenceLength - 1]) / close[seq + sequenceLength - 1]`
- ✅ **Test 8: `nextBatchShuffled()` respects provided order**
- ✅ **Test 9: Input tensor shape is correct** `[currentBatch, sequenceLength, numFeatures]`

**No StockData bugs detected by current tests** for covered cases.

**StockData behavior is tested for covered cases, not exhaustively proven.**

**Remaining StockData risks (unresolved)**:
- Scaled feature values were not exhaustively numerically validated in this milestone (focus was batching/targets/order/error flow)
- Cross-module concerns (validation scaler preloading and leakage guarantees across folds) remain Milestone 12 work

### **Milestone 7: Loss and Metrics Tests** ✅ COMPLETE (June 26, 2026)

**Files Changed** (intentionally modified: 3 files):

- `tests/loss_metrics_test.cpp` (NEW) – Dedicated Huber loss + metrics test suite (10 tests)
- `CMakeLists.txt` (UPDATED) – Added `loss_metrics_test` executable target
- `PROJECT_STATE.md` (UPDATED) – Documented Milestone 7 results

**Production code not modified**: No changes to `src/` or `include/` directories

**Build/Run Commands**:
```bash
# Configure and build
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target loss_metrics_test

# Run tests from repo root
cd /home/caidenmarley/stock-bot
./build/loss_metrics_test
```

**Test Results**: ✅ **10/10 PASSED**

**Huber loss coverage**:
- ✅ Forward pass in quadratic region (`|residual| <= delta`) with expected `0.5 * residual^2`
- ✅ Forward pass in linear region (`|residual| > delta`) with expected `delta * (abs(residual) - 0.5 * delta)`
- ✅ Batch averaging equals mean of per-example Huber losses
- ✅ Backward gradient sign in quadratic region is correct (`dL/dprediction = -residual / n`)
- ✅ Backward gradient magnitude clips to `delta / n` in linear region

**Metrics coverage**:
- ✅ `predictionsToPositions`: `prediction > threshold` maps to `1.0`, otherwise `0.0`
- ✅ `calcDailyPnLAndTurnover`: gross return, turnover, trading costs, and net return match hand-calculated values
- ✅ `mean` and sample `stddev` (Bessel correction) match expected values
- ✅ `sharpe`: returns `0.0` for too-small input/zero stddev and matches a simple non-zero case
- ✅ `calcSharpeAndTurnover`: end-to-end pipeline (`predictions -> positions -> net returns -> sharpe/turnover`) matches expected values

**No Huber/metrics bugs detected by current tests** for covered cases.

**Huber/metrics behavior is tested for covered cases, not exhaustively proven.**

**Remaining Huber/metrics risks (unresolved)**:
- `predictionsToScaledPositions` remains declared but not implemented in `metrics.cpp` (not invoked in tests)
- Trading metrics are mathematically validated for tested formulas, but still not proof of real-world profitability

### **Milestone 8: Dense Layer Gradient Check** ✅ COMPLETE (June 26, 2026)

**Files Changed** (intentionally modified: 3 files):

- `tests/dense_gradient_test.cpp` (NEW) – Dedicated Dense gradient test suite (6 tests)
- `CMakeLists.txt` (UPDATED) – Added `dense_gradient_test` executable target
- `PROJECT_STATE.md` (UPDATED) – Documented Milestone 8 results

**Production code not modified**: No changes to `src/` or `include/` directories

**Build/Run Commands**:
```bash
# Configure and build
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target dense_gradient_test

# Run tests from repo root
cd /home/caidenmarley/stock-bot
./build/dense_gradient_test
```

**Test Results**: ✅ **6/6 PASSED**

- ✅ **Test 1: Dense forward** matches `y = W dot h + b`
- ✅ **Test 2: Dense backward analytical gradients** match `dW = dLdy * h` and `db = dLdy`
- ✅ **Test 3: Numerical gradients for each weight** match analytical `dW` via central finite differences
- ✅ **Test 4: Numerical gradient for bias** matches analytical `db`
- ✅ **Test 5: `zeroGrad()` clears `dW` and `db`**
- ✅ **Test 6: Backward accumulation behavior** is explicit (`backward` twice without `zeroGrad` accumulates)

**No Dense bugs detected by current tests** for covered cases.

**Dense behavior is tested for covered cases, not exhaustively proven.**

**Remaining Dense risks (unresolved)**:
- Gradient check currently uses a single deterministic shape/example; broader multi-shape coverage may still reveal edge cases
- Integration interaction with full training loop still depends on later milestones (Milestones 9–12)

### **Milestone 9: LSTM Parameter Ordering Check** ✅ COMPLETE (June 26, 2026)

**Files Changed** (intentionally modified: 3 files):

- `tests/lstm_parameter_order_test.cpp` (NEW) – LSTM parameter/gradient vector consistency tests (6 tests)
- `CMakeLists.txt` (UPDATED) – Added `lstm_parameter_order_test` executable target
- `PROJECT_STATE.md` (UPDATED) – Documented Milestone 9 results

**Production code not modified**: No changes to `src/` or `include/` directories

**Build/Run Commands**:
```bash
# Configure and build
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target lstm_parameter_order_test

# Run tests from repo root
cd /home/caidenmarley/stock-bot
./build/lstm_parameter_order_test
```

**Test Results**: ✅ **6/6 PASSED**

- ✅ **Test 1: parameter count formula** passed
- ✅ **Test 2: set/get deterministic round-trip** passed
- ✅ **Test 3: single-index mutation safety** passed using simplified deterministic logic:
  - create deterministic parameter vector
  - mutate one fixed index
  - set mutated vector
  - verify `getParametersVector()` matches all elements of mutated vector within tiny tolerance
- ✅ **Test 4: gradient vector length matches parameter vector length** passed
- ✅ **Test 5: zeroGrad returns all-zero gradient vector** passed
- ✅ **Test 6: tiny forward/backward produces finite gradient vector entries** passed

**Observed parameter flattening order from implementation**:
- `Wf, Uf, bf, Wi, Ui, bi, Wc, Uc, bc, Wo, Uo, bo`

**Status**:
- Milestone 9 is complete for parameter-vector ordering consistency checks.
- Original Test 3 failure was due to a test issue, not a confirmed production code ordering defect.
- Per recovery rules, production code was not modified.

**Remaining LSTM parameter-order risks (unresolved)**:
- Flat ordering appears internally consistent for covered checks, but LSTM numerical gradient correctness is still unverified
- Full LSTM numerical gradient checking remains Milestone 10

### **Milestone 10: LSTM Gradient Check** ✅ COMPLETE (June 26, 2026)

**Files Changed** (intentionally modified: 3 files):

- `tests/lstm_gradient_test.cpp` (NEW) – Tiny deterministic LSTM BPTT finite-difference gradient check
- `CMakeLists.txt` (UPDATED) – Added `lstm_gradient_test` executable target
- `PROJECT_STATE.md` (UPDATED) – Documented Milestone 10 results

**Production code not modified**: No changes to `src/` or `include/` directories

**Test design**:
- LSTM shape: `inputSize=2`, `hiddenSize=4`, `sequenceLength=3`
- Deterministic parameters: `params[i] = 0.01 * sin(i + 1)`
- Fixed input sequence:
  - `x0 = [0.10, -0.20]`
  - `x1 = [0.05, 0.30]`
  - `x2 = [-0.15, 0.07]`
- Loss on final hidden state only: `0.5 * ||h_final - target||^2`
- Deterministic target: `target[j] = 0.03 * (j + 1)`
- Analytical gradient: forward over sequence, then reverse-time `backwardPass` with `deltaH = h_final - target`, `deltaC = 0`
- Numerical gradient: central finite differences with `epsilon = 1e-5`
- Tolerances used:
  - absolute tolerance `1e-4`
  - relative tolerance `1e-3`

**Build/Run Commands**:
```bash
# Configure and build
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target lstm_gradient_test

# Run test from repo root
cd /home/caidenmarley/stock-bot
./build/lstm_gradient_test
```

**Test Results**: ✅ **1/1 PASSED**

- ✅ Analytical gradient vector length equals parameter vector length
- ✅ All analytical gradients are finite
- ✅ All numerical gradients are finite
- ✅ Analytical and numerical gradients match within tolerance

**Reported error metrics**:
- Max absolute error: `2.70113e-12`
- Max relative error: `2.70113e-12`
- Worst index: `83`
- Worst analytical value: `-0.0504472`
- Worst numerical value: `-0.0504472`

**No LSTM gradient bug detected by current test** for the covered tiny deterministic case.

**Remaining LSTM gradient risks (unresolved)**:
- This is a single small-case gradient check; wider coverage (different hidden sizes, longer sequences, and alternate loss setups) is still needed
- Full training-loop-level LSTM behavior and time-series validation leakage concerns remain Milestone 11–12 scope

### **Milestone 11: Training Loop Review**
- [ ] Audit training loop:
  - [ ] Hidden and cell state reset per batch/sequence
  - [ ] Gradient accumulation during backprop
  - [ ] Gradient clipping correctness
  - [ ] LSTM and Dense update correctness
  - [ ] Learning rate decay logic
  - [ ] Early stopping logic
- [ ] Add comments for each step

### **Milestone 12: Time-Series Validation Review**
- [ ] Verify no data leakage:
  - [ ] Training data precedes validation data (no shuffling)
  - [ ] Validation rolling scaler may be preloaded with past training data as context, but must not see future validation data
  - [ ] Validation batches are contiguous and sequential
  - [ ] Rolling validation folds executed with time-ordered train/validation splits; validation ranges are nested/overlapping by design and should be reviewed later
- [ ] Add integration test for 3-fold validation integrity

### **Milestone 13: Refactor and Documentation**
- [ ] After tests pass, refactor if needed:
  - Better variable names
  - Clearer function decomposition
  - Inline comments for mathematical logic
  - Reduce code duplication
- [ ] Create `docs/` folder with:
  - `BUILD.md` – build steps and dependencies
  - `ARCHITECTURE.md` – data flow and module overview
  - `ML_CORRECTNESS.md` – gradient derivations, parameter ordering
  - `KNOWN_RISKS.md` – data leakage, overfitting, metric interpretation

---

## Summary

**Current State**: 
- Core LSTM, training loop, and rolling validation are implemented and verified to run
  - **Milestone 2–10 Status**: Build, runtime, parser, rolling-scaler, StockData, loss/metrics, Dense gradient, and tiny-case LSTM gradient verification PASSED ✅
    - Milestone 2: CMake configured, both targets compiled cleanly
    - Milestone 3: Both executables run successfully, output is reasonable
    - Milestone 4: CSVLoader robustness verified (8 comprehensive parser tests all passed)
    - Milestone 5: RollingWindowScaler behavior verified (8 dedicated scaler tests all passed)
    - Milestone 6: StockData behavior verified (9 dedicated StockData tests all passed)
    - Milestone 7: Huber loss and metrics behavior verified (10 dedicated loss/metrics tests all passed)
    - Milestone 8: Dense backward gradient check verified (6 dedicated Dense gradient tests all passed)
    - Milestone 10: LSTM tiny-case finite-difference gradient check verified (1 dedicated test passed)
  - Milestone 9: LSTM parameter ordering consistency verified (6/6 tests passed)
  - Test suites now include `testbed`, `parser_test`, `rolling_window_scaler_test`, `stock_data_test`, `loss_metrics_test`, `dense_gradient_test`, `lstm_parameter_order_test`, and `lstm_gradient_test`
- LSTM backward/BPTT now has tiny-case numerical verification; broader-case verification is still pending
- Seed option is implemented and was accepted during the smoke test, but full reproducibility still requires repeated-run comparison

**Main Risks**: 
- Limited-scope LSTM gradient verification (only tiny deterministic case) and data leakage
- Low coverage for end-to-end validation integrity
- Trading metrics are unvalidated and should not be interpreted as profit signals

**Next Actions** (in priority order):
1. ✅ Verify build and configuration (Milestone 2–3) – **COMPLETE**
2. ✅ Parser tests (Milestone 4) – **COMPLETE (June 26, 2026)**
3. ✅ Rolling scaler tests (Milestone 5) – **COMPLETE (June 26, 2026)**
4. ✅ StockData tests (Milestone 6) – **COMPLETE (June 26, 2026)**
5. ✅ Loss/metrics tests (Milestone 7) – **COMPLETE (June 26, 2026)**
6. ✅ Dense gradient check (Milestone 8) – **COMPLETE (June 26, 2026)**
7. ✅ LSTM parameter-order checks (Milestone 9) – **COMPLETE (June 26, 2026)**
8. ✅ LSTM gradient check (Milestone 10) – **COMPLETE (June 26, 2026)**
9. Audit training loop and validation logic (Milestone 11–12) – **Next step**
10. Refactor and document (Milestone 13)

This recovery approach prioritizes understanding and correctness before expansion to multi-model ensemble or web scraping.
