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
11. ✅ **Gradient clipping** – Prevents exploding gradients (implementation present)
12. ✅ **Early stopping** – Halts if no improvement for N epochs
13. ✅ **Rolling validation folds** – 3 time-ordered splits (60%, 70%, 80%)
14. ✅ **Build system** – CMake configuration with Eigen3 dependency (Milestone 2: Build Verification PASSED on June 26, 2026)
15. ✅ **Reproducibility hooks** – Seed setting via `LSTMCell::setGlobalInitSeed()` is implemented (full same-seed reproducibility is still pending verification)

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

5. ⚠️ **Time-series validation coverage is improved but not exhaustive**
  - Milestone 12 found no clear future-data leakage for inspected paths.
  - Past-context validation scaler preloading appears intentional.
  - Focused synthetic tests now cover split ordering, sequential validation batching, target alignment, and early-window no-future influence checks.
  - Full end-to-end validation integrity is still not exhaustively proven.

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
  - Milestone 12 audit and focused synthetic tests passed for covered checks
  - No clear future-data leakage detected for inspected paths
  - Trainer preloads validation scaler using training data tail; past-context preloading appears intentional
  - Validation ranges are nested/overlapping by design and current-day scaling remains a modelling/evaluation assumption
  - Not exhaustively proven
- **Mitigation**: Keep leakage-focused regression tests and expand end-to-end fold-level validation integrity checks

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
- Flat ordering appears internally consistent for covered checks
- At this milestone stage, broader LSTM numerical gradient coverage was still pending; Milestone 10 later added a tiny-case finite-difference check

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

### **Milestone 11: Training Loop Review** ✅ COMPLETE (June 26, 2026)

**Scope**: Review-first audit only. No production code changes were made.

**Files Inspected**:
- `src/model/trainer.cpp`
- `include/model/trainer.h`
- `src/model/lstm.cpp`
- `include/model/lstm.h`
- `include/model/dense.h`
- `src/model/huber_loss_function.cpp`
- `src/model/ada_belief.cpp`
- `src/inputs/stock_data.cpp`
- `include/inputs/stock_data.h`
- `main.cpp`
- `.gitignore`

**Files Changed**:
- `PROJECT_STATE.md` (this milestone update only)

**Build/Test Commands Used**:
- None (review-only milestone; no code execution required)

**Audit Findings (Training Loop)**:

1. **Hidden/cell state reset behavior**
- In training, `lstm.reset()` is called once per sequence (inside per-example loop) before time-step forward passes.
- In validation, `lstm.reset()` is also called once per sequence.
- This means state is reset per sequence, not carried across sequences or batches, which appears consistent with the current window-based sequence design.

2. **Gradient lifecycle**
- `lstm.zeroGrad()` and `outputLayer.zeroGrad()` are called once per batch, before iterating the sequences in that batch.
- Within the batch, gradients accumulate intentionally across all sequences/time steps.
- After parameter updates, next batch starts from zeroed gradients.
- For inspected paths, accumulation behavior appears intentional rather than accidental.

3. **Backward pass flow**
- Observed chain is: Huber backward (`dL/dy`) -> Dense backward (accumulate `dW`, `db`) -> LSTM backward through time using `dL/dh = W^T * dL/dy` and reverse-time `backwardPass` calls.
- LSTM gradients are updated via AdaBelief using flattened parameter/gradient vectors.
- Dense gradients are updated in a separate SGD-style step (`W -= lr*dW`, `b -= lr*db`).
- Both LSTM and Dense parameter sets are updated each training batch.

4. **Parameter update flow**
- LSTM flow: `getParametersVector()` + `getGradientsVector()` -> clip -> `optimiser.update(params, grads)` -> `setParametersVector(params)`.
- Dense flow: gradient norm clip on `(dW, db)` -> direct learning-rate step.
- Milestone 9 and 10 evidence supports covered assumptions for parameter ordering and tiny-case gradient behavior; this audit did not re-run those tests.

5. **Gradient clipping**
- LSTM uses global-norm clipping on the flattened LSTM gradient vector before AdaBelief update.
- Dense uses separate norm clipping for `dW` and `db` before SGD step.
- Concern: clipping is not performed as one combined model-wide norm across LSTM + Dense together; clipping is component-wise. This may be acceptable by design, but should remain a documented risk/assumption.

6. **Training vs validation behavior**
- Training uses shuffled sequence order via `nextBatchShuffled(...)`.
- Validation uses sequential contiguous batches via `nextBatch()` and `hasAnotherBatch()`.
- Validation loop performs forward/loss/metrics only; no optimizer updates and no backward calls are made.

7. **Learning rate decay and early stopping**
- If validation does not improve by tolerance, `noImproveCount` increments.
- At patience threshold, LR decays (bounded by `minLR`) while decay tries remain; otherwise training stops early.
- On decay, both AdaBelief LR and Dense LR are decayed, and `noImproveCount` resets.
- No clear off-by-one bug was detected in inspected control flow, but behavior remains policy-sensitive and not exhaustively proven.

8. **Results/logging behavior**
- Trainer appends epoch metrics to `tests/results.csv`.
- `main.cpp` truncates `tests/results.csv` at startup.
- `.gitignore` includes `tests/results.csv`; however, the file is currently present in the repository and will still show modifications if tracked. This is a workflow hygiene concern for generated artifacts.

**Milestone 11 Conclusion**:
- No clear training-loop bug detected during this audit.
- Behavior appears consistent with current design for inspected paths.
- Not exhaustively proven.
- Remaining concerns are primarily around clipping policy consistency and generated-results file hygiene, not a confirmed math/control-flow defect.

### **Milestone 12: Time-Series Validation Review** ✅ COMPLETE (June 26, 2026)

**Scope**: Validation-integrity audit and focused synthetic testing only. No production code changes were made.

**Files Inspected**:
- `main.cpp`
- `src/model/trainer.cpp`
- `include/model/trainer.h`
- `src/inputs/stock_data.cpp`
- `include/inputs/stock_data.h`
- `src/inputs/rolling_window_scaler.cpp`
- `include/inputs/rolling_window_scaler.h`
- `.gitignore`

**Files Changed**:
- `tests/time_series_validation_test.cpp` (NEW)
- `CMakeLists.txt` (added test target)
- `PROJECT_STATE.md` (this milestone update)

**Build/Run Commands Used**:
```bash
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target time_series_validation_test

cd /home/caidenmarley/stock-bot
./build/time_series_validation_test
```

**Test Result**: ✅ **3/3 PASSED**

- `[PASS] fold split ordering and overlap`
- `[PASS] validation batching and target alignment`
- `[PASS] validation preloading no future influence on early windows`

**Time-series validation review findings**:

1. **Fold split ordering**
- `main.cpp` constructs folds at 60%, 70%, and 80% of `maxStartSequence = rawSize - sequenceLength`.
- For each fold, training is `[0, splitStart)` and validation is `[splitStart, rawSize)`, so training data precedes validation data in time.
- Validation ranges are time-ordered and nested/overlapping by design (later folds are tail subsets of earlier validation ranges).
- This is acceptable for expanding-window style evaluation, but fold dependence should remain documented when interpreting aggregate metrics.

2. **Validation scaler preloading**
- In `Trainer` constructor, validation scaler is preloaded with only the tail of training data (`max(windowSize, split)` constrained to pre-split indices).
- Validation rows are then processed in-order during `StockData` construction; no direct path was found where future validation rows are used to scale earlier validation rows.
- Past-context validation scaler preloading appears intentional.

3. **StockData scaling behavior**
- `StockData` calls `scaler.add(rawData[day])` and then `scaledValuesPerDay()` for that same day.
- This means current-day features are scaled using a window containing current day + prior days (bounded by `windowSize`).
- Current-day scaling is a modelling assumption for next-day prediction and should remain documented.

4. **Validation batching**
- Validation path in `Trainer::run` uses sequential `nextBatch()` with `hasAnotherBatch()`.
- Training path uses shuffled order with `nextBatchShuffled(...)`.
- This matches current design.

5. **Target alignment**
- `StockData` targets are computed as next-day return from the sequence end day: `(close[t+1] - close[t]) / close[t]`.
- Milestone 6 evidence and Milestone 12 synthetic test support that validation targets follow the same next-day alignment logic.
- Full end-to-end target integrity across all folds is still not exhaustively proven.

6. **Results/logging hygiene**
- Trainer writes/appends epoch metrics to `tests/results.csv`; `main.cpp` truncates that file at startup.
- `.gitignore` includes `tests/results.csv`.
- `tests/results.csv` is currently tracked by git (verified via `git ls-files`), so it can still appear modified despite ignore rules.
- Recommended hygiene (no code change applied in this milestone): untrack generated CSVs if the team wants clean status by default.

**Milestone 12 Conclusion**:
- No clear future-data leakage detected for inspected paths.
- Past-context validation scaler preloading appears intentional.
- Current-day scaling is a modelling assumption for next-day prediction and should remain documented.
- Not exhaustively proven.

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

### **Milestone 13A: Documentation Extraction** ✅ COMPLETE (June 26, 2026)

**Scope**: Documentation-only extraction from `PROJECT_STATE.md` into standalone `docs/` files.

**Files Changed**:
- `docs/BUILD.md` (NEW)
- `docs/ARCHITECTURE.md` (NEW)
- `docs/ML_CORRECTNESS.md` (NEW)
- `docs/KNOWN_RISKS.md` (NEW)
- `PROJECT_STATE.md` (this milestone note only)

**No production code changed**:
- No changes to `src/`, `include/`, `tests/`, `CMakeLists.txt`, or `main.cpp` were made in this milestone.

**Docs Created**:
- `docs/BUILD.md`: dependency requirements, configure/build/run commands, generated-results file notes.
- `docs/ARCHITECTURE.md`: high-level scope, data flow, module responsibilities, training vs validation behavior, and not-yet-implemented components.
- `docs/ML_CORRECTNESS.md`: tested coverage summary, LSTM parameter ordering, Dense/LSTM gradient-check summaries, and documented modelling assumptions.
- `docs/KNOWN_RISKS.md`: current known risks, including leakage, gradient-coverage limits, clipping policy, generated-file hygiene, and missing roadmap components.

**Remaining Milestone 13 work**:
- Optional refactor planning and code-structure cleanup can be considered after this documentation extraction is committed.

### **Milestone 13B: Refactor Planning** ✅ COMPLETE (June 26, 2026)

**Scope**: Planning-only milestone. No refactor implementation performed.

**Files Changed**:
- `docs/REFACTOR_PLAN.md` (NEW)
- `PROJECT_STATE.md` (this milestone note only)

**Refactor Plan Created**:
- `docs/REFACTOR_PLAN.md` now defines prioritized categories and small, separate candidate tasks with motivation, likely files, risk level, validation command, and expected production-code impact.

**No production code changed**:
- No changes to `src/`, `include/`, `tests/`, `CMakeLists.txt`, `main.cpp`, or existing test files were made in this milestone.

**Recommended first follow-up task**:
- Generated-file hygiene for `tests/results.csv` (especially index/untracking hygiene if it is currently tracked) before larger refactor items.

### **Milestone 13C: Test Infrastructure Convenience (CTest Full Suite Command)** ✅ COMPLETE (June 26, 2026)

**Scope**: Infrastructure-only update to run the current standalone test executables through one CTest command.

**Files Changed**:
- `CMakeLists.txt` (added `enable_testing()`, `add_test(...)` registrations, and `run_tests` custom target)
- `docs/BUILD.md` (documented full-suite CTest commands and `run_tests` target)
- `docs/REFACTOR_PLAN.md` (marked top-level test command/target task as completed)
- `PROJECT_STATE.md` (this milestone note)

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target run_tests

ctest --test-dir build --output-on-failure
```

**Result**: ✅ **ALL REGISTERED TESTS PASSED (9/9)**

**Note**:
- This improves test-running convenience and consistency; it does not by itself increase model correctness guarantees.

### **Milestone 13D: Documentation Cleanup** ✅ COMPLETE (June 26, 2026)

**Documentation cleanup completed**.

**Files Changed**:
- `PROJECT_STATE.md` (stale numbering/summary/next-actions wording cleanup)

**No production code changed**:
- No changes to `src/`, `include/`, `tests/`, `CMakeLists.txt`, or `main.cpp` were made in this cleanup milestone.

### **Milestone 13E: Same-Seed Reproducibility Verification** ✅ COMPLETE (June 26, 2026)

**Scope**: Test/audit milestone to verify currently supported reproducibility guarantees without changing production behavior.

**Files Changed**:
- `tests/reproducibility_test.cpp` (NEW)
- `CMakeLists.txt` (added `reproducibility_test` target, CTest registration, and `run_tests` dependency)
- `docs/ML_CORRECTNESS.md` (added reproducibility verification scope)
- `docs/KNOWN_RISKS.md` (updated reproducibility risk wording)
- `docs/REFACTOR_PLAN.md` (marked reproducibility test task complete; moved next likely follow-up to broader LSTM gradient coverage)
- `PROJECT_STATE.md` (this milestone note)

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot/build
cmake ..
cmake --build . --target run_tests

cd /home/caidenmarley/stock-bot
ctest --test-dir build --output-on-failure
```

**Result**: ✅ **ALL REGISTERED TESTS PASSED (10/10)**

**What reproducibility is now verified**:
- Same LSTM global seed + same LSTM dimensions -> identical initial LSTM parameter vectors for tested cases.
- Different LSTM seeds -> different initial LSTM parameter vectors for tested cases.
- Same LSTM parameters + same fixed input sequence -> identical forward outputs (deterministic forward path after reset).

**What remains unverified**:
- Full end-to-end `stock_bot` determinism across runs is not exhaustively proven.
- Not all stochastic paths are validated under one unified same-seed guarantee in current implementation.

### **Milestone 13F: Broader LSTM Gradient Coverage** ✅ COMPLETE (July 4, 2026)

**Scope**: Test-only correctness-confidence expansion for LSTM BPTT finite-difference checks. No production behavior changes.

**Files Changed**:
- `tests/lstm_gradient_test.cpp` (expanded existing gradient test from one deterministic case to three deterministic full-vector finite-difference cases)
- `docs/ML_CORRECTNESS.md` (updated LSTM gradient coverage summary)
- `docs/KNOWN_RISKS.md` (adjusted LSTM gradient risk wording)
- `docs/REFACTOR_PLAN.md` (marked broader LSTM gradient coverage task complete; updated next follow-up)
- `PROJECT_STATE.md` (this milestone note)

**No production code changed**:
- No changes to `src/`, `include/`, `main.cpp`, or model/input/search implementations.

**Gradient cases now covered in `lstm_gradient_test`**:
1. Existing tiny base case: `inputSize=2`, `hiddenSize=4`, `sequenceLength=3`, final hidden-state loss only.
2. Additional short-sequence case: `inputSize=3`, `hiddenSize=2`, `sequenceLength=1`, final hidden-state loss only.
3. Additional longer-sequence case: `inputSize=1`, `hiddenSize=3`, `sequenceLength=5`, final hidden-state loss plus weighted final-cell loss (non-zero final `deltaC` path).

**Numerical method and tolerances**:
- Central finite differences with `epsilon=1e-5`.
- Full-vector comparison (all parameters in each case).
- Tolerances unchanged and conservative: absolute `1e-4`, relative `1e-3`.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target lstm_gradient_test
cd /home/caidenmarley/stock-bot && ./build/lstm_gradient_test
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
```

**Results**:
- `./build/lstm_gradient_test`: ✅ **3/3 PASSED**
  - Case 1 worst: max abs `2.70113e-12`, max rel `2.70113e-12`, worst index `83`.
  - Case 2 worst: max abs `4.52065e-13`, max rel `4.52065e-13`, worst index `34`.
  - Case 3 worst: max abs `4.67853e-13`, max rel `4.67853e-13`, worst index `44`.
- `ctest --test-dir build --output-on-failure`: ✅ **10/10 tests passed**.

**Milestone 13F Conclusion**:
- Broader deterministic gradient coverage improves confidence in LSTM backward behavior for covered configurations.
- This still does **not** prove exhaustive BPTT correctness across all shapes, losses, and training-loop contexts.

### **Milestone 13G: Integration-Level Validation Coverage Expansion** ✅ COMPLETE (July 4, 2026)

**Scope**: Small deterministic, test-first integration coverage across multiple numerical pipeline components without long training or production behavior changes.

**Files Changed**:
- `tests/integration_validation_test.cpp` (NEW deterministic integration test)
- `CMakeLists.txt` (added `integration_validation_test` target, CTest registration, and `run_tests` dependency)
- `docs/BUILD.md` (added build/run references for `integration_validation_test`)
- `docs/ML_CORRECTNESS.md` (added integration-validation coverage summary)
- `docs/KNOWN_RISKS.md` (slight risk wording reduction with non-exhaustive caveat preserved)
- `docs/REFACTOR_PLAN.md` (marked integration-level validation coverage expansion complete and moved next likely follow-up)
- `PROJECT_STATE.md` (this milestone note)

**No production code changed**:
- No changes to `src/`, `include/`, `main.cpp`, or model/input/search implementations.

**Deterministic integration path covered**:
1. Construct synthetic chronological OHLCV-style `PriceData` rows fully in memory.
2. Split into train and validation by time and assert validation rows occur strictly after training rows.
3. Build train `StockData` with rolling scaling.
4. Preload validation scalers from training-tail context only.
5. Build validation `StockData` sequentially and assert early validation window scaling is unchanged when only later validation rows are perturbed.
6. Assert non-empty train/validation windows and expected input tensor dimensions.
7. Run deterministic LSTM + Dense forward pass on a validation batch and assert prediction/target shape compatibility.
8. Compute Huber loss and assert finiteness.
9. Repeat deterministic forward path and assert identical outputs after reset.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target integration_validation_test
cd /home/caidenmarley/stock-bot && ./build/integration_validation_test
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
```

**Results**:
- `./build/integration_validation_test`: ✅ **1/1 PASSED**
- `cmake --build build --target run_tests`: ✅ built and executed registered tests successfully
- `ctest --test-dir build --output-on-failure`: ✅ **11/11 tests passed**

**Milestone 13G Conclusion**:
- Integration confidence improved for a deterministic cross-component validation path.
- This does **not** prove exhaustive validation integrity, exhaustive training-path correctness, full end-to-end determinism, or profitability.

### **Milestone 13H: Full End-to-End Determinism Audit (Current Supported Scope)** ✅ COMPLETE (July 4, 2026)

**Scope**: Audit/test-first determinism verification for the strongest currently controllable in-memory numerical pipeline path.

**Files Changed**:
- `tests/end_to_end_determinism_test.cpp` (NEW determinism audit test)
- `CMakeLists.txt` (added `end_to_end_determinism_test` target, CTest registration, and `run_tests` dependency)
- `docs/BUILD.md` (added build/run references for `end_to_end_determinism_test`)
- `docs/ML_CORRECTNESS.md` (updated determinism verification scope and boundaries)
- `docs/KNOWN_RISKS.md` (updated determinism-risk wording with specific blockers)
- `docs/REFACTOR_PLAN.md` (marked determinism audit complete and set next low-risk follow-up)
- `PROJECT_STATE.md` (this milestone note)

**No production code changed**:
- No changes to `src/`, `include/`, `main.cpp`, or model/input/search implementations.

**Deterministic path tested**:
1. Synthetic chronological in-memory data split into train/validation.
2. Deterministic rolling scaling and `StockData` construction across repeated runs.
3. Deterministic `nextBatchShuffled` behavior under fixed explicit order.
4. LSTM initialization controlled by `LSTMCell::setGlobalInitSeed(...)`.
5. Dense initialization controlled in test scope by explicit deterministic parameter override.
6. Repeated deterministic forward/loss path checks before update.
7. One tiny deterministic training-style update path (Huber backward + Dense backward + LSTM backward + AdaBelief/SGD update).
8. Repeated-run equality checks after update for predictions, losses, and model parameters.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target end_to_end_determinism_test
cd /home/caidenmarley/stock-bot && ./build/end_to_end_determinism_test
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
```

**Results**:
- `./build/end_to_end_determinism_test`: ✅ **2/2 PASSED**
- `cmake --build build --target run_tests`: ✅ built and executed registered tests successfully
- `ctest --test-dir build --output-on-failure`: ✅ **12/12 tests passed**

**What determinism is now verified**:
- Same-seed repeated runs are deterministic for the covered in-memory path when model states, ordering, and update steps are explicitly controlled.
- Different LSTM seeds change initial LSTM parameter vectors in this controlled path.

**What remains unverified / blockers for stronger claims**:
- Full executable-level determinism of `stock_bot` across process runs is still not exhaustively proven.
- Dense has no production seed API; deterministic Dense initialization currently requires explicit parameter override in tests.
- `Trainer::run` uses internal static thread-local shuffle RNG state and does not expose external seed control for shuffle order via CLI seed.
- Trainer writes `tests/results.csv`, adding file side effects for trainer-level repeated-run audits.

**Milestone 13H Conclusion**:
- Determinism confidence improved for the strongest currently controllable path.
- This does **not** prove exhaustive full-pipeline determinism for all runtime paths.

### **Milestone 13I: Results-File/Output-Path Hygiene** ✅ COMPLETE (July 4, 2026)

**Scope**: Small hygiene/refactor task to remove hardcoded source-tree result writes and make trainer output explicit/controlled, without changing ML behavior.

**Files Changed**:
- `include/model/trainer.h` (added optional `resultsFilePath` constructor parameter; empty path disables trainer CSV writing)
- `src/model/trainer.cpp` (removed hardcoded `tests/results.csv` writer; implemented configurable/disable-able output path)
- `main.cpp` (removed hardcoded truncation of `tests/results.csv`; added `--results-file` and `--no-results`; default output now `build/results/trainer_results.csv`)
- `.gitignore` (added generated output ignores: `data/results.csv`, `build/results/`, `build/test_outputs/`)
- `tests/results_file_hygiene_test.cpp` (NEW test covering disabled output and explicit safe-path output)
- `CMakeLists.txt` (added `results_file_hygiene_test` target, CTest registration, and `run_tests` dependency)
- `docs/BUILD.md` (documented trainer output controls and safe output paths)
- `docs/KNOWN_RISKS.md` (updated artifact-risk wording)
- `docs/REFACTOR_PLAN.md` (marked hygiene follow-up complete; updated next likely follow-up)
- `PROJECT_STATE.md` (this milestone note)

**No ML behavior changes**:
- No changes to LSTM/Dense implementations, optimizer math, loss formulas, scaling logic, validation split behavior, or metrics formulas.

**Result-writing behavior before**:
- `Trainer::run` always appended metrics to hardcoded `tests/results.csv`.
- `main.cpp` truncated `tests/results.csv` at startup.
- This produced source-tree side effects during runs/audits.

**Result-writing behavior after**:
- `Trainer` output is now explicit and controlled:
  - empty `resultsFilePath` disables trainer CSV output
  - non-empty `resultsFilePath` writes/appends there (parent directories auto-created)
- `main.cpp` defaults to build-local `build/results/trainer_results.csv`
- CLI options:
  - `--results-file PATH` to choose output file
  - `--no-results` to disable trainer CSV output
- Hyperparameter search behavior remains separate and unchanged: `gridSearch` still writes `data/results.csv`.

**Hygiene test coverage (`results_file_hygiene_test`)**:
1. Run trainer with output disabled and assert no modification to `tests/results.csv`.
2. Run trainer with explicit safe path (`build/test_outputs/results_file_hygiene.csv`) and assert file is created there.
3. Assert source-tree `tests/results.csv` snapshot remains unchanged.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
cd /home/caidenmarley/stock-bot && git status --short
```

**Results**:
- `ctest --test-dir build --output-on-failure`: ✅ **13/13 tests passed** (including `results_file_hygiene_test`)
- No source-tree generated results file modifications were introduced by tests in this task.

**Milestone 13I Conclusion**:
- Generated trainer results output is now explicit, controllable, and safer for tests/audits.
- Hyperparameter-search results remain a separate output path concern (`data/results.csv`) and should be treated as generated artifact output.

### **Milestone 13J: Focused AdaBelief Optimizer Coverage** ✅ COMPLETE (July 4, 2026)

**Scope**: Test/audit-first coverage of currently implemented `AdaBelief::update` behavior, without changing optimizer or model behavior.

**Files Changed**:
- `tests/adabelief_test.cpp` (NEW focused deterministic AdaBelief tests)
- `CMakeLists.txt` (added `adabelief_test` target, CTest registration, and `run_tests` dependency)
- `docs/BUILD.md` (added build/run references for `adabelief_test`)
- `docs/ML_CORRECTNESS.md` (added focused AdaBelief coverage section)
- `docs/KNOWN_RISKS.md` (updated optimizer-risk wording)
- `docs/REFACTOR_PLAN.md` (marked optimizer coverage follow-up complete and updated next likely follow-up)
- `PROJECT_STATE.md` (this milestone note)

**No production optimizer/model code changed**:
- No changes to `src/model/ada_belief.cpp`, LSTM/Dense/loss/metrics implementations, or trainer/main behavior.

**AdaBelief behaviors covered**:
1. Zero gradient update leaves parameters unchanged.
2. One-step deterministic update matches hand-computed implementation formula.
3. Opposite gradient signs move parameters in opposite directions.
4. Two-step constant-gradient update matches closed-form behavior derived from the current implementation.

**Implementation details explicitly validated in test expectations**:
- Bias correction for both first and second moments.
- Belief residual second moment uses `(g - m)^2`.
- Epsilon applied in denominator after square root.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target adabelief_test
cd /home/caidenmarley/stock-bot && ./build/adabelief_test
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
cd /home/caidenmarley/stock-bot && git status --short
```

**Results**:
- `./build/adabelief_test`: ✅ **4/4 PASSED**
- `ctest --test-dir build --output-on-failure`: ✅ **14/14 tests passed**

**Milestone 13J Conclusion**:
- Confidence improved for focused deterministic AdaBelief update mechanics in the currently implemented API.
- This does **not** prove full optimizer correctness across all hyperparameters, dimensionalities, clipping interactions, or full training-loop regimes.

### **Milestone 13K: Focused Trainer-Level Behavior Coverage** ✅ COMPLETE (July 4, 2026)

**Scope**: Add small deterministic Trainer-level black-box behavior coverage to verify coordination of existing data/model/loss/optimizer path without modifying training math or model behavior.

**Files Changed**:
- `tests/trainer_behavior_test.cpp` (NEW deterministic Trainer-level behavior test)
- `CMakeLists.txt` (added `trainer_behavior_test` target, CTest registration, and `run_tests` dependency)
- `docs/BUILD.md` (added build/run references for `trainer_behavior_test`)
- `docs/ML_CORRECTNESS.md` (added Trainer-level behavior coverage section)
- `docs/KNOWN_RISKS.md` (updated Trainer-level risk wording for covered scope)
- `docs/REFACTOR_PLAN.md` (marked Trainer-level behavior follow-up complete; updated next likely follow-up)
- `PROJECT_STATE.md` (this milestone note)

**No production trainer/model math changes**:
- No changes to Trainer training logic, LSTM/Dense/loss/metrics implementations, scaler/parser/StockData behavior, or optimizer formulas.

**Trainer behaviors covered**:
1. Tiny 1-epoch `Trainer::run(...)` on synthetic in-memory chronological data completes without throwing.
2. Returned `TrainingResult` is finite and epoch-bounded (`totalEpochs`, `epochOfBestValLoss`, `bestValLoss`).
3. Trainer output can be disabled via empty `resultsFilePath`.
4. Trainer output can be redirected to a safe build-local path (`build/test_outputs/trainer_behavior_results.csv`).
5. Source-tree outputs are guarded: `tests/results.csv` and `data/results.csv` remain unmodified by this Trainer test.

**Coverage boundaries / practical API limits**:
- Current Trainer API does not expose internal model parameters, so this test remains black-box and does not directly assert parameter deltas post-epoch.
- Validation non-shuffle behavior is validated at lower levels and by existing integration tests; this Trainer-level test focuses on practical run/output behavior.
- No claims are made about one-epoch loss improvement, long-run convergence, or profitability.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target trainer_behavior_test
cd /home/caidenmarley/stock-bot && ./build/trainer_behavior_test
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
cd /home/caidenmarley/stock-bot && git status --short
```

**Results**:
- `./build/trainer_behavior_test`: ✅ **2/2 PASSED**
- `ctest --test-dir build --output-on-failure`: ✅ **15/15 tests passed**
- No source-tree result file modifications observed for `tests/results.csv` or `data/results.csv` in this task.

**Milestone 13K Conclusion**:
- Confidence improved that Trainer can coordinate existing components in a tiny deterministic run while respecting output-path hygiene controls.
- This does **not** prove full Trainer correctness across all hyperparameters, data regimes, or long training behavior.

### **Milestone 13L: CLI Smoke Tests for stock_bot Executable** ✅ COMPLETE (July 5, 2026)

**Scope**: Add small smoke/integration coverage for the real `stock_bot` executable CLI with short safe runs and output-path hygiene checks.

**Files Changed**:
- `tests/cli_smoke_test.cpp` (NEW CLI smoke test invoking the built `stock_bot` executable)
- `CMakeLists.txt` (added `cli_smoke_test` target, CTest registration, `run_tests` dependency, and `cli_smoke_test -> stock_bot` target dependency)
- `docs/BUILD.md` (added `cli_smoke_test` build/run references and safe short CLI examples)
- `docs/KNOWN_RISKS.md` (updated CLI/output-side-effect risk wording for covered scope)
- `docs/REFACTOR_PLAN.md` (marked CLI smoke follow-up complete; updated next likely follow-up)
- `PROJECT_STATE.md` (this milestone note)

**Production-code change status**:
- No changes to ML math, training behavior, validation logic, model implementations, parser/scaler/StockData, or Trainer logic.
- No `main.cpp` change was required; one CMake wiring fix was made so `cli_smoke_test` reliably builds `stock_bot` first.

**CLI behaviors covered**:
1. Real executable minimal run exits successfully:
  - `stock_bot --epochs 1 --seed 0 --early-stop-patience 1 --no-results`
2. Real executable minimal run exits successfully with safe output path:
  - `stock_bot --epochs 1 --seed 0 --early-stop-patience 1 --results-file build/test_outputs/cli_smoke_results.csv`
3. Build-local results file is created and non-empty when `--results-file` is provided.
4. Source-tree output guards: `tests/results.csv` and `data/results.csv` are not created/modified by the smoke test runs.

**Observed limitation documented**:
- CLI smoke test currently depends on repository dataset presence because `stock_bot` uses fixed input path `data/AAAU.csv` and does not expose a CLI data-path argument.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target cli_smoke_test
cd /home/caidenmarley/stock-bot && ./build/cli_smoke_test
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
cd /home/caidenmarley/stock-bot && git status --short
```

**Results**:
- `./build/cli_smoke_test`: ✅ **1/1 PASSED**
- `ctest --test-dir build --output-on-failure`: ✅ **16/16 tests passed**
- No source-tree result-file modifications observed for `tests/results.csv` or `data/results.csv` in this task.

**Milestone 13L Conclusion**:
- Confidence improved for short executable-level CLI safety after result-path hygiene changes.
- This remains smoke-level coverage and does not prove full end-to-end CLI/behavior correctness across all options and runtime regimes.

### **Milestone 13M: Hyperparameter-Search Cleanup and Output Hygiene Audit** ✅ COMPLETE (July 5, 2026)

**Scope**: Audit and small cleanup of hyperparameter search behavior focused on output-path hygiene and API clarity, without changing ML training math.

**Files Changed**:
- `include/search/hyperparam_search.h` (added optional explicit `resultsFilePath` for `gridSearch`; documented `randomSearch` not-implemented behavior)
- `src/search/hyperparam_search.cpp` (made `gridSearch` output path configurable with safe build-local default; added explicit `randomSearch` stub that throws `std::logic_error`)
- `tests/hyperparam_search_test.cpp` (NEW focused tiny test for safe output path and randomSearch behavior)
- `CMakeLists.txt` (added `hyperparam_search_test` target, CTest registration, and `run_tests` dependency)
- `docs/BUILD.md` (documented hyperparameter-search output path and new test target)
- `docs/KNOWN_RISKS.md` (updated hyperparameter-search output-side-effect and randomSearch wording)
- `docs/REFACTOR_PLAN.md` (marked hyperparameter cleanup/audit complete; updated next likely follow-up)
- `PROJECT_STATE.md` (this milestone note)

**No ML behavior changes**:
- No changes to LSTM/Dense/optimizer/loss/math, parser/scaler/StockData behavior, validation logic, trainer internals, or main training flow.

**Hyperparameter-search behavior before**:
1. `gridSearch` always wrote to hardcoded `data/results.csv`.
2. Output path could not be configured from API.
3. `randomSearch` was declared in header but not defined in source.

**Hyperparameter-search behavior after**:
1. `gridSearch` accepts explicit `resultsFilePath` and now defaults to safe build-local `build/results/hyperparam_search_results.csv`.
2. `gridSearch` creates parent directories for output path as needed and throws clear error if output file cannot be opened.
3. `randomSearch` is now explicitly defined as unavailable and throws clear `std::logic_error("randomSearch is not implemented yet")`.

**Focused test coverage (`hyperparam_search_test`)**:
1. Tiny synthetic-data `gridSearch` run writes to explicit safe build-local path (`build/test_outputs/hyperparam_search_results.csv`).
2. Guard checks verify no modification to `tests/results.csv` or `data/results.csv`.
3. `randomSearch` call is verified to throw clear not-implemented error.

**Commands Used**:
```bash
cd /home/caidenmarley/stock-bot && cmake --build build --target hyperparam_search_test
cd /home/caidenmarley/stock-bot && ./build/hyperparam_search_test
cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests
cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure
cd /home/caidenmarley/stock-bot && git status --short
```

**Results**:
- `./build/hyperparam_search_test`: ✅ **2/2 PASSED**
- `ctest --test-dir build --output-on-failure`: ✅ **17/17 tests passed**
- No source-tree `tests/results.csv` or `data/results.csv` created/modified in this task.

**Milestone 13M Conclusion**:
- Hyperparameter-search output hygiene is improved with explicit configurable path and safe default.
- API behavior is clearer: `randomSearch` is now explicitly unavailable rather than silently undefined.
- This does not implement full random search or integrate search flow into main runtime path.

---

## Summary

**Current State**: 
- Core LSTM, training loop, and rolling validation are implemented and verified to run
  - **Milestone 2–13M Status**: Build/runtime/test milestones plus documentation extraction, refactor planning, CTest full-suite convenience updates, documentation cleanup, reproducibility verification, broader deterministic LSTM gradient coverage, integration-level validation coverage expansion, end-to-end determinism audit, results-file hygiene improvements, focused AdaBelief optimizer coverage, focused Trainer-level behavior coverage, executable CLI smoke coverage, and hyperparameter-search cleanup/audit are complete ✅
    - Milestone 2: CMake configured, both targets compiled cleanly
    - Milestone 3: Both executables run successfully, output is reasonable
    - Milestone 4: CSVLoader robustness verified (8 comprehensive parser tests all passed)
    - Milestone 5: RollingWindowScaler behavior verified (8 dedicated scaler tests all passed)
    - Milestone 6: StockData behavior verified (9 dedicated StockData tests all passed)
    - Milestone 7: Huber loss and metrics behavior verified (10 dedicated loss/metrics tests all passed)
    - Milestone 8: Dense backward gradient check verified (6 dedicated Dense gradient tests all passed)
    - Milestone 9: LSTM parameter ordering consistency verified (6/6 tests passed)
    - Milestone 10: LSTM tiny-case finite-difference gradient check verified (1 dedicated test passed)
    - Milestone 11: Training-loop review completed; no clear bug found for inspected paths (review-only, not exhaustive)
    - Milestone 12: Time-series validation review completed; no clear future-data leakage found for inspected paths (review + focused synthetic tests)
    - Milestone 13A: Documentation extraction completed (`docs/BUILD.md`, `docs/ARCHITECTURE.md`, `docs/ML_CORRECTNESS.md`, `docs/KNOWN_RISKS.md`)
    - Milestone 13B: Refactor planning completed (`docs/REFACTOR_PLAN.md`)
    - Milestone 13C: CTest full-suite runner completed (single-command `ctest --output-on-failure` workflow)
    - Milestone 13D: Documentation cleanup completed
    - Milestone 13E: Same-seed reproducibility verification completed (`reproducibility_test`)
    - Milestone 13F: Broader deterministic LSTM finite-difference gradient coverage completed (`lstm_gradient_test` expanded to 3 cases)
    - Milestone 13G: Integration-level deterministic validation path coverage completed (`integration_validation_test`)
    - Milestone 13H: End-to-end determinism audit completed for current supported scope (`end_to_end_determinism_test`)
    - Milestone 13I: Results/output-path hygiene completed (`results_file_hygiene_test` + configurable trainer output path)
    - Milestone 13J: Focused AdaBelief optimizer coverage completed (`adabelief_test`)
    - Milestone 13K: Focused Trainer-level behavior coverage completed (`trainer_behavior_test`)
    - Milestone 13L: CLI smoke coverage completed (`cli_smoke_test`)
    - Milestone 13M: Hyperparameter-search cleanup/audit completed (`hyperparam_search_test`)
  - Test suites now include `testbed`, `parser_test`, `rolling_window_scaler_test`, `stock_data_test`, `loss_metrics_test`, `dense_gradient_test`, `lstm_parameter_order_test`, `lstm_gradient_test`, `time_series_validation_test`, `reproducibility_test`, `integration_validation_test`, `end_to_end_determinism_test`, `results_file_hygiene_test`, `adabelief_test`, `trainer_behavior_test`, `cli_smoke_test`, and `hyperparam_search_test`
- LSTM backward/BPTT now has broader deterministic numerical verification across multiple configurations; coverage is improved but not exhaustive
- Integration-level validation coverage is improved for one deterministic cross-component path, but still not exhaustive
- Determinism coverage now includes controlled repeated-run path checks, but full executable-level determinism remains unverified
- Trainer output path side effects are reduced via configurable/disable-able CSV writing, with remaining generated-artifact risk for user-selected paths and hyperparameter search output
- AdaBelief behavior now has focused deterministic coverage, but optimizer correctness is still not exhaustively proven in full training contexts
- Trainer-level behavior now has focused black-box coverage for tiny-run coordination and output hygiene, but internal parameter-transition assertions are still limited by API visibility
- CLI smoke coverage now verifies short real-executable runs and output-flag hygiene (`--no-results`/safe `--results-file`) for covered commands, but broad CLI-option behavior remains unverified
- Hyperparameter search now has safe default output hygiene and explicit randomSearch unavailability, but search flow remains outside main runtime integration

**Main Risks**: 
- LSTM gradient verification is broader than before but still not exhaustive, plus residual validation leakage risk (reduced by Milestone 12 but not exhaustively eliminated)
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
9. ✅ Training loop review (Milestone 11) – **COMPLETE (June 26, 2026)**
10. ✅ Time-series validation review (Milestone 12) – **COMPLETE (June 26, 2026)**
11. ✅ Documentation extraction (Milestone 13A) – **COMPLETE (June 26, 2026)**
12. ✅ Refactor planning (Milestone 13B) – **COMPLETE (June 26, 2026)**
13. ✅ CTest full-suite runner (Milestone 13C) – **COMPLETE (June 26, 2026)**
14. Generated-file hygiene for `tests/results.csv` (index/untracking cleanup) – **COMPLETE**
15. ✅ Reproducibility test for same seed (Milestone 13E component-level scope) – **COMPLETE (June 26, 2026)**
16. ✅ Broader LSTM gradient coverage across additional shapes/sequences/loss setups (Milestone 13F) – **COMPLETE (July 4, 2026)**
17. ✅ Expand end-to-end validation/integration coverage for time-series integrity and training-path behavior (Milestone 13G) – **COMPLETE (July 4, 2026)**
18. ✅ Full end-to-end determinism audit for the numerical pipeline (Milestone 13H current supported scope) – **COMPLETE (July 4, 2026)**
19. ✅ Generated output path hygiene and Trainer result-file behavior audit (Milestone 13I) – **COMPLETE (July 4, 2026)**
20. ✅ Focused AdaBelief optimizer coverage (Milestone 13J) – **COMPLETE (July 4, 2026)**
21. ✅ Focused Trainer-level behavior coverage (Milestone 13K) – **COMPLETE (July 4, 2026)**
22. ✅ CLI-level smoke coverage for output flags (`--results-file` / `--no-results`) and executable-path output hygiene checks (Milestone 13L) – **COMPLETE (July 5, 2026)**
23. ✅ Hyperparameter-search output-path cleanup and API-clarity audit (Milestone 13M) – **COMPLETE (July 5, 2026)**
24. Dense initialization seeding/API design discussion for stronger end-to-end reproducibility controls without changing training math – **Next step**

This recovery approach prioritizes understanding and correctness before expansion to multi-model ensemble or web scraping.
