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

## Current Build System

**CMake** with C++20 standard:
- `add_executable(stock_bot main.cpp ${SRC_FILES})` → main executable
- `add_executable(testbed tests/testbed.cpp ${SRC_FILES})` → testbed executable
- Requires Eigen3 (linear algebra library)
- Compiler flags: `-Wall -Wextra -Wpedantic -Werror` (strict warnings as errors)

**Build commands**:
```bash
mkdir build && cd build
cmake ..
cmake --build . --target stock_bot   # Main training executable
cmake --build . --target testbed     # Minimal test harness
```

---

## What Appears Complete

1. ✅ **CSV parsing** – CSVLoader handles OHLCV data efficiently
2. ✅ **Rolling-window scaling** – RollingWindowScaler works correctly for independent features
3. ✅ **Sequence construction** – StockData converts data into LSTM-ready batches
4. ✅ **LSTM forward pass** – Standard equations, Xavier init
5. ⚠️ **LSTM backward pass** – Unrolled BPTT for full sequences (implemented but not yet numerically verified)
6. ⚠️ **Dense output layer** – Simple linear regression with gradient accumulation (implemented but not yet numerically verified)
7. ⚠️ **Huber loss** – Forward and backward passes (implemented but not yet numerically verified)
8. ✅ **AdaBelief optimizer** – Parameter updates with bias correction
9. ✅ **Training loop** – Epoch iteration, batch processing, validation
10. ✅ **Learning rate decay** – Halves LR on validation plateau
13. ✅ **Gradient clipping** – Prevents exploding gradients (implementation present)
12. ✅ **Early stopping** – Halts if no improvement for N epochs
13. ✅ **Rolling validation folds** – 3 time-ordered splits (60%, 70%, 80%)
15. ⚠️ **Build system** – CMake configuration exists with Eigen3 dependency; actual build verification is Milestone 2 (not yet performed)
16. ✅ **Reproducibility** – Seed setting via `LSTMCell::setGlobalInitSeed()` (implementation present but full reproducibility not yet verified)

---

## What Appears Incomplete

1. ⚠️ **Testing** – Only one minimal test file (`testbed.cpp`) exists
   - Tests rolling scaler with synthetic data (5 days, window size 3)
   - No tests for parser, StockData, LSTM, Dense, Huber, or metrics
   - No numerical gradient checks for LSTM or Dense layers
   - No tests for data leakage or time-series integrity

2. ⚠️ **Documentation** – Limited inline comments
   - Headers have good docstrings, but implementation files may lack clarity
   - No walkthrough of the training flow
   - No explanation of gradient computation or parameter ordering

3. ⚠️ **ML correctness validation** – No numerical gradient checks
   - Dense layer: no finite-difference verification
   - LSTM: no gradient check for BPTT
   - Could catch bugs in backprop or parameter vector ordering

4. ⚠️ **Data leakage checks** – Validation scaler usage not verified
   - Does validation data use its own rolling scaler?
   - Does validation scaler peek at future validation data?
   - No tests confirm time-series integrity

5. ⚠️ **Integration tests** – No end-to-end test with real data
   - Only manual execution of `./stock_bot`
   - No CI/CD or automated test suite

6. ⚠️ **Hyperparameter search** – Exists but not connected to main flow
   - Functions in `search/hyperparam_search.h` defined but not called
   - No integration with `main.cpp`

7. ⚠️ **Multi-model ensemble** – Long-term goal, not implemented
   - Sentiment model: planned but not started
   - Source-reliability model: planned but not started
   - Ensemble layer: planned but not started

8. ⚠️ **Web scraping** – Planned for sentiment data, not started

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
  - Backward pass unrolls through sequence (implemented but not numerically verified)
  - Parameter vector ordering is assumed correct but unverified
- **Mitigation**: Add numerical gradient check (finite differences vs. analytical) as Milestone 10

### 3. **Parameter Vector Ordering** (High)
- **Risk**: `getParametersVector()`, `setParametersVector()`, `getGradientsVector()` must be perfectly aligned
- **Current Status**: 
  - Parameter flattening order is not documented
  - Implementation present but consistency not verified
  - No test verifies round-trip: get → set → get equality
- **Mitigation**: Add explicit ordering documentation and round-trip unit test as Milestone 9

### 4. **Dense Layer Gradient Accumulation** (High)
- **Risk**: Gradients might be summed instead of averaged, or vice versa
- **Current Status**: 
  - `Dense::backward()` accumulates via `dW += dLdy * h^T` (implemented)
  - Optimizer handles learning rate scaling
  - No numerical verification yet
- **Mitigation**: Add finite-difference gradient check as Milestone 8

### 5. **Validation Loss Interpretation** (High)
- **Risk**: Low validation loss does not guarantee profitability or realistic trading performance
- **Current Status**: 
  - Loss is reported as Huber loss (MSE-like)
  - Trading metrics (Sharpe, PnL, turnover) are implemented but not validated for realism
  - Transaction cost model is simple; slippage not simulated
- **Mitigation**: Document limitations and review cost assumptions in Milestone 12

### 6. **`std::string_view` Lifetime Issues** (Medium)
- **Risk**: `PriceData::date` references the CSV buffer; if buffer is moved/freed, view becomes invalid
- **Current Status**: 
  - CSVLoader stores buffer and data in the same object
  - Careful copying in StockData
  - No explicit test for lifetime correctness
- **Mitigation**: Add comments and tests confirming buffer pinning

### 7. **Eigen Tensor Memory Layout** (Medium)
- **Risk**: Row-major vs. column-major assumptions could cause subtle indexing bugs
- **Current Status**: 
  - StockData uses `Eigen::RowMajor` for consistency
  - No explicit documentation of memory layout expectations
- **Mitigation**: Add inline comments and a consistency test

### 8. **Random Seed and Reproducibility** (Medium)
- **Risk**: Seeding only affects LSTM weight init; other randomness might persist
- **Current Status**: 
  - `LSTMCell::setGlobalInitSeed()` controls Xavier initialization
  - AdaBelief determinism not verified
  - No full reproducibility test (same seed → exact same results)
- **Mitigation**: Add end-to-end reproducibility test

---

## Recommended Next Steps (Milestone Sequence)

Follow the **Codebase Recovery** milestones in order:

### **Milestone 2: Build Verification** (Next immediate step)
- [ ] Confirm current build workflow:
  - [ ] Run `cmake --build build --target stock_bot`
  - [ ] Run `./build/stock_bot` with small dataset
  - [ ] Document exact command, expected output, any warnings
- [ ] Document build dependencies (Eigen3 version, C++ compiler version)
- [ ] Create or update `docs/BUILD.md`

### **Milestone 3: Minimal Run Verification**
- [ ] Run with tiny configuration:
  ```bash
  ./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1
  ```
- [ ] Verify it completes without crash
- [ ] Verify output is reasonable (loss value, fold results)
- [ ] Document sample output

### **Milestone 4: Parser Tests**
- [ ] Add tests for CSVLoader:
  - Normal row parsing (6 columns + volume)
  - Header skip
  - Blank line handling
  - Windows vs. Unix line endings
  - Numeric precision (floats, uint64)
  - Error handling for malformed files
  - `std::string_view` lifetime (buffer not freed while views live)

### **Milestone 5: Rolling Scaler Tests**
- [ ] Verify scaling correctness:
  - First value after reset (should be 0 when stddev = 0 for uniform values)
  - Mean and stddev calculation accuracy
  - Window drop-off behavior (values entering and leaving the window)
  - Independence per feature
  - Confirmation: current day is scaled using window statistics, not future data

### **Milestone 6: StockData Tests**
- [ ] Verify sequence construction:
  - Window count calculation
  - Tensor shapes ([numWindows, sequenceLength, features])
  - Batch slicing ([batchSize, ...])
  - Partial batch handling
  - Target alignment (next-day return)
  - No temporal shuffling

### **Milestone 7: Loss and Metrics Tests**
- [ ] Test HuberLossFunction:
  - Forward pass (quadratic vs. linear terms)
  - Backward pass (gradient signs)
  - Batch averaging
- [ ] Test metrics:
  - Prediction-to-position conversion
  - Daily PnL calculation
  - Turnover calculation
  - Sharpe ratio (if applicable)

### **Milestone 8: Dense Layer Gradient Check**
- [ ] Numerical gradient check:
  - Compute analytical gradient from `Dense::backward()`
  - Compute numerical gradient via finite differences
  - Compare with tolerance (e.g., 1e-6)
  - Catches accumulation or sign errors

### **Milestone 9: LSTM Parameter Ordering Check**
- [ ] Verify parameter vector consistency:
  - Flatten all weights and biases in one order
  - Test `getParametersVector() → setParametersVector() → getParametersVector()`
  - Confirm byte-for-byte identical
  - Document the order explicitly in code

### **Milestone 10: LSTM Gradient Check**
- [ ] Minimal numerical gradient test:
  - Tiny LSTM: 2 features, 4 hidden, sequence length 3
  - Fixed seed
  - Compare analytical BPTT vs. finite-difference gradient
  - Catches backprop bugs early
- [ ] Expand to realistic size only after tiny case passes

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
  - [ ] Fold splits are correctly non-overlapping
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
- Core LSTM, training loop, and rolling validation are implemented
  - **Caveat**: Build and runtime not yet verified (Milestone 2)
- Single minimal test exists; no comprehensive test suite
- Critical components (LSTM backward, Dense gradients, metrics) implemented but not numerically verified
- Reproducibility via seeding is designed in but not yet fully verified

**Main Risks**: 
- Unverified LSTM gradients, parameter ordering, and data leakage
- Low test coverage leaves subtle bugs undetected
- Trading metrics are unvalidated and should not be interpreted as profit signals

**Next Actions** (in priority order):
1. Verify build and minimal execution (Milestone 2–3) – **START HERE**
2. Add parser and scaler tests (Milestone 4–5)
3. Add numerical gradient checks for Dense and LSTM (Milestone 8–10)
4. Verify parameter vector ordering consistency (Milestone 9)
5. Audit training loop and validation logic (Milestone 11–12)
6. Refactor and document (Milestone 13)

This recovery approach prioritizes understanding and correctness before expansion to multi-model ensemble or web scraping.
