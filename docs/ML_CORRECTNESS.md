# ML_CORRECTNESS.md

## Current Validation and Test Coverage

The following areas have dedicated coverage from milestones 4-13:
- Parser behavior (`parser_test`)
- Rolling scaler behavior (`rolling_window_scaler_test`)
- StockData batching and target alignment basics (`stock_data_test`)
- Huber loss and metrics formulas (`loss_metrics_test`)
- Dense gradients via finite differences (`dense_gradient_test`)
- LSTM parameter ordering/vector consistency (`lstm_parameter_order_test`)
- LSTM multi-case deterministic BPTT finite-difference gradient checks (`lstm_gradient_test`)
- Time-series validation checks (`time_series_validation_test`)
- Integration-level deterministic validation path checks (`integration_validation_test`)
- Reproducibility checks for currently supported seed/path guarantees (`reproducibility_test`)
- End-to-end determinism audit checks for currently controllable pipeline path (`end_to_end_determinism_test`)
- Focused AdaBelief optimizer behavior checks (`adabelief_test`)
- Focused Trainer-level behavior checks for tiny deterministic coordination path (`trainer_behavior_test`)

Coverage improves confidence for inspected paths and tested cases, but does not prove complete correctness.

## LSTM Parameter Ordering

Observed flattened ordering used by tests and implementation:

```text
Wf, Uf, bf, Wi, Ui, bi, Wc, Uc, bc, Wo, Uo, bo
```

This ordering passed covered consistency checks in Milestone 9.

## Dense Gradient Check Summary

Milestone 8 added deterministic finite-difference checks for Dense:
- Forward equation check (`y = W dot h + b`)
- Analytical backward check (`dW`, `db`)
- Numerical gradient comparisons for each weight and bias
- Gradient accumulation and `zeroGrad` behavior

Result for covered case: passed.

## LSTM Gradient Check Summary (Broadened Deterministic Coverage)

Milestone 10 introduced the initial tiny deterministic LSTM BPTT finite-difference check.
Milestone 13F expanded this into multiple deterministic configurations while keeping central finite differences and full-vector comparisons:

- Case A (original tiny case): `inputSize=2`, `hiddenSize=4`, `sequenceLength=3`, hidden-state loss
- Case B (short sequence): `inputSize=3`, `hiddenSize=2`, `sequenceLength=1`, hidden-state loss
- Case C (longer sequence + cell term): `inputSize=1`, `hiddenSize=3`, `sequenceLength=5`, hidden-state loss plus weighted final-cell loss

For each case:
- Analytical gradient is computed from reverse-time `backwardPass`
- Numerical gradient uses central finite differences over all parameters
- Comparison checks the full gradient vector (not a single-index subset)
- Tolerances remain conservative and unchanged: `epsilon=1e-5`, `absTol=1e-4`, `relTol=1e-3`

Result for covered cases: passed.

Coverage is broader than a single tiny case, but still not exhaustive across all shapes, sequence lengths, objectives, and training-loop contexts.

## Time-Series Assumptions to Keep Documented

- Current-day scaling assumption: scaling is computed after adding the current day, so each day is scaled using current + past window context.
- Validation scaler past-context preload assumption: validation scaler is preloaded from training tail to provide context before validation-day processing.

Both assumptions are intentional in current design and should remain explicit in documentation.

## Integration Validation Coverage (Milestone 13G)

Milestone 13G adds a small deterministic integration test that exercises a cross-component path without long training:

- Synthetic chronological OHLCV-like data split into time-ordered train and validation segments
- Validation scaler preloaded only from past training tail context
- Validation rows processed sequentially in `StockData`
- Leakage-style guard where future validation rows are perturbed and early validation-window scaled features are verified unchanged
- Deterministic LSTM + Dense forward pass on validation batch windows
- Huber loss computation on predicted values versus `StockData` targets
- Deterministic replay check (same parameters + reset path + same inputs -> same outputs)

This increases integration confidence for the covered path, but does not prove full end-to-end correctness of all training/validation scenarios.

## Reproducibility Verification (Current Scope)

The reproducibility test currently verifies these component-level guarantees:
- Same `LSTMCell::setGlobalInitSeed(seed)` + same LSTM dimensions -> identical initial LSTM parameter vector.
- Different LSTM seeds -> different initial LSTM parameter vectors for tested cases.
- Same LSTM parameters + same fixed input sequence -> identical forward outputs (deterministic forward path after reset).

What this does not prove yet:
- Full end-to-end `stock_bot` run determinism across process runs.
- Determinism of all randomness across the entire training stack.

In particular, current seeding in `main.cpp` is wired to LSTM initialization, while other components (such as Dense initialization path) are not explicitly tied to the same seed path in the current implementation.

## End-to-End Determinism Audit (Milestone 13H)

Milestone 13H adds a deterministic audit test for the strongest currently controllable in-memory numerical path:

- Same-seed repeated runs over synthetic chronological train/validation data
- Rolling scaling and `StockData` construction for train and validation paths
- Deterministic `nextBatchShuffled` behavior under fixed explicit order
- LSTM initialization controlled by `LSTMCell::setGlobalInitSeed(...)`
- Dense weights/bias explicitly overridden to deterministic values in test scope
- One tiny deterministic training-style update (Huber backward + Dense backward + LSTM backward + AdaBelief/SGD update)
- Equality checks before and after update (predictions, losses, and updated parameters)

Audit result for covered path: passed.

Important boundary conditions:

- This does not prove full executable-level determinism for `stock_bot`.
- Dense does not currently expose a production seed API, so deterministic Dense initialization is test-controlled by explicit parameter override.
- `Trainer::run` uses an internal static thread-local shuffle RNG seeded to a fixed value and not wired to CLI seed, limiting externally controlled same-seed reproducibility claims for training-order behavior.
- Trainer writes `tests/results.csv`, so trainer-level repeated-run checks include file side effects unless isolated.

## AdaBelief Coverage (Milestone 13J)

Milestone 13J adds focused deterministic tests for the currently implemented `AdaBelief::update` behavior:

- Zero-gradient update leaves parameters unchanged.
- One-step update matches hand-computed values from the implementation formula (including bias correction and epsilon placement).
- Opposite gradient signs move parameters in opposite directions.
- Two-step constant-gradient update matches a closed-form derivation consistent with the current implementation.

Implementation-specific notes verified by tests:

- Bias correction is applied to both first and second moments.
- Second moment tracks belief residual `(g - m)^2`, not plain `g^2`.
- Epsilon is added in the denominator after square root.

Coverage is still limited to small deterministic vectors and does not prove optimizer correctness for all dimensionalities, all hyperparameter settings, or full training-loop interactions.

## Trainer-Level Behavior Coverage (Milestone 13K)

Milestone 13K adds a focused deterministic Trainer-level smoke/behavior test without changing Trainer math or model implementations:

- Tiny 1-epoch Trainer run using synthetic in-memory chronological OHLCV-like data only.
- Existing production path exercised through `Trainer` construction and `Trainer::run(...)` with LSTM, Dense, Huber, AdaBelief, and StockData components.
- Returned `TrainingResult` is validated for finite/non-sentinel best validation loss and epoch bounds.
- Output disabled path is validated (empty `resultsFilePath`).
- Output redirect path is validated to build-local `build/test_outputs/trainer_behavior_results.csv`.
- Source-tree artifact guard verifies no modification to `tests/results.csv` or `data/results.csv`.

Coverage boundary for this milestone:

- This is black-box Trainer behavior coverage; it does not introspect private model parameters from `Trainer`.
- It does not prove validation-loss improvement trends, long-run convergence, or profitability.
- It does not prove full trainer determinism under all runtime/process conditions.

## Confidence Statement

Tests and audits improve confidence in implementation behavior for inspected paths.
They do not prove complete mathematical correctness, complete leakage absence, or production readiness.
