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
- Reproducibility checks for currently supported seed/path guarantees (`reproducibility_test`)

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

## Reproducibility Verification (Current Scope)

The reproducibility test currently verifies these component-level guarantees:
- Same `LSTMCell::setGlobalInitSeed(seed)` + same LSTM dimensions -> identical initial LSTM parameter vector.
- Different LSTM seeds -> different initial LSTM parameter vectors for tested cases.
- Same LSTM parameters + same fixed input sequence -> identical forward outputs (deterministic forward path after reset).

What this does not prove yet:
- Full end-to-end `stock_bot` run determinism across process runs.
- Determinism of all randomness across the entire training stack.

In particular, current seeding in `main.cpp` is wired to LSTM initialization, while other components (such as Dense initialization path) are not explicitly tied to the same seed path in the current implementation.

## Confidence Statement

Tests and audits improve confidence in implementation behavior for inspected paths.
They do not prove complete mathematical correctness, complete leakage absence, or production readiness.
