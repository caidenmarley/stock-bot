# ML_CORRECTNESS.md

## Current Validation and Test Coverage

The following areas have dedicated coverage from milestones 4-12:
- Parser behavior (`parser_test`)
- Rolling scaler behavior (`rolling_window_scaler_test`)
- StockData batching and target alignment basics (`stock_data_test`)
- Huber loss and metrics formulas (`loss_metrics_test`)
- Dense gradients via finite differences (`dense_gradient_test`)
- LSTM parameter ordering/vector consistency (`lstm_parameter_order_test`)
- LSTM tiny-case BPTT finite-difference gradient check (`lstm_gradient_test`)
- Time-series validation checks (`time_series_validation_test`)

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

## LSTM Tiny-Case Gradient Check Summary

Milestone 10 added a deterministic tiny-shape LSTM BPTT finite-difference check:
- Small fixed shape and deterministic parameter vector
- Analytical gradient from reverse-time `backwardPass`
- Numerical central differences
- Tight absolute/relative error agreement in the tested case

Result for covered case: passed.

This is still limited-scope coverage and not exhaustive across shapes, sequence lengths, or loss setups.

## Time-Series Assumptions to Keep Documented

- Current-day scaling assumption: scaling is computed after adding the current day, so each day is scaled using current + past window context.
- Validation scaler past-context preload assumption: validation scaler is preloaded from training tail to provide context before validation-day processing.

Both assumptions are intentional in current design and should remain explicit in documentation.

## Confidence Statement

Tests and audits improve confidence in implementation behavior for inspected paths.
They do not prove complete mathematical correctness, complete leakage absence, or production readiness.
