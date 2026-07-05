# ML_CORRECTNESS.md

## Current Validation and Test Coverage

The following areas have dedicated coverage from milestones 4-13O:
- Parser behavior (`parser_test`)
- Rolling scaler behavior (`rolling_window_scaler_test`)
- StockData batching and target alignment basics (`stock_data_test`)
- Huber loss and metrics formulas (`loss_metrics_test`)
- Dense gradients via finite differences (`dense_gradient_test`)
- Dense optional deterministic initialization behavior (`dense_seed_test`)
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

In particular, current seeding in `main.cpp` is wired to LSTM initialization and now also into Trainer Dense initialization path when `--seed` is provided.

Dense now supports optional deterministic initialization via constructor seed parameter, but Trainer/main seed plumbing is not yet fully wired for broad production same-seed determinism claims.

## End-to-End Determinism Audit (Milestone 13H)

Milestone 13H adds a deterministic audit test for the strongest currently controllable in-memory numerical path:

- Same-seed repeated runs over synthetic chronological train/validation data
- Rolling scaling and `StockData` construction for train and validation paths
- Deterministic `nextBatchShuffled` behavior under fixed explicit order
- LSTM initialization controlled by `LSTMCell::setGlobalInitSeed(...)`
- Dense initialized through optional seeded constructor in test scope
- One tiny deterministic training-style update (Huber backward + Dense backward + LSTM backward + AdaBelief/SGD update)
- Equality checks before and after update (predictions, losses, and updated parameters)

Audit result for covered path: passed.

Important boundary conditions:

- This does not prove full executable-level determinism for `stock_bot`.
- Dense now exposes optional deterministic initialization through constructor seed parameter.
- `Trainer::run` uses an internal static thread-local shuffle RNG seeded to a fixed value and not wired to CLI seed, limiting externally controlled same-seed reproducibility claims for training-order behavior.
- Trainer writes `tests/results.csv`, so trainer-level repeated-run checks include file side effects unless isolated.

## Trainer/Main Seed Plumbing Coverage (Milestone 13P)

Milestone 13P wires available deterministic initialization paths more cleanly through production entry points:

- `main.cpp --seed` now controls:
	- LSTM initialization via existing `LSTMCell::setGlobalInitSeed(...)`
	- Dense initialization in Trainer path via optional Dense constructor seed parameter
- Default behavior is preserved when no seed is supplied.

Focused covered behavior:

- `seed_plumbing_test` runs short CLI executions and verifies same-seed runs produce identical `[SUMMARY]` output in covered configuration.
- Unseeded CLI short run still constructs/runs successfully.

Important limit retained:

- Trainer shuffle ordering still uses internal static thread-local RNG (`std::mt19937(42)`) not externally controlled by CLI seed.
- Full executable-level determinism for all runtime conditions remains unproven.

## Dense Seeding/API Design Audit (Milestone 13N)

Milestone 13N is a design-only audit (no code changes) on how to make Dense initialization deterministic in production paths.

Current Dense determinism situation:

- Dense initializes weights with `Eigen::RowVectorXd::Random(hiddenSize) * 0.01` and bias to `0.0`.
- Dense currently has no constructor seed parameter and no global seed hook.
- Dense parameters are publicly accessible (`W`, `b`), so tests can override values directly.
- Determinism tests currently rely on direct test-side Dense overrides for controlled-path checks.

Design options considered:

- Option A: add optional constructor seed parameter to Dense.
	- Example shape: `Dense(int hiddenSize, std::optional<uint32_t> initSeed = std::nullopt)`.
	- Preserve current behavior when no seed is provided.
	- Use local RNG path only when seed is supplied.
- Option B: add global/static seed path similar to `LSTMCell::setGlobalInitSeed(...)`.
	- Consistent with existing LSTM style, but introduces/extends hidden global state.
- Option C: add explicit initializer/setter API for initialization path.
	- Could be `setParameters(...)` or `initializeFromSeed(...)` style, while keeping constructor unchanged.
	- More explicit but requires extra call sites and lifecycle discipline.
- Option D: leave Dense unchanged and continue test-side deterministic overrides only.
	- Lowest implementation effort, but keeps production same-seed determinism claims limited.

Recommended direction for a future implementation milestone:

- Prefer Option A (optional constructor seed parameter) as the smallest practical production-path improvement.
- Rationale:
	- Minimal API disruption: default path preserves current behavior.
	- Better testability and cleaner Trainer/main propagation of seed values.
	- Avoids adding new hidden global state.
	- Keeps room for future alignment with LSTM seeding docs without forcing global seeding.

## Dense Optional Seed Implementation Coverage (Milestone 13O)

Milestone 13O implements the preferred Dense seed design with minimal API disruption:

- Dense now supports `Dense(hiddenSize, std::optional<uint32_t> initSeed)`.
- Existing `Dense(hiddenSize)` behavior remains available and unchanged for call sites that do not provide a seed.
- Dense forward/backward/gradient accumulation math is unchanged.

Focused coverage added in `dense_seed_test`:

- same seed + same dimensions -> identical initial `W` and `b`
- different seeds -> different initial `W` in tested case
- unseeded constructor remains valid (shape and zero-bias/grad init checks)
- seeded Dense forward is repeatable for same input

This improves reproducibility control for covered Dense initialization paths, but does not alone prove full executable-level determinism.

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
