# REFACTOR_PLAN.md

## 1. Purpose

This document is a planning artifact for post-recovery cleanup and improvement work.
It does not assert that all listed refactors are mandatory, and it is not proof that current implementation is incorrect.

Each task should be executed in a small, reviewable, separate commit/branch with focused validation.

## 2. Refactor Priorities

### Safety / Correctness
- Preserve time-series validation integrity assumptions while improving clarity.
- Expand correctness checks only in incremental, test-backed changes.

### Test Infrastructure
- Reduce test execution friction and improve repeatability.
- Improve reproducibility and regression safety over time.

### Documentation Cleanup
- Keep milestone tracking readable and avoid stale/duplicated checklist content.
- Keep assumptions and limits explicit for future contributors.

### Training-Loop Clarity
- Clarify optimizer and clipping policy decisions.
- Keep current behavior explicit before any behavioral changes.

### Evaluation/Metrics Clarity
- Keep trading metrics framed as diagnostics, not profitability proof.
- Clarify status of optional/public API items that are declared but not implemented.

### Future Feature Readiness
- Keep numerical model foundation stable while preparing for future sentiment/source/ensemble work.

## 3. Proposed Small Tasks

### Task: Untrack generated tests/results.csv if currently tracked
- Motivation: Generated training output can create noisy diffs and accidental commits.
- Files likely touched: `.gitignore` (if needed), git index state, documentation notes in `docs/BUILD.md` and/or `PROJECT_STATE.md`.
- Risk level: Low.
- Suggested validation command:
  - `git ls-files tests/results.csv`
  - `git status --short`
- Production code changes expected: No.

### Task: Add a simple top-level test command/target
- Status: Completed in follow-up infrastructure task (CTest registration + `run_tests` target).
- Motivation: Running all test executables manually is error-prone and slow for iteration.
- Files likely touched: `CMakeLists.txt` and `docs/BUILD.md`.
- Risk level: Low.
- Suggested validation command:
  - `cd build && cmake .. && cmake --build . --target run_tests`
  - `ctest --test-dir build --output-on-failure`
- Production code changes expected: No.

### Task: Clean up PROJECT_STATE.md numbering and stale duplicate checklist content
- Status: Completed in documentation cleanup follow-up (Milestone 13D).
- Motivation: Improves readability and reduces confusion while preserving evidence history.
- Files likely touched: `PROJECT_STATE.md`.
- Risk level: Low.
- Suggested validation command:
  - manual review + `git diff -- PROJECT_STATE.md`
- Production code changes expected: No.

### Task: Add explicit comments/docs for current-day scaling assumption
- Motivation: Current-day scaling is intentional but easy to misinterpret as leakage if undocumented near code.
- Files likely touched: `docs/ML_CORRECTNESS.md`, `docs/KNOWN_RISKS.md`, and optionally `src/inputs/stock_data.cpp` comments.
- Risk level: Low to Medium (if code comments are added).
- Suggested validation command:
  - docs-only: review diff
  - if code comments touched: `cd build && cmake --build . --target stock_data_test && cd .. && ./build/stock_data_test`
- Production code changes expected: Optional (comments only) or No if docs-only.

### Task: Add explicit comments/docs for validation scaler past-context preload
- Motivation: The preload behavior is central to leakage interpretation and should remain explicit.
- Files likely touched: `docs/ARCHITECTURE.md`, `docs/ML_CORRECTNESS.md`, and optionally `src/model/trainer.cpp` comments.
- Risk level: Low to Medium (if code comments are added).
- Suggested validation command:
  - docs-only review
  - if code comments touched: `cd build && cmake --build . --target time_series_validation_test && cd .. && ./build/time_series_validation_test`
- Production code changes expected: Optional (comments only) or No if docs-only.

### Task: Evaluate Dense optimizer choice (SGD-style) vs AdaBelief consistency
- Motivation: Current loop uses AdaBelief for LSTM and separate SGD-style updates for Dense; decision should be documented or intentionally revised later.
- Files likely touched: `docs/ARCHITECTURE.md`, `docs/ML_CORRECTNESS.md`, `docs/KNOWN_RISKS.md`, and potentially trainer/model files in a later behavioral change milestone.
- Risk level: Medium (High if optimizer behavior is changed).
- Suggested validation command:
  - docs-only: review
  - behavior change later: run full current test suite and smoke run (`./build/stock_bot --epochs 1 --seed 0 --early-stop-patience 1`).
- Production code changes expected: Not for documentation task; Yes only if behavior is intentionally changed later.

### Task: Evaluate clipping policy (component-wise vs model-wide)
- Motivation: Current clipping is separate for LSTM vector and Dense parameters; policy should be explicit before any change.
- Files likely touched: `docs/ML_CORRECTNESS.md`, `docs/KNOWN_RISKS.md`, and potentially trainer/model files in a later milestone.
- Risk level: Medium (High if clipping behavior changes).
- Suggested validation command:
  - docs-only: review
  - behavior change later: rerun gradient-related tests and smoke training run.
- Production code changes expected: Not for planning/docs; Yes only for later behavior change.

### Task: Add broader LSTM gradient coverage cases
- Status: Completed in Milestone 13F (`tests/lstm_gradient_test.cpp` expanded to multiple deterministic full-vector finite-difference cases).
- Motivation: Current tiny-case check is useful but limited; broader shapes/sequences reduce residual correctness risk.
- Files likely touched: new/expanded tests under `tests/` (future milestone).
- Risk level: Medium.
- Suggested validation command:
  - `cd build && cmake .. && cmake --build . --target lstm_gradient_test && cd .. && ./build/lstm_gradient_test`
- Production code changes expected: No (test expansion first).

### Task: Add reproducibility test for fixed seed
- Status: Completed for currently supported component-level guarantees (see `tests/reproducibility_test.cpp`).
- Motivation: Seed handling exists, but repeated-run equivalence has not been fully verified.
- Files likely touched: new test or script under `tests/` and possibly documentation.
- Risk level: Medium.
- Suggested validation command:
  - `cd build && cmake .. && cmake --build . --target reproducibility_test && cd .. && ./build/reproducibility_test`
  - `ctest --test-dir build --output-on-failure`
- Production code changes expected: No (test/script first).

### Task: Clarify hyperparameter search status and randomSearch decision
- Motivation: `gridSearch` exists but is not integrated; `randomSearch` is declared but may be unimplemented.
- Files likely touched: `docs/ARCHITECTURE.md`, `docs/KNOWN_RISKS.md`, and possibly `include/search/hyperparam_search.h` in a later API cleanup.
- Risk level: Low to Medium.
- Suggested validation command:
  - docs-only review
  - if header/API cleanup later: rebuild all targets that include search headers.
- Production code changes expected: Not for planning/docs; possible header/API cleanup later.

### Task: Decide whether predictionsToScaledPositions should be implemented or removed from public header
- Motivation: Public declaration without implementation can confuse consumers and future tests.
- Files likely touched: `include/model/metrics.h`, `src/model/metrics.cpp`, tests/docs depending on decision.
- Risk level: Medium.
- Suggested validation command:
  - `cd build && cmake .. && cmake --build . --target loss_metrics_test && cd .. && ./build/loss_metrics_test`
- Production code changes expected: Yes (if implementing/removing declaration).

### Task: Add optional CI workflow later
- Motivation: Automates regression checks and reduces manual verification burden.
- Files likely touched: `.github/workflows/` and docs.
- Risk level: Low to Medium.
- Suggested validation command:
  - lint/validate workflow syntax and run local equivalent test commands.
- Production code changes expected: No.

## 4. Recommended Next Actual Code Change

Recommended first follow-up task after planning:
- Expand end-to-end validation/integration coverage for time-series integrity and training-path correctness.

Reason:
- Reproducibility baseline checks are now in place for supported guarantees.
- Broader deterministic LSTM gradient coverage is now in place.
- The next likely high-value correctness step is deeper integration-level validation coverage while keeping changes test-first.
