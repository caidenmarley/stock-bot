# REFACTOR_PLAN.md

## 1. Purpose

This document is a planning artifact for post-recovery cleanup and improvement work.
It does not assert that all listed refactors are mandatory, and it is not proof that current implementation is incorrect.

Each task should be executed in a small, reviewable, separate commit/branch with focused validation.

## 1.1 Recovery Status Handoff

- Recovery-phase work is complete through Milestone 13Q.
- Recovery completion means the project has build/test/documentation confidence for covered paths, not exhaustive correctness or profitability proof.
- Recovery milestones 2-13Q are now treated as completed baseline work.
- Planning focus moves to incremental model-improvement/research tasks.

## 1.2 Completed Recovery Track (Milestones 2-13Q)

- Build and run verification completed.
- Parser, scaler, stock-data, loss/metrics, dense/lstm gradient/order, and validation tests completed for covered paths.
- Integration, reproducibility, determinism, output-path hygiene, AdaBelief, trainer behavior, CLI smoke, hyperparameter-search hygiene, Dense seed path, Trainer/main seed plumbing, and invalid CLI-argument smoke coverage completed.
- Milestone 13Q marks the end of the recovery track.

## 1.3 Next-Phase Candidate Work (Model Improvement / Research)

Use small, test-backed experiments; avoid one large rewrite.

Agent routing note:
- Future model-improvement tasks should use `.github/agents/model-improvement.agent.md` rather than the recovery agent.

- target/label design review
- feature engineering
- walk-forward validation improvement
- training/evaluation experiment tracking
- transaction-cost/slippage realism
- model architecture improvements
- later sentiment/text-source model planning

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
- Status: Completed in Milestone 13M (`gridSearch` output-path cleanup + `randomSearch` explicit not-implemented behavior + `hyperparam_search_test`).

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

Status update:
- Completed in Milestone 13G by adding `tests/integration_validation_test.cpp` and wiring it into CTest/run_tests.

Reason:
- Reproducibility baseline checks are now in place for supported guarantees.
- Broader deterministic LSTM gradient coverage is now in place.
- Integration-level validation coverage is now broader for a deterministic cross-component path.
- Full end-to-end determinism audit is now completed in Milestone 13H (`tests/end_to_end_determinism_test.cpp`).
- Generated output path hygiene and Trainer result-file behavior audit is now completed in Milestone 13I.
- Focused AdaBelief optimizer coverage is now completed in Milestone 13J (`tests/adabelief_test.cpp`).
- Trainer-level behavior coverage for a controlled tiny run is now completed in Milestone 13K (`tests/trainer_behavior_test.cpp`).
- CLI-level smoke coverage for result-output flags (`--results-file` / `--no-results`) is now completed in Milestone 13L (`tests/cli_smoke_test.cpp`).
- Hyperparameter-search output-path cleanup and behavior audit are now completed in Milestone 13M (`tests/hyperparam_search_test.cpp`).
- Dense initialization seed/API design discussion is now completed in Milestone 13N (design-only docs audit, no code changes).
- Dense optional deterministic initialization implementation is now completed in Milestone 13O (`include/model/dense.h` + `tests/dense_seed_test.cpp`).
- Trainer/main seed plumbing pass is now completed in Milestone 13P (`main.cpp`, `trainer.*`, and `tests/seed_plumbing_test.cpp`).
- Invalid CLI argument handling smoke coverage is now completed in Milestone 13Q (`main.cpp` + `tests/cli_invalid_args_test.cpp`).
- Next likely low-risk follow-up: final recovery-summary consolidation and milestone handoff, or shift to model-improvement experiments under current safety constraints.

### Task: Implement Dense deterministic initialization API (follow-up to Milestone 13N design audit)
- Motivation: production same-seed reproducibility claims remain limited because Dense has no production seed path.
- Preferred design from audit: optional constructor seed parameter that preserves current behavior when omitted.
- Scope guardrails:
  - No ML math changes.
  - Preserve existing constructor behavior by default.
  - Keep implementation minimal and reviewable.
- Files likely touched: `include/model/dense.h`, Dense call sites in trainer/tests as needed, and determinism docs/tests.
- Risk level: Medium.
- Suggested validation command:
  - `cd /home/caidenmarley/stock-bot && cmake --build build --target run_tests`
  - `cd /home/caidenmarley/stock-bot && ctest --test-dir build --output-on-failure`
- Production code changes expected: Yes.
- Status: Completed in Milestone 13O with optional constructor seed parameter and focused `dense_seed_test` coverage.
