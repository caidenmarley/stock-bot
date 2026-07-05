# KNOWN_RISKS.md

## Time-Series and Validation Risks

- Time-series leakage risk is reduced by Milestone 12 checks plus Milestone 13G integration-path checks, but not exhaustively eliminated.
- Current-day scaling remains a modelling assumption for next-day prediction.
- Validation fold ranges are nested/overlapping by design; aggregate interpretation should account for fold dependence.
- End-to-end validation integrity across all scenarios is not exhaustively proven.

## ML Correctness Risks

- LSTM gradient verification now includes multiple deterministic finite-difference configurations, but is still not exhaustive.
- AdaBelief update mechanics are now covered for focused deterministic cases, but optimizer behavior is not exhaustively proven across all settings and training interactions.
- Gradient clipping policy is component-wise (LSTM vector and Dense parameters clipped separately), not one combined whole-model norm.
- Reproducibility is only partially verified: deterministic behavior is covered for a controlled in-memory path, but full executable-level determinism is not exhaustively proven.
- Dense deterministic initialization is now wired through Trainer/main when `--seed` is explicitly provided, but end-to-end reproducibility remains limited because trainer shuffle ordering is still not wired to CLI seed control.
- Trainer-level behavior is now covered for a tiny deterministic black-box run (including safe output-path handling), but full training-loop correctness and optimizer-policy interactions are still not exhaustively proven.

## Metrics and Evaluation Risks

- Trading-style metrics (Sharpe/PnL/turnover) are useful diagnostics but are not proof of profitability.
- Cost/slippage realism remains limited.

## Workflow and Artifact Risks

- Trainer CSV output path is now configurable/disable-able, reducing accidental source-tree writes.
- CLI smoke coverage now verifies short `stock_bot` runs with `--no-results` and with safe build-local `--results-file` path, plus source-tree result-file non-modification for those covered commands.
- Generated results files can still create noisy working-tree diffs if users point output to tracked source-tree paths.
- Hyperparameter search now defaults to build-local `build/results/hyperparam_search_results.csv` and supports explicit output-path selection, reducing source-tree output risk for covered paths.
- `randomSearch` remains intentionally unavailable (throws clear not-implemented error) and is not integrated in main flow.
- `stock_bot` currently uses fixed `data/AAAU.csv` path in `main.cpp` (no CLI data-path argument), so CLI smoke tests depend on repository-local dataset presence.

## Integration and Product Scope Risks

- Hyperparameter search exists but is not integrated into the main execution path.
- Sentiment model is not implemented.
- Source-reliability/text-quality model is not implemented.
- Ensemble model is not implemented.
- Web scraping pipeline is not implemented.
