# MODEL_IMPROVEMENT_PLAN.md

## MI-1 Audit Scope

This document is a documentation-only audit for Model Improvement Milestone MI-1.

Goal of this phase:
- Improve realistic out-of-sample trading performance under walk-forward evaluation.
- Keep validation integrity strict and avoid unsupported profitability claims.

This audit does not change production model behavior.

## 1) Current target/label

The current model target is next-day close-to-close return as a continuous regression label:

label_t = (Close_{t+1} - Close_t) / Close_t

This means each sequence predicts the percentage return for the day immediately after the last day in that sequence.

## 2) How the current target is constructed in StockData

Current construction logic in StockData:
- Build scaled feature rows for each day using RollingWindowScaler.
- Build sliding input windows with shape [numWindows, sequenceLength, numFeatures].
- For each window index seq:
  - lastDay = seq + sequenceLength - 1
  - close_t = rawData[lastDay].close
  - close_tp1 = rawData[lastDay + 1].close
  - target[seq] = (close_tp1 - close_t) / close_t

Interpretation:
- Inputs use days seq through seq + sequenceLength - 1.
- Target is the next day return right after that window.
- This is the expected alignment for one-step-ahead return prediction.

## 3) Alignment with long-term profitable buy/sell/hold objective

The current regression target is directionally relevant to trading, but only partially aligned with the long-term goal.

What aligns well:
- It predicts an economically meaningful quantity (future return).
- It supports deriving trading decisions from predicted sign/magnitude.

What is still missing for full alignment:
- Explicit modeling of transaction costs/slippage in the training objective.
- Explicit optimization for trade quality and turnover control.
- Explicit buy/sell/hold action objective (current pipeline is effectively long-or-flat in metrics conversion).
- Multi-horizon or risk-adjusted objectives that better reflect practical portfolio behavior.

Conclusion:
- Current target is a valid baseline target.
- It should be treated as a starting point, not the final profitability-aligned objective.

## 4) Current evaluation metrics that exist

From current Trainer and metrics code, the project reports:

Model-fit diagnostics:
- Validation Huber loss (used for early stopping and best-epoch tracking)
- MAE
- RMSE
- Directional accuracy (DA)

Trading-style diagnostics (validation predictions converted to positions):
- Net Sharpe (annualized)
- Average turnover

Underlying trading pipeline includes:
- Prediction threshold to enter long position
- Gross return, turnover, trading cost, and net return calculations

## 5) Which metrics are model diagnostics

Model diagnostics in this project:
- Validation loss (Huber)
- MAE
- RMSE
- Directional accuracy

These primarily indicate prediction error characteristics and sign agreement, not realized deployable strategy quality.

## 6) Which metrics are trading diagnostics

Trading diagnostics in this project:
- Sharpe computed from daily net returns after simple costs
- Daily gross return
- Daily net return
- Turnover

These are useful for evaluating decision-to-return behavior under assumptions, but they are still diagnostics rather than profitability proof.

## 7) Why validation loss alone is not enough for profitability

Lower validation loss is necessary for prediction quality, but not sufficient for trading profitability because:
- Loss does not directly encode transaction costs and slippage.
- Small error improvements may not improve trading decisions after thresholding.
- A model can have lower error but produce higher turnover and worse net returns.
- Validation loss can improve while edge is too weak to survive realistic execution friction.
- One-ticker/one-window improvements may not generalize out-of-sample.

Practical implication:
- Loss should be tracked, but profitability claims must rely on robust net-after-cost walk-forward diagnostics with benchmark comparison.

## 8) Current limitations in profitability evidence

Current evidence limitations:
- Validation windows are limited and overlapping; independence is limited.
- Current reports are mainly fold-level diagnostics, not robust multi-regime out-of-sample evidence.
- Metrics are produced on one fixed dataset path in main flow.
- No comprehensive benchmark panel is reported alongside model metrics in Trainer output.
- Current results do not establish durable edge across assets, periods, and market regimes.

Therefore, current outputs are progress indicators, not profitability confirmation.

## 9) Current transaction-cost/slippage limitations

Current trading-cost assumptions are simple:
- Fixed threshold-to-enter.
- Fixed linear cost per position change.
- Long-or-flat position mapping.

Current realism limitations:
- No explicit slippage model beyond simple cost proxy.
- No spread/latency/market-impact treatment.
- No instrument-specific fee model.
- No dynamic liquidity-aware cost scaling.

Implication:
- Net-return and Sharpe values are assumption-dependent diagnostics and can overstate real-world deployability.

## 10) Benchmark requirements for future performance experiments

For performance experiments or trading-metric reports, include at least:
- Cash baseline (always flat)
- Buy-and-hold baseline
- Random/no-skill baseline (with reproducible seed)
- Simple rule-based baseline (for example, momentum or moving-average threshold)

Benchmark rules:
- Use identical data splits, date ranges, and cost/slippage assumptions.
- Report model and benchmarks side-by-side.
- Do not claim model improvement unless it exceeds relevant baselines under the same assumptions.

## 11) Alternative labels to consider later

Candidates for later controlled experiments:
- Next-day return regression (current baseline)
- Direction classification (up/down)
- Thresholded buy/hold/sell classification
- Volatility-adjusted return target
- Multi-horizon returns (for example 1-day, 5-day, 10-day)
- Risk-adjusted target (for example return penalized by volatility/turnover proxy)

Guidance for label exploration:
- Change one label family at a time.
- Keep walk-forward protocol fixed while comparing labels.
- Evaluate against the required benchmark set.

## 12) Recommended next smallest code/test task

Recommended next smallest step after this audit:

Add a benchmark-evaluation utility and test coverage without changing training behavior.

Scope:
- Add a small evaluation module or helper that computes side-by-side validation diagnostics for:
  - current model output
  - cash
  - buy-and-hold
  - random/no-skill
  - one simple rule-based baseline
- Keep existing Trainer optimization and model behavior unchanged.
- Start with deterministic synthetic-data tests for metric consistency and benchmark comparability.

Why this is the next smallest step:
- It directly improves decision-quality evidence.
- It enforces benchmark discipline before architecture changes.
- It reduces risk of over-claiming based on loss or isolated Sharpe/PnL figures.

## Practical guardrail summary

The objective is not to maximize one metric on one short window.
The objective is to improve robust out-of-sample, walk-forward, net-after-cost diagnostics under realistic assumptions, while preserving leakage safety and reproducibility.

## MI-2 Implementation Note (Benchmark Helpers)

A small benchmark evaluation helper is now implemented in the metrics module and covered by focused tests.

Added benchmark helpers:
- cash baseline positions (always 0.0)
- buy-and-hold baseline positions (always 1.0)
- deterministic random/no-skill positions (explicit seed)
- previous-return momentum positions where position[t] uses return[t-1], never return[t]

Added benchmark summary evaluation:
- stable benchmark names for tracking
- sharpeNet after costs
- avgTurnover
- cumulativeNetReturn

Added model-vs-benchmark comparison helper:
- fixed row order: model, cash, buy_and_hold, random_noskill, prev_return_momentum
- shared schema: name, sharpeNet, avgTurnover, cumulativeNetReturn, numObservations
- model row uses existing threshold-to-position logic and the same PnL/cost path as benchmarks

Added compact serializer helpers:
- stable header: strategy,sharpe_net,avg_turnover,cumulative_net_return,num_observations
- one CSV row per comparison row
- fixed numeric formatting at 6 decimal places for floating-point fields
- preserves input row order
- empty input convention: header exists conceptually via helper, row serializer returns zero data rows

Added complete CSV block helper:
- returns one in-memory CSV string built from existing header + row serializers
- deterministic newline convention uses '\n'
- empty input convention is explicit: header plus trailing newline only
- no file writes and no recomputation/evaluation logic

Current convention note:
- helper size checks follow existing metrics conventions and rely on assert for prediction/return length mismatch
- this is acceptable for current internal/testing usage but is not a user-facing runtime error API

Design guardrail retained:
- benchmark helpers are evaluation utilities only
- no training behavior, target construction, optimizer, or validation split logic changed
