# ARCHITECTURE.md

## Project Goal

This project is a C++ stock prediction system built for ML learning from first principles, with emphasis on correctness and explicit implementation details.

## Current Scope

Current implemented scope is the numerical price-history model only:
- LSTM-based next-day return prediction from OHLCV sequences
- Training/validation loop with time-ordered fold splits
- Evaluation metrics for validation analysis

The broader multi-model system is not implemented yet.

## High-Level Data Flow

```text
CSVLoader
  -> RollingWindowScaler
  -> StockData
  -> LSTMCell
  -> Dense
  -> HuberLossFunction
  -> AdaBelief (for LSTM parameter updates)
  -> metrics (validation reporting)
```

## Module Responsibilities

- `inputs/parser.*` (`CSVLoader`): Parse OHLCV CSV rows into `PriceData`.
- `inputs/rolling_window_scaler.*` (`RollingWindowScaler`): Per-feature rolling mean/std scaling.
- `inputs/stock_data.*` (`StockData`): Build sliding windows, targets, and batches.
- `model/lstm.*` (`LSTMCell`): Sequence model forward/BPTT backward.
- `model/dense.h` (`Dense`): Linear output layer.
- `model/huber_loss_function.*` (`HuberLossFunction`): Robust regression loss.
- `model/ada_belief.*` (`AdaBelief`): Optimizer used for LSTM parameter vector updates.
- `model/metrics.*`: Validation-side trading-style metrics.
- `model/trainer.*` (`Trainer`): Orchestrates training/validation control flow.
- `search/hyperparam_search.*`: Search utilities (not integrated in main flow).

## Training vs Validation Flow

Training path (per epoch):
- Shuffled sequence order (`nextBatchShuffled`)
- Forward pass -> loss -> backward
- Parameter updates for LSTM (AdaBelief) and Dense (SGD-style)
- Gradient clipping before updates

Validation path (per epoch):
- Sequential batches (`nextBatch`)
- Forward/loss/metrics only
- No optimizer updates

Validation preprocessing uses past-context scaler preload from the training tail, then processes validation rows in order.

## Not Implemented Yet

The long-term roadmap components below are still pending:
- Sentiment model from financial websites
- Source-reliability / text-quality model
- Ensemble decision layer combining models
- Web scraping pipeline for sentiment/text sources
