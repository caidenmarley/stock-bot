# AGENTS.md

## Project overview

This is a C++ stock prediction project built for learning machine learning from first principles and while the 
emphasis is on learning it should still be clean and efficient.

The current core model is a numerical stock-price model. It loads OHLCV-style CSV data, scales features with a rolling window, converts price history into sliding sequences, trains a custom LSTM, predicts next-day return, and evaluates validation loss plus trading-style metrics.

The long-term goal is to build a multi-model stock prediction system:

1. Numerical price-history model.
2. Sentiment-rating model from financial websites.
3. Text/source-reliability model using online opinions and historical source quality.
4. Ensemble layer that combines the models into buy/sell/hold signals.

The primary goal is deep understanding and correct implementation. The secondary goal is to improve realistic trading performance, but never by weakening validation, introducing data leakage, overfitting to historical results, ignoring transaction costs, or making unsupported profitability claims.

## Instruction priority

When instructions conflict, follow this priority:

1. Correctness and absence of data leakage.
2. Mathematical validity of the ML implementation.
3. Reproducible build and tests.
4. Code clarity and maintainability.
5. Runtime performance.
6. New features.
7. Profit-seeking experiments.

Profit-seeking experiments must never override correctness, validation integrity, or realistic evaluation.

## Core principles

- Prefer clear, educational but efficient C++ over black-box ML frameworks.
- Do not introduce TensorFlow, PyTorch, Keras, or similar ML frameworks.
- Eigen is allowed for linear algebra and tensors.
- Preserve the from-scratch nature of the LSTM, loss functions, optimiser, metrics, and data pipeline.
- Correctness comes first. Clarity comes second. Performance matters once behaviour is verified, especially in hot paths such as parsing, scaling, tensor batching, training loops, and metrics.
- Small, reviewable changes are preferred over large rewrites.
- Do not hide complexity from the user. Explain important ML, C++, and time-series decisions clearly.

## Current architecture

Expected major areas:

- `main.cpp`: command-line entry point and training/validation orchestration.
- `inputs/`: CSV parsing, rolling scaling, and conversion into model-ready data.
- `model/`: LSTM, dense output layer, optimiser, loss function, trainer, and metrics.
- `search/`: hyperparameter search.
- `data/`: local CSV datasets.
- `tests/`: test programs, test data, and result outputs.
- `agents/`: extra AI-agent instruction files.

## Build and run

Before changing code, inspect the existing build files and use the project's established build workflow.

Do not invent a new build system unless explicitly asked.

When modifying code:

1. Build the project.
2. Run relevant tests if they exist.
3. If no tests exist for the changed area, suggest or add a minimal test.
4. Report the exact command used and whether it passed or failed.

## Development workflow

For every task:

1. Inspect the relevant files first.
2. Summarise the current behaviour.
3. Identify the smallest safe change.
4. Make the change.
5. Build and test.
6. Explain what changed and why.
7. Mention any remaining risks or follow-up tasks.

Do not immediately rewrite large sections of the project.

## Safety rules for this project

- Be extremely careful about time-series data leakage.
- Do not shuffle validation or test data.
- Do not normalise validation/test data using future values.
- Do not use random train/test splits for time-series experiments unless the user explicitly asks for a non-time-series baseline.
- Be careful with `std::string_view` lifetimes.
- Be careful with Eigen object shapes, row-major/column-major assumptions, and tensor memory layout.
- Be careful with gradient signs and averaging across batches.
- Do not claim the model is profitable based only on validation loss.
- Do not treat Sharpe or PnL metrics as reliable until the evaluation method has been reviewed.

## ML correctness priorities

When reviewing or changing ML code, prioritise:

1. Target alignment.
2. Scaling and leakage.
3. Loss derivative signs.
4. Dense layer gradient correctness.
5. LSTM backpropagation through time.
6. Parameter-vector ordering.
7. Optimiser update correctness.
8. Batch gradient averaging.
9. Validation split correctness.
10. Reproducibility through seeds.

## C++ style

- Use modern C++17 or later if the project already supports it.
- Prefer explicit, readable code over clever tricks. However don't avoid using clever tricks if it increases 
speed, accuracy or efficiency, just clearly explain these when used.
- Avoid unnecessary heap allocations in hot loops, but do not sacrifice correctness, efficiency or speed.
- Keep ownership and lifetimes clear.
- Add comments for mathematical logic, not for obvious syntax.
- Avoid global state unless there is a clear reason.
- Keep headers minimal where practical.
- Preserve the existing folder structure unless asked to reorganise.

## What not to do

- Do not add web scraping yet unless the numerical model has been verified.
- Do not introduce a database unless explicitly asked.
- Do not add a GUI.
- Do not convert the project to Python.
- Do not replace the custom LSTM with a library implementation.
- Do not make performance claims without evidence.
- Do not delete existing code without explaining why.

## Reporting format

When finishing a task, report:

- Files inspected.
- Files changed.
- Build/test commands run.
- Whether the commands passed.
- What changed.
- Why it changed.
- Any risks or TODOs.
