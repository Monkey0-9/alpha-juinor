# Unified Research Environment

This directory is the sandbox for Quantitative Researchers.
It is designed to be a seamless mirror of the production environment, but with read-only access to data.

## Workflow

1. **Hypothesis**: Create a new notebook/script in `alpha_research/`.
2. **Data**: Access `data/parquet` or `data/alternative` loaders.
3. **Simulation**: Use `backtest.engine` with `RealisticExecutionHandler`.
4. **Promotion**: Once an alpha is proven (Sharpe > 2.0, Low Correlation), move logic to `alpha_families/`.

## Directory Structure

- `alpha_research/`: Notebooks and experimental scripts.
- `notebooks/`: Jupyter notebooks.
- `models/`: Saved ML models (onnx/pkl).
