# Institutional Hardening & Data Hygiene

This document outlines the mandatory patterns for data hygiene and numeric safety in the quantitative trading system.

## 1. Strict Returns Calculation
Always use `safe_pct_change` or the following pattern to avoid implicit forward-filling and handle infinities:

```python
returns = series.pct_change(fill_method=None)
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
results = pd.to_numeric(returns, errors='coerce').astype(float)
```

## 2. Safe Clipping & Alpha Guards
Alpha outputs must never contain NaNs or Infinities. Use `safe_clip` to ensure outputs are in the 0..1 range and handle empty inputs by returning aligned zeros.

```python
def safe_clip(raw: pd.Series, prices_index: pd.Index) -> pd.Series:
    raw = pd.to_numeric(raw, errors='coerce').astype(float)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return pd.Series(0.0, index=prices_index)
    return raw.clip(lower=0.0, upper=1.0)
```

## 3. Numeric Pipelines
After any transform (rolling, ewm, diff), ensure the result is numeric and clean:

```python
s = s.astype(float, copy=False)
s = s.replace([np.inf, -np.inf], np.nan).dropna()
```

## 4. Failure Modes
- **FutureWarnings**: Treated as errors in CI. Be explicit with pandas arguments.
- **Empty History**: Data layers must return aligned zero-series instead of crashing.
- **OHLC Inconsistency**: The `DataValidator` will drop malformed bars and log warnings.

## 5. Observability
Every run logs the count of dropped/cleaned bars per ticker. Monitor `main.log` for `[VALIDATOR]` tags.
