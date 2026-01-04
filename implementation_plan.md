# Implementation Plan - Latency Optimization (Pre-Slicing)

## Goal
Reduce signal generation latency further by minimizing DataFrame slicing overhead.

## Proposed Changes
### `strategies/institutional_strategy.py`
-   **Pre-Slicing**: Move the time-window slicing (`iloc[-60:]`) *outside* the ticker loop.
    -   Current: Loop -> `market_data[tk]` (Full History) -> `iloc[-60:]`.
    -   New: `window = market_data.iloc[-60:]` -> Loop -> `window[tk]`.
-   This reduces the data access cost significantly, especially for longer history.

## Verification
-   **Benchmark**: Run `tests/benchmark_pipeline.py`.
    -   Expect avg latency to drop or stay stable (already low 40ms, might shave off 5-10ms).
-   **Correctness**: Run `tests/test_institutional_full.py`.
