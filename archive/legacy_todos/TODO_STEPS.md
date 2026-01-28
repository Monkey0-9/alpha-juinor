# OPS Stability Fixes Implementation Steps

## Primary Fixes (A-K)

### A. Fix SyntaxError in risk/engine.py
- [ ] Locate unmatched `)` near line ~298
- [ ] Correct function signature/parentheses
- [ ] Run py_compile to verify

### B. Resolve 'name 'logger' is not defined'
- [ ] Add `import logging; logger = logging.getLogger(__name__)` to every module that logs
- [ ] Add central logging config in main.py
- [ ] Replace bare logger references with module-level logger

### C. RoutingProvider.get_latest_price missing
- [ ] Create data/routing_provider.py or add to main.py's RoutingProvider
- [ ] Implement get_latest_price(self, symbol, timestamp=None)
- [ ] Delegate to provider-specific method or fallback to cached last close price
- [ ] Raise NoPriceAvailable exception if all fail
- [ ] Update callers to catch exceptions and use safe fallback

### D. Network & external provider resiliency
- [ ] Add requests.Session() with Retry adapter (3 retries, exponential backoff)
- [ ] Set timeouts (e.g., timeout=5)
- [ ] Catch RequestException, SSLError, gaierror
- [ ] Mark provider as disabled on persistent failure
- [ ] Continue pipeline with remaining sources

### E. Fix deprecated pandas fillna(method=...) and pct_change usage
- [ ] Replace df.fillna(method='ffill') / .fillna(method='bfill') with df.ffill() / df.bfill()
- [ ] Replace equity_curve.pct_change().fillna(0) with proper handling
- [ ] Run codebase to ensure no FutureWarning

### F. Vol estimation fallback spam
- [x] Add MIN_HISTORY_BARS constant
- [x] If history length < MIN, return FALLBACK_VOL
- [x] Log fallback once per run per symbol (use set or LRU cache)

### G. Empty equity / NaN metrics protections
- [ ] Check for equity_curve.empty or all-NaN before computing metrics
- [ ] Set Final Equity = starting_nav, Annualized Return = 0.0, Sharpe = None, Max Drawdown = 0.0%
- [ ] Log WARNING with helpful message
- [ ] Avoid nan outputs in final artifacts

### H. MarketListener anomaly cross-check robustness
- [x] Ensure cross-check uses at least two independent providers/time windows
- [x] Add configurable thresholds in configs/golden_config.yaml
- [x] Unit tests for glitch simulation
- [x] Mark anomalies as "INCONCLUSIVE" instead of "FAILED cross-check" if missing provider responses

### I. Add basic CI
- [ ] Add .github/workflows/ci.yml
- [ ] Run on PR: python 3.10/3.11, install requirements, pytest -q, flake8

### J. Secrets & logging hygiene
- [ ] Remove any API keys from code
- [ ] Replace with os.environ['API_KEY']
- [ ] Add .env.example
- [ ] Ensure logs don't print raw API keys/secrets

### K. Observability & guardrails
- [x] Add monitoring/health.py or extend monitoring.alerts
- [x] Expose provider health, last successful data timestamp, suppressed exceptions count
- [x] Add CIRCUIT_BREAKER_THRESHOLD in config
- [x] Pause trading loop and escalate if > X anomalies in Y minutes

## Implementation Steps
1. Create branch: fix/ops-stability-pp-20260106
2. Implement fixes A-K in atomic commits
3. Add unit tests for behavioral changes
4. Run pytest -q, flake8 .
5. Test python main.py startup
6. Open PR with checklist
