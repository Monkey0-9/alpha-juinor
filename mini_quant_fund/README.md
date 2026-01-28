# mini_quant_fund

Institutional-grade autonomous trading architecture.

## Architecture

- **Layer 0 (Data):** Multi-provider ingestion with validation (YFinance/AlphaVantage).
- **Layer 1 (Alpha):** Distributional agents (Momentum, Mean-Rev, Tail-Risk).
- **Layer 2 (Meta):** PM Meta-Brain with confidence-weighted aggregation and Kelly sizing.
- **Layer 3 (Execution):** Market impact models and institutional allocation.
- **Layer 4 (Verification):** Walk-forward simulator and audit trails.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install .
   ```

2. **Run satisfying a single cycle:**
   ```bash
   python -m mini_quant_fund.main --run-once --mode paper
   ```

3. **Run Tests:**
   ```bash
   pytest
   ```

## Docker

```bash
docker-compose up --build
```

## Features

- **Deterministic Logic:** All random seeds default to 42.
- **Data Quality:** Automatic 75% sizing reduction if quality score < 0.6.
- **Audit Trails:** JSONL logs per cycle in `logs/` directory.
- **Institutional Allocator:** Metadata-aware API for risk systems.
