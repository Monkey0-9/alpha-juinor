# High-Performance Stack (C++ + TypeScript)

This repository now includes a low-latency path:

- C++ decision core for fast signal ranking.
- TypeScript WebSocket gateway for live tick fanout.

## 1) Build C++ fast decision core

```powershell
python scripts/build_cpp_fastpath.py
```

After build, set:

```powershell
$env:MQF_CPP_CORE_LIB="C:\mini-quant-fund\cpp\fast_decision\build\Release\fast_decision_core.dll"
```

If not set, Python fallback scoring remains active.

## 2) Start TypeScript live-data gateway

```powershell
cd ts/live-data-gateway
npm install
npm run dev
```

Endpoints:

- Ingest: `ws://127.0.0.1:8787/ingest`
- Stream: `ws://127.0.0.1:8787/stream`
- Health: `http://127.0.0.1:8787/health`

## 3) Optional bridge from Alpaca to TypeScript gateway

```powershell
$env:PYTHONPATH="src"
python scripts/bridge_alpaca_to_ts_gateway.py
```

Optional env:

- `TS_GATEWAY_INGEST_URL` (default `ws://127.0.0.1:8787/ingest`)
- `ALPACA_STREAM_SYMBOLS` (default `AAPL,MSFT,TSLA`)
- `ALPACA_FEED` (default `iex`)

## 4) Python consumer for gateway stream

Use `mini_quant_fund.data.providers.ts_gateway_client.TSGatewayClient` to subscribe to gateway ticks from Python runtime.
