# Live Data Gateway (TypeScript)

Low-latency WebSocket fanout service for live market ticks.

## Run

```powershell
cd ts/live-data-gateway
npm install
npm run dev
```

## Endpoints

- Ingest ticks: `ws://127.0.0.1:8787/ingest`
- Consume stream: `ws://127.0.0.1:8787/stream`
- Health: `http://127.0.0.1:8787/health`

## Tick payload format

```json
{
  "symbol": "AAPL",
  "price": 189.42,
  "bid": 189.41,
  "ask": 189.43,
  "volume": 200,
  "ts": 1710000000000,
  "source": "alpaca"
}
```

## Optional persistence

Set `TS_GATEWAY_PERSIST=1` to append all ticks to `runtime/live_ticks.jsonl`.
