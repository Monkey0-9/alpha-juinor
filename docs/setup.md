# Deployment & Setup Guide

## 1. Environment Configuration
Create a `.env` file in the root directory:
```bash
# Broker Keys
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret

# Monitoring (Webhooks)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## 2. Hardened Dependencies
We use pinned versions to ensure no "FutureWarning" drift or API breakage:
```bash
pip install -r requirements.txt
```

## 3. Running Modes
### Backtest (Protected)
Runs the strategy against historical data with full hygiene checks.
```bash
python main.py --mode backtest
```

### Paper Trading (Autonomous)
Launches the 24/7 rebalance loop using the `MockBroker` or Alpaca Paper API.
```bash
python main.py --mode paper
```

### Live Trading (High Security)
**Requires manual string confirmation 'CONFIRM' before execution.**
```bash
python main.py --mode live
```

## 4. Production Observability
- **Heartbeats**: The system pings Discord/Slack every hour (or rebalance cycle).
- **Audit Logs**: Check `main.log` and `output/meta.json` for run IDs and git hashes.
- **Circuit Breakers**: Scaling factors will be logged automatically if risk limits are approached.
