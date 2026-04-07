# Trading Env — Self-Evolving Paper Trader

Single-asset paper-trading environment on historical OHLCV (yfinance).

- **Action**: `TradeAction(action_type, size)` — `0=hold, 1=buy, 2=sell`, size ∈ [0,1]
- **Observation**: 20-candle OHLCV window + RSI/MACD/Bollinger + portfolio state + regime label
- **Reward**: `AdaptiveTradingRubric` — weights persisted in `rubric_config.json`, adjusted by an LLM critic between episodes
- **Starting capital**: $10,000 virtual

## Run locally

```bash
cd envs/trading_env
pip install -r server/requirements.txt
PYTHONPATH=. uvicorn server.app:app --port 8000
```

## Env variables

| Var | Default | Meaning |
|---|---|---|
| `TRADING_TICKER` | `AAPL` | yfinance ticker symbol |
| `TRADING_INTERVAL` | `1d` | candle interval (`1d`, `1h`, `15m`, ...) |
| `TRADING_PERIOD` | `5y` | history to download |
| `TRADING_INITIAL_CASH` | `10000` | starting virtual cash |
| `TRADING_MAX_STEPS` | `500` | episode length cap |
