"""FastAPI app exposing the TradingEnvironment over HTTP + WebSocket."""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so flat imports (models, tasks) work.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openenv.core.env_server.http_server import create_app

try:
    from models import MarketObservation, TradeAction
except ModuleNotFoundError:
    from trading_env.models import MarketObservation, TradeAction

from server.trading_environment import TradingEnvironment


def _make_env() -> TradingEnvironment:
    return TradingEnvironment(
        ticker=os.environ.get("TRADING_TICKER", "AAPL"),
        interval=os.environ.get("TRADING_INTERVAL", "1d"),
        period=os.environ.get("TRADING_PERIOD", "5y"),
        initial_cash=float(os.environ.get("TRADING_INITIAL_CASH", "10000")),
        max_steps=int(os.environ.get("TRADING_MAX_STEPS", "500")),
    )


app = create_app(
    _make_env,
    TradeAction,
    MarketObservation,
    env_name="trading_env",
    max_concurrent_envs=1,
)


def main():
    """Entry point for ``openenv serve`` / ``uv run server``."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
