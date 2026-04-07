"""FastAPI app exposing the TradingEnvironment over HTTP + WebSocket."""

import os

from openenv.core.env_server.http_server import create_app

try:
    from ..models import MarketObservation, TradeAction
    from .trading_environment import TradingEnvironment
except ModuleNotFoundError:
    from models import MarketObservation, TradeAction
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


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
