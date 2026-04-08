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

# ── Status page ────────────────────────────────────────────────────────────
from fastapi.responses import HTMLResponse  # noqa: E402

_STATUS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Self-Evolving Trading Agent</title>
  <style>
    body{font-family:system-ui,sans-serif;max-width:680px;margin:60px auto;padding:0 20px;color:#1a1a2e}
    h1{font-size:1.6rem;margin-bottom:4px}
    .tag{display:inline-block;background:#22c55e;color:#fff;border-radius:4px;padding:2px 10px;font-size:.8rem;font-weight:600;margin-left:8px;vertical-align:middle}
    table{width:100%;border-collapse:collapse;margin:20px 0}
    th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #e5e7eb}
    th{background:#f9fafb;font-size:.85rem;color:#6b7280}
    code{background:#f3f4f6;padding:2px 6px;border-radius:3px;font-size:.9rem}
    a{color:#2563eb}
    .dim{color:#9ca3af;font-size:.85rem}
  </style>
</head>
<body>
  <h1>Self-Evolving Trading Agent <span class="tag">LIVE</span></h1>
  <p class="dim">OpenEnv environment · paper trading · adaptive reward weights</p>

  <table>
    <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
    <tr><td><code>/health</code></td><td>GET</td><td>Health check</td></tr>
    <tr><td><code>/reset</code></td><td>POST</td><td>Start new episode</td></tr>
    <tr><td><code>/step</code></td><td>POST</td><td>Send action, get observation</td></tr>
    <tr><td><code>/state</code></td><td>GET</td><td>Current portfolio state</td></tr>
    <tr><td><code>/schema</code></td><td>GET</td><td>Action / observation schemas</td></tr>
    <tr><td><code>/mcp</code></td><td>POST</td><td>MCP tool endpoint</td></tr>
    <tr><td><code>/docs</code></td><td>GET</td><td>Interactive API docs (Swagger)</td></tr>
  </table>

  <h3>Quick start</h3>
  <pre><code>from openenv.core import EnvClient
from models import TradeAction, MarketObservation

with EnvClient("https://vikramronavrsc-self-evolving-trading-agent.hf.space") as env:
    obs = env.reset()
    obs = env.step(TradeAction(action_type=1, size=0.5))  # buy 50%
    print(obs.portfolio_value, obs.reward)</code></pre>

  <p class="dim">
    <a href="/docs">Swagger UI</a> ·
    <a href="https://github.com/Lokiammu/Sivuku">GitHub</a>
  </p>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return HTMLResponse(_STATUS_HTML)


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
