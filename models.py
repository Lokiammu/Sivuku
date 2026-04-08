from __future__ import annotations

from typing import Optional

from openenv.core.env_server.types import Action, Observation, State


class TradeAction(Action):
    """Action the RL agent sends each step.

    action_type: 0=hold, 1=buy, 2=sell
    size: fraction of available capital (buy) or position (sell), 0.0–1.0
    """

    action_type: int = 0
    size: float = 0.5


class MarketObservation(Observation):
    """Observation returned to the agent each step."""

    # --- price features (100 floats: 20 candles × [o,h,l,c,v] normalised) ---
    ohlcv_window: list[float] = []

    # technical indicators (all normalised 0–1 except rsi which is 0–100)
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0

    # portfolio state
    portfolio_value: float = 10_000.0
    cash_ratio: float = 1.0          # cash / portfolio_value
    position_ratio: float = 0.0      # position_value / portfolio_value
    unrealized_pnl: float = 0.0      # as fraction of portfolio_value

    # context
    regime: int = 0                  # 0=sideways, 1=bull, 2=bear
    step_num: int = 0

    # human-readable trade outcome (shown in web interface)
    trade_info: str = ""
    error: Optional[str] = None

    # hackathon grader score in [0, 1] — set on episode completion
    task_score: Optional[float] = None


class PortfolioState(State):
    """Full internal state (WebSocket training infrastructure only)."""

    cash: float = 10_000.0
    shares: float = 0.0
    current_price: float = 0.0
    portfolio_value: float = 10_000.0
    initial_value: float = 10_000.0

    trade_history: list[dict] = []
    episode_returns: list[float] = []   # per-step portfolio returns

    current_ticker: str = "AAPL"
    current_data_idx: int = 0
    total_steps: int = 0
