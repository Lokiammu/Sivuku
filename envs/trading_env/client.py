"""Trading environment client.

Talks to the FastAPI+WebSocket server exposing TradingEnvironment.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import MarketObservation, PortfolioState, TradeAction


class TradingEnv(EnvClient[TradeAction, MarketObservation, PortfolioState]):
    """Client for the self-evolving trading environment.

    Example:
        >>> with TradingEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset().observation
        ...     obs = env.step(TradeAction(action_type=1, size=0.5)).observation
    """

    def _step_payload(self, action: TradeAction) -> Dict:
        return {
            "action_type": int(action.action_type),
            "size": float(action.size),
        }

    def _parse_result(self, payload: Dict) -> StepResult[MarketObservation]:
        obs_data = payload.get("observation", {}) or {}
        observation = MarketObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            ohlcv_window=obs_data.get("ohlcv_window", []),
            rsi=obs_data.get("rsi", 50.0),
            macd=obs_data.get("macd", 0.0),
            macd_signal=obs_data.get("macd_signal", 0.0),
            bb_upper=obs_data.get("bb_upper", 0.0),
            bb_lower=obs_data.get("bb_lower", 0.0),
            bb_mid=obs_data.get("bb_mid", 0.0),
            portfolio_value=obs_data.get("portfolio_value", 10_000.0),
            cash_ratio=obs_data.get("cash_ratio", 1.0),
            position_ratio=obs_data.get("position_ratio", 0.0),
            unrealized_pnl=obs_data.get("unrealized_pnl", 0.0),
            regime=obs_data.get("regime", 0),
            step_num=obs_data.get("step_num", 0),
            trade_info=obs_data.get("trade_info", ""),
            error=obs_data.get("error"),
            task_score=obs_data.get("task_score"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> PortfolioState:
        return PortfolioState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            cash=payload.get("cash", 10_000.0),
            shares=payload.get("shares", 0.0),
            current_price=payload.get("current_price", 0.0),
            portfolio_value=payload.get("portfolio_value", 10_000.0),
            initial_value=payload.get("initial_value", 10_000.0),
            trade_history=payload.get("trade_history", []),
            episode_returns=payload.get("episode_returns", []),
            current_ticker=payload.get("current_ticker", "AAPL"),
            current_data_idx=payload.get("current_data_idx", 0),
            total_steps=payload.get("total_steps", 0),
        )
