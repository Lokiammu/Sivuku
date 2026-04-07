"""Trading environment — typed OpenEnv Environment.

Conforms to the dual API invariant: rewards are computed inside step() via the
adaptive rubric. The agent sends TradeAction; the env returns MarketObservation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

# These imports work because envs/ is on PYTHONPATH and rubrics/ is at repo root.
from trading_env.models import MarketObservation, PortfolioState, TradeAction

from .market_sim import MarketSimulator, Portfolio

logger = logging.getLogger(__name__)

# rubric_config.json lives at repo root so the dashboard + critic can all see it
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "rubric_config.json"


def _make_rubric():
    """Load the adaptive rubric lazily so the env file doesn't hard-depend on it."""
    try:
        from rubrics.trading_rubric import AdaptiveTradingRubric
        return AdaptiveTradingRubric(config_path=_DEFAULT_CONFIG_PATH)
    except Exception as e:
        logger.warning("Could not load AdaptiveTradingRubric: %s — falling back to PnL only", e)
        return None


class TradingEnvironment(Environment[TradeAction, MarketObservation, PortfolioState]):
    """Single-asset paper-trading environment over historical OHLCV."""

    def __init__(
        self,
        ticker: str = "AAPL",
        interval: str = "1d",
        period: str = "5y",
        initial_cash: float = 10_000.0,
        max_steps: int = 500,
        window_size: int = 20,
        cache_dir: str = ".data_cache",
    ):
        rubric = _make_rubric()
        super().__init__(rubric=rubric)

        self.ticker = ticker
        self.interval = interval
        self.period = period
        self.initial_cash = initial_cash
        self.max_steps = max_steps
        self.window_size = window_size

        self.market = MarketSimulator(
            ticker=ticker,
            interval=interval,
            period=period,
            cache_dir=cache_dir,
            window_size=window_size,
        )
        self.portfolio = Portfolio(initial_cash=initial_cash)

        self._state: PortfolioState = PortfolioState(
            episode_id=str(uuid4()),
            step_count=0,
            cash=initial_cash,
            shares=0.0,
            current_price=0.0,
            portfolio_value=initial_cash,
            initial_value=initial_cash,
            current_ticker=ticker,
        )
        self._episode_steps: int = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float,
        done: bool,
        trade_info: str = "",
        error: Optional[str] = None,
    ) -> MarketObservation:
        features = self.market.get_features()
        return MarketObservation(
            done=done,
            reward=reward,
            ohlcv_window=features.get("ohlcv_window", [0.0] * (self.window_size * 5)),
            rsi=features.get("rsi", 50.0),
            macd=features.get("macd", 0.0),
            macd_signal=features.get("macd_signal", 0.0),
            bb_upper=features.get("bb_upper", 0.0),
            bb_lower=features.get("bb_lower", 0.0),
            bb_mid=features.get("bb_mid", 0.0),
            portfolio_value=self.portfolio.portfolio_value,
            cash_ratio=self.portfolio.cash_ratio,
            position_ratio=self.portfolio.position_ratio,
            unrealized_pnl=self.portfolio.unrealized_pnl,
            regime=int(features.get("regime", 0)),
            step_num=self._episode_steps,
            trade_info=trade_info,
            error=error,
            metadata={
                "ticker": self.ticker,
                "price": self.market.current_price,
                "cash": self.portfolio.cash,
                "shares": self.portfolio.shares,
            },
        )

    def _sync_state(self) -> None:
        self._state.cash = self.portfolio.cash
        self._state.shares = self.portfolio.shares
        self._state.current_price = self.market.current_price
        self._state.portfolio_value = self.portfolio.portfolio_value
        self._state.trade_history = list(self.portfolio.trade_history)
        self._state.episode_returns = list(self.portfolio.returns)
        self._state.current_data_idx = self.market._cursor if hasattr(self.market, "_cursor") else 0
        self._state.total_steps = self._episode_steps
        self._state.step_count = self._episode_steps

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MarketObservation:
        self.market.reset(seed=seed)
        self.portfolio = Portfolio(initial_cash=self.initial_cash)

        # Mark-to-market at the starting price so indicators are populated
        self.portfolio.mark_to_market(self.market.current_price)

        self._state = PortfolioState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            cash=self.portfolio.cash,
            shares=self.portfolio.shares,
            current_price=self.market.current_price,
            portfolio_value=self.portfolio.portfolio_value,
            initial_value=self.initial_cash,
            current_ticker=self.ticker,
        )
        self._episode_steps = 0
        self._reset_rubric()
        return self._build_observation(reward=0.0, done=False, trade_info="reset")

    def step(self, action: TradeAction, **kwargs: Any) -> MarketObservation:
        try:
            price = self.market.current_price
            trade = self.portfolio.execute(
                action_type=int(action.action_type),
                size=float(action.size),
                price=price,
                timestamp=self.market.current_timestamp,
            )

            # Advance market one candle
            has_next = self.market.step_forward()

            # Mark to market at new price
            new_price = self.market.current_price
            self.portfolio.mark_to_market(new_price)

            self._episode_steps += 1
            done = (not has_next) or (self._episode_steps >= self.max_steps)

            obs = self._build_observation(
                reward=0.0,  # rubric fills this in
                done=done,
                trade_info=trade.get("info", ""),
            )
            obs.reward = self._apply_rubric(action, obs)

            if done:
                self._finalise_episode(obs)

            self._sync_state()
            return obs

        except Exception as e:
            logger.exception("step failed: %s", e)
            self._episode_steps += 1
            self._sync_state()
            return self._build_observation(
                reward=0.0, done=True, trade_info="", error=str(e)
            )

    def _finalise_episode(self, obs: MarketObservation) -> None:
        """Hook for end-of-episode work (shared with critic via metadata)."""
        if self.rubric is not None and hasattr(self.rubric, "episode_summary"):
            summary = self.rubric.episode_summary()
        else:
            summary = self.portfolio.episode_stats()
        obs.metadata["episode_summary"] = summary
        obs.metadata["trade_history"] = list(self.portfolio.trade_history)

    @property
    def state(self) -> PortfolioState:
        return self._state

    def close(self) -> None:
        pass
