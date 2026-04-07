"""Adaptive trading rubric with self-evolving weights.

Per-step reward = α * step_pnl
                - β * downside_volatility_penalty
                - γ * drawdown_penalty
                - δ * overtrading_penalty
                + bias

The critic agent rewrites rubric_config.json at episode boundaries.
The environment reloads weights on reset(), so new episodes use the new config.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

from openenv.core.rubrics.base import Rubric

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "alpha_pnl": 1.0,
    "beta_downside_vol": 0.5,
    "gamma_drawdown": 1.5,
    "delta_overtrade": 0.2,
    "bias": 0.0,
}


class AdaptiveTradingRubric(Rubric):
    """Reward function whose weights are adjusted by the self-evolution critic.

    Weights persist in a JSON file. On `reset()`, we reload them — so the critic
    can write new weights between episodes without restarting the container.
    """

    def __init__(
        self,
        config_path: str | Path = "rubric_config.json",
        initial_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.config_path = Path(config_path)
        self.weights: Dict[str, float] = dict(initial_weights or DEFAULT_WEIGHTS)
        self._last_portfolio_value: float = 0.0
        self._peak_portfolio_value: float = 0.0
        self._trade_count: int = 0
        self._step_count: int = 0
        self._return_history: list[float] = []
        self._evolution_history: list[dict] = []
        self._load_weights()

    # ------------------------------------------------------------------
    # weight persistence
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        """Load weights from disk; keep current values if file missing/bad."""
        if not self.config_path.exists():
            self._save_weights()
            return
        try:
            data = json.loads(self.config_path.read_text())
            if "weights" in data and isinstance(data["weights"], dict):
                for k, v in data["weights"].items():
                    if k in self.weights and isinstance(v, (int, float)):
                        self.weights[k] = float(v)
            if "evolution_history" in data:
                self._evolution_history = list(data["evolution_history"])[-50:]
        except Exception as e:
            logger.warning("Could not read %s: %s", self.config_path, e)

    def _save_weights(self) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                json.dumps(
                    {
                        "weights": self.weights,
                        "evolution_history": self._evolution_history[-50:],
                    },
                    indent=2,
                )
            )
        except Exception as e:
            logger.warning("Could not write %s: %s", self.config_path, e)

    def update_weights(self, new_weights: Dict[str, float], reasoning: str = "") -> None:
        """Called by the evolution critic at episode boundaries.

        Clamps weights to sane ranges to prevent runaway adaptation.
        """
        clamped: Dict[str, float] = {}
        for k, v in new_weights.items():
            if k not in self.weights or not isinstance(v, (int, float)):
                continue
            if k == "alpha_pnl":
                clamped[k] = max(0.1, min(10.0, float(v)))
            elif k == "bias":
                clamped[k] = max(-1.0, min(1.0, float(v)))
            else:
                clamped[k] = max(0.0, min(10.0, float(v)))

        old = dict(self.weights)
        self.weights.update(clamped)
        self._evolution_history.append(
            {
                "old_weights": old,
                "new_weights": dict(self.weights),
                "reasoning": reasoning,
            }
        )
        self._save_weights()
        logger.info("Rubric weights updated: %s", self.weights)

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Called at the start of each episode."""
        self._last_portfolio_value = 0.0
        self._peak_portfolio_value = 0.0
        self._trade_count = 0
        self._step_count = 0
        self._return_history.clear()
        self._load_weights()   # pick up any critic updates

    # ------------------------------------------------------------------
    # per-step reward
    # ------------------------------------------------------------------

    def forward(self, action: Any, observation: Any) -> float:
        """Compute the step reward.

        Expects observation to have: portfolio_value, unrealized_pnl.
        Expects action to have: action_type (0=hold, 1=buy, 2=sell).
        """
        portfolio_value = float(getattr(observation, "portfolio_value", 0.0) or 0.0)
        action_type = int(getattr(action, "action_type", 0) or 0)

        # First step: seed state, no reward yet
        if self._step_count == 0:
            self._last_portfolio_value = portfolio_value
            self._peak_portfolio_value = portfolio_value
            self._step_count = 1
            return 0.0

        # --- per-step PnL (in % of previous portfolio value) ---
        if self._last_portfolio_value > 0:
            step_return = (portfolio_value - self._last_portfolio_value) / self._last_portfolio_value
        else:
            step_return = 0.0
        self._return_history.append(step_return)

        # --- downside volatility (std of negative returns in rolling window) ---
        window = self._return_history[-30:]
        negatives = [r for r in window if r < 0]
        if len(negatives) >= 2:
            mean_neg = sum(negatives) / len(negatives)
            var = sum((r - mean_neg) ** 2 for r in negatives) / len(negatives)
            downside_vol = math.sqrt(var)
        else:
            downside_vol = 0.0

        # --- drawdown penalty (new low-water mark) ---
        self._peak_portfolio_value = max(self._peak_portfolio_value, portfolio_value)
        drawdown = 0.0
        if self._peak_portfolio_value > 0:
            drawdown = max(
                0.0, (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value
            )

        # --- overtrading penalty (per-trade cost) ---
        traded_this_step = action_type in (1, 2)
        if traded_this_step:
            self._trade_count += 1
        overtrade_penalty = 1.0 if traded_this_step else 0.0

        # --- compose ---
        w = self.weights
        reward = (
            w["alpha_pnl"] * step_return * 100.0            # scale % returns up
            - w["beta_downside_vol"] * downside_vol * 100.0
            - w["gamma_drawdown"] * drawdown * 100.0
            - w["delta_overtrade"] * overtrade_penalty
            + w["bias"]
        )

        self._last_portfolio_value = portfolio_value
        self._step_count += 1
        return float(reward)

    # ------------------------------------------------------------------
    # trajectory-level metrics (for the critic to consume)
    # ------------------------------------------------------------------

    def episode_summary(self) -> Dict[str, Any]:
        """Metrics the critic sees at episode end."""
        returns = self._return_history
        n = len(returns)
        if n == 0:
            return {
                "n_steps": 0,
                "total_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "num_trades": self._trade_count,
                "volatility": 0.0,
                "weights": dict(self.weights),
            }

        mean_r = sum(returns) / n
        var = sum((r - mean_r) ** 2 for r in returns) / n
        vol = math.sqrt(var)
        sharpe = (mean_r / vol * math.sqrt(252)) if vol > 1e-9 else 0.0

        neg = [r for r in returns if r < 0]
        if neg:
            d_mean = sum(neg) / len(neg)
            d_vol = math.sqrt(sum((r - d_mean) ** 2 for r in neg) / len(neg))
            sortino = (mean_r / d_vol * math.sqrt(252)) if d_vol > 1e-9 else 0.0
        else:
            sortino = sharpe

        # reconstruct drawdown from returns
        peak = 1.0
        cum = 1.0
        max_dd = 0.0
        for r in returns:
            cum *= (1.0 + r)
            peak = max(peak, cum)
            max_dd = max(max_dd, (peak - cum) / peak)

        total_return = math.prod(1.0 + r for r in returns) - 1.0

        return {
            "n_steps": n,
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_dd),
            "num_trades": self._trade_count,
            "volatility": float(vol),
            "weights": dict(self.weights),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "evolution_history": list(self._evolution_history[-50:]),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "weights" in state:
            self.weights.update(state["weights"])
        if "evolution_history" in state:
            self._evolution_history = list(state["evolution_history"])
