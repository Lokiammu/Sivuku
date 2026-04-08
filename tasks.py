"""Deterministic evaluation tasks for the trading environment.

Each task defines:
- a deterministic scenario (market regime + seed) used by MarketSimulator
- a grader that consumes episode statistics and returns a score in [0.0, 1.0]

The three tasks cover different difficulty tiers so baselines can be reported
against a fixed benchmark (hackathon requirement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class TradingTask:
    name: str
    difficulty: str
    description: str
    scenario: str          # "bull", "sideways", "bear"
    seed: int              # deterministic data seed
    max_steps: int
    grader: Callable[[Dict[str, Any]], float]

    def grade(self, stats: Dict[str, Any]) -> float:
        score = float(self.grader(stats) or 0.0)
        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# graders  —  each returns a score in [0, 1]
# ---------------------------------------------------------------------------


def _grade_trend_following(stats: Dict[str, Any]) -> float:
    """Easy: beat a passive buy-and-hold benchmark in a bullish market.

    Target: total_return >= 0.05 (+5%) and max_drawdown <= 0.15.
    """
    total_return = float(stats.get("total_return", 0.0))
    max_dd = float(stats.get("max_drawdown", 1.0))

    ret_score = max(0.0, min(1.0, total_return / 0.05))        # full credit at +5%
    dd_score = max(0.0, min(1.0, (0.15 - max_dd) / 0.15))      # full credit at 0 dd
    return 0.7 * ret_score + 0.3 * dd_score


def _grade_volatility_control(stats: Dict[str, Any]) -> float:
    """Medium: in a choppy market, produce a positive risk-adjusted return.

    Target: sharpe >= 1.0 and max_drawdown <= 0.10.
    """
    sharpe = float(stats.get("sharpe", 0.0))
    max_dd = float(stats.get("max_drawdown", 1.0))
    num_trades = int(stats.get("num_trades", 0))

    sharpe_score = max(0.0, min(1.0, sharpe / 1.0))
    dd_score = max(0.0, min(1.0, (0.10 - max_dd) / 0.10))
    # penalise doing nothing (at least some trading activity is required)
    activity_bonus = 1.0 if num_trades >= 5 else num_trades / 5.0
    return 0.5 * sharpe_score + 0.35 * dd_score + 0.15 * activity_bonus


def _grade_bear_survival(stats: Dict[str, Any]) -> float:
    """Hard: survive a declining market with better-than-flat returns.

    Target: total_return > -0.02 (losing ≤2%) and max_drawdown <= 0.20.
    """
    total_return = float(stats.get("total_return", -1.0))
    max_dd = float(stats.get("max_drawdown", 1.0))

    # full credit at +0%, zero at -10%
    ret_score = max(0.0, min(1.0, (total_return + 0.10) / 0.10))
    dd_score = max(0.0, min(1.0, (0.20 - max_dd) / 0.20))
    return 0.65 * ret_score + 0.35 * dd_score


# ---------------------------------------------------------------------------
# task registry
# ---------------------------------------------------------------------------


TASKS: Dict[str, TradingTask] = {
    "trend_following": TradingTask(
        name="trend_following",
        difficulty="easy",
        description=(
            "Ride a trending bull market. Reach +5% total return while "
            "keeping drawdown under 15%."
        ),
        scenario="bull",
        seed=17,
        max_steps=250,
        grader=_grade_trend_following,
    ),
    "volatility_control": TradingTask(
        name="volatility_control",
        difficulty="medium",
        description=(
            "A sideways, volatile market. Produce a positive Sharpe ratio "
            "(≥1.0) while capping drawdown at 10%."
        ),
        scenario="sideways",
        seed=42,
        max_steps=300,
        grader=_grade_volatility_control,
    ),
    "bear_market_survival": TradingTask(
        name="bear_market_survival",
        difficulty="hard",
        description=(
            "A declining market. Preserve capital — losing ≤2% is a win, "
            "staying positive is excellent. Max drawdown ≤20%."
        ),
        scenario="bear",
        seed=99,
        max_steps=250,
        grader=_grade_bear_survival,
    ),
}


def list_tasks() -> List[Dict[str, str]]:
    return [
        {
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
            "scenario": t.scenario,
        }
        for t in TASKS.values()
    ]


def get_task(name: str) -> TradingTask:
    if name not in TASKS:
        raise KeyError(
            f"unknown task '{name}'. available: {sorted(TASKS.keys())}"
        )
    return TASKS[name]
