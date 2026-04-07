"""Self-evolution critic.

At the end of each episode, this critic inspects trajectory metrics and updates
the AdaptiveTradingRubric's weights. Three backends:

1. **heuristic**  — no LLM, rule-based (always works, zero dependencies)
2. **transformers** — tiny local LLM via HuggingFace (SmolLM / Qwen2.5-0.5B)
3. **openai**       — OpenAI API (if OPENAI_API_KEY set)

The critic outputs JSON: new weights + reasoning. The reasoning is persisted
and displayed in the dashboard as the agent's "internal monologue".
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


CRITIC_SYSTEM_PROMPT = """You are a risk-management critic for a reinforcement learning trading agent.

You will receive the current reward-function weights and the episode's performance metrics.
Your job is to propose small adjustments to the weights so the agent learns safer, more profitable behaviour.

WEIGHTS (all clamped to sensible ranges):
- alpha_pnl        (0.1–10): how much to reward raw profit
- beta_downside_vol(0–10):   how much to penalise downside volatility
- gamma_drawdown   (0–10):   how much to penalise max drawdown
- delta_overtrade  (0–10):   per-trade penalty (discourages churn)
- bias             (-1–1):   constant offset

DIAGNOSTIC RULES:
- If Sharpe is very low AND drawdown is high -> increase gamma_drawdown, increase beta_downside_vol
- If num_trades is huge AND total_return is modest -> increase delta_overtrade
- If total_return is strongly positive and drawdown is small -> keep weights (small alpha bump OK)
- If the agent never trades (num_trades == 0) -> DECREASE delta_overtrade (reduce the penalty)
- Never move any weight by more than 30% of its current value in a single step.

OUTPUT FORMAT (strict JSON, no extra text):
{"weights": {"alpha_pnl": float, "beta_downside_vol": float, "gamma_drawdown": float, "delta_overtrade": float, "bias": float}, "reasoning": "one or two sentences"}
"""


@dataclass
class CriticDecision:
    new_weights: Dict[str, float]
    reasoning: str
    raw_response: str = ""


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------


def heuristic_critic(
    current_weights: Dict[str, float], summary: Dict[str, Any]
) -> CriticDecision:
    """Rule-based critic. No model. Always available."""
    w = dict(current_weights)
    total_return = float(summary.get("total_return", 0.0))
    sharpe = float(summary.get("sharpe", 0.0))
    max_dd = float(summary.get("max_drawdown", 0.0))
    num_trades = int(summary.get("num_trades", 0))
    n_steps = int(summary.get("n_steps", 1)) or 1

    notes: List[str] = []

    # 1. punished for drawdown
    if max_dd > 0.15:
        w["gamma_drawdown"] = min(10.0, w["gamma_drawdown"] * 1.2)
        notes.append(f"drawdown {max_dd:.1%} high -> bump gamma_drawdown")

    # 2. bad risk-adjusted return
    if sharpe < -0.5:
        w["beta_downside_vol"] = min(10.0, w["beta_downside_vol"] * 1.15)
        notes.append(f"sharpe {sharpe:.2f} weak -> bump beta_downside_vol")

    # 3. overtrading
    trade_rate = num_trades / n_steps
    if trade_rate > 0.4 and total_return < 0.01:
        w["delta_overtrade"] = min(10.0, w["delta_overtrade"] * 1.25 + 0.05)
        notes.append(f"overtrading ({trade_rate:.0%} steps) -> bump delta_overtrade")

    # 4. agent refuses to trade — reduce the per-trade cost
    if num_trades == 0:
        w["delta_overtrade"] = max(0.0, w["delta_overtrade"] * 0.6 - 0.02)
        notes.append("agent never traded -> cut delta_overtrade")

    # 5. reward good performance
    if total_return > 0.05 and max_dd < 0.1:
        w["alpha_pnl"] = min(10.0, w["alpha_pnl"] * 1.05)
        notes.append("strong risk-adjusted return -> small alpha_pnl bump")

    # 6. slow decay of overtrade penalty so it doesn't blow up
    if trade_rate < 0.2 and num_trades > 0:
        w["delta_overtrade"] = max(0.0, w["delta_overtrade"] * 0.95)

    reasoning = "; ".join(notes) if notes else "within normal bounds — no change"
    return CriticDecision(new_weights=w, reasoning=reasoning, raw_response=reasoning)


def openai_critic(
    current_weights: Dict[str, float], summary: Dict[str, Any], model: str = "gpt-4o-mini"
) -> CriticDecision:
    """OpenAI-backed critic. Requires OPENAI_API_KEY."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, falling back to heuristic")
        return heuristic_critic(current_weights, summary)

    client = OpenAI()
    user_msg = json.dumps({"current_weights": current_weights, "episode_summary": summary})
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return _parse_critic_json(content, current_weights)
    except Exception as e:
        logger.warning("openai critic failed: %s — falling back to heuristic", e)
        return heuristic_critic(current_weights, summary)


def transformers_critic(
    current_weights: Dict[str, float],
    summary: Dict[str, Any],
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
) -> CriticDecision:
    """Local HuggingFace model. Small (<1B) to fit on free Spaces CPU."""
    try:
        from transformers import pipeline
    except ImportError:
        logger.warning("transformers not installed, falling back to heuristic")
        return heuristic_critic(current_weights, summary)

    try:
        pipe = _get_cached_pipeline(model_name)
        user_msg = json.dumps(
            {"current_weights": current_weights, "episode_summary": summary}
        )
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        out = pipe(messages, max_new_tokens=200, do_sample=False)
        generated = out[0]["generated_text"]
        # pipeline may return a list of messages or a raw string
        if isinstance(generated, list):
            content = generated[-1].get("content", "")
        else:
            content = str(generated)
        return _parse_critic_json(content, current_weights)
    except Exception as e:
        logger.warning("transformers critic failed: %s — falling back to heuristic", e)
        return heuristic_critic(current_weights, summary)


_pipeline_cache: Dict[str, Any] = {}


def _get_cached_pipeline(model_name: str):
    if model_name not in _pipeline_cache:
        from transformers import pipeline
        logger.info("Loading transformers model: %s", model_name)
        _pipeline_cache[model_name] = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
        )
    return _pipeline_cache[model_name]


def _parse_critic_json(text: str, fallback_weights: Dict[str, float]) -> CriticDecision:
    """Extract the first JSON object from `text` and coerce into a CriticDecision."""
    # find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        logger.warning("critic response had no JSON block: %r", text[:200])
        return CriticDecision(new_weights=fallback_weights, reasoning="parse_failed", raw_response=text)
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning("critic JSON parse failed: %s — %r", e, text[:200])
        return CriticDecision(new_weights=fallback_weights, reasoning="parse_failed", raw_response=text)

    weights = data.get("weights", {}) or {}
    reasoning = str(data.get("reasoning", "")).strip()
    merged = dict(fallback_weights)
    for k, v in weights.items():
        if k in merged and isinstance(v, (int, float)):
            merged[k] = float(v)
    return CriticDecision(new_weights=merged, reasoning=reasoning or "(no reasoning)", raw_response=text)


# ----------------------------------------------------------------------
# Unified interface
# ----------------------------------------------------------------------


@dataclass
class EvolutionCritic:
    """Picks a backend and applies decisions to the rubric."""

    backend: str = "heuristic"
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.backend == "auto":
            self.backend = self._auto_select()

    def _auto_select(self) -> str:
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        try:
            import transformers  # noqa: F401
            return "transformers"
        except ImportError:
            return "heuristic"

    def decide(
        self, current_weights: Dict[str, float], summary: Dict[str, Any]
    ) -> CriticDecision:
        fn: Callable[[Dict[str, float], Dict[str, Any]], CriticDecision]
        if self.backend == "openai":
            fn = openai_critic
        elif self.backend == "transformers":
            fn = lambda w, s: transformers_critic(w, s, model_name=self.model_name)  # noqa: E731
        else:
            fn = heuristic_critic

        decision = fn(current_weights, summary)
        self.history.append(
            {
                "backend": self.backend,
                "summary": summary,
                "old_weights": dict(current_weights),
                "new_weights": dict(decision.new_weights),
                "reasoning": decision.reasoning,
            }
        )
        return decision

    def apply(self, rubric, summary: Dict[str, Any]) -> CriticDecision:
        """Query the critic and write the new weights into `rubric`."""
        decision = self.decide(dict(rubric.weights), summary)
        rubric.update_weights(decision.new_weights, reasoning=decision.reasoning)
        return decision
