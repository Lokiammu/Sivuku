"""Baseline inference runner for the Self-Evolving Trading Agent.

Hackathon-required entry point. Executes the three graded trading tasks
against an in-process TradingEnvironment and reports the per-task score.

It uses an OpenAI-compatible chat completion endpoint (configurable via
environment variables) to ask a ~2B-parameter language model to make
discrete trading decisions each step. The default model is
``Qwen/Qwen2.5-1.5B-Instruct`` which is free via the HuggingFace Inference
Router and reliably returns small JSON objects on CPU-only Spaces.

Environment variables
---------------------
API_BASE_URL   (default: https://router.huggingface.co/v1)
MODEL_NAME     (default: Qwen/Qwen2.5-1.5B-Instruct)
API_KEY        — forwarded to the OpenAI client; leave unset for no auth
HF_TOKEN       — used as API_KEY fallback (HuggingFace convention)

Output format
-------------
Emits the exact lines required by the hackathon grader:

    [START] {"task": "...", "model": "..."}
    [STEP] {"task": "...", "step": i, "action": "...", "reward": x, "pv": y}
    [END] {"task": "...", "score": s, "summary": {...}}

If the LLM is unreachable the script falls back to a rule-based policy so
baseline scores are always produced.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ``envs/trading_env`` importable when running from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from trading_env.models import TradeAction  # noqa: E402
from trading_env.server.trading_environment import TradingEnvironment  # noqa: E402
from trading_env.tasks import TASKS  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("inference")


DEFAULT_API_BASE = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ---------------------------------------------------------------------------
# LLM policy
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a disciplined trading agent. Each step you receive market "
    "features and your portfolio state. Respond with ONE JSON object of the "
    'form {"action": "buy"|"sell"|"hold", "size": 0.25|0.5|0.75|1.0} and '
    "nothing else. Be concise and decisive."
)


def _obs_to_prompt(obs: Any, task_name: str, step: int) -> str:
    return (
        f"Task: {task_name}\n"
        f"Step: {step}\n"
        f"RSI: {float(getattr(obs, 'rsi', 50.0)):.2f}\n"
        f"MACD: {float(getattr(obs, 'macd', 0.0)):.4f}  "
        f"Signal: {float(getattr(obs, 'macd_signal', 0.0)):.4f}\n"
        f"BB upper: {float(getattr(obs, 'bb_upper', 0.0)):.4f}  "
        f"lower: {float(getattr(obs, 'bb_lower', 0.0)):.4f}\n"
        f"Cash ratio: {float(getattr(obs, 'cash_ratio', 1.0)):.3f}  "
        f"Position: {float(getattr(obs, 'position_ratio', 0.0)):.3f}  "
        f"Unrealised PnL: {float(getattr(obs, 'unrealized_pnl', 0.0)):.4f}\n"
        f"Regime: {int(getattr(obs, 'regime', 0))}  "
        f"(0=sideways, 1=bull, 2=bear)\n"
        'Respond with JSON only: {"action": "...", "size": ...}'
    )


class LLMPolicy:
    """OpenAI-compatible chat completion policy."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_API_BASE,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_failures: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
        self._client = None
        self._failures = 0
        self._disabled = False
        self._max_failures = max_failures
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "sk-none",
                timeout=self.timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("openai client init failed: %s", e)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None and not self._disabled

    def decide(self, obs: Any, task_name: str, step: int) -> Dict[str, Any]:
        if self._client is None or self._disabled:
            return _rule_based_decide(obs)

        prompt = _obs_to_prompt(obs, task_name, step)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=48,
            )
            text = resp.choices[0].message.content or ""
            self._failures = 0
            return _parse_llm_output(text)
        except Exception as e:  # noqa: BLE001
            self._failures += 1
            if self._failures <= self._max_failures:
                logger.warning("llm call failed (%s) — using rule-based fallback", e)
            if self._failures >= self._max_failures:
                if not self._disabled:
                    logger.warning(
                        "disabling LLM after %d failures — continuing with rule-based policy",
                        self._failures,
                    )
                self._disabled = True
            return _rule_based_decide(obs)


def _parse_llm_output(text: str) -> Dict[str, Any]:
    """Extract {"action", "size"} from a (possibly noisy) LLM response."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            action = str(data.get("action", "hold")).lower()
            size = float(data.get("size", 0.5))
            if action not in {"buy", "sell", "hold"}:
                action = "hold"
            size = max(0.0, min(1.0, size))
            return {"action": action, "size": size}
        except Exception:
            pass
    return {"action": "hold", "size": 0.0}


def _rule_based_decide(obs: Any) -> Dict[str, Any]:
    """Fallback policy: mean-reversion with RSI + trend filter."""
    rsi = float(getattr(obs, "rsi", 50.0))
    position = float(getattr(obs, "position_ratio", 0.0))
    regime = int(getattr(obs, "regime", 0))

    if rsi < 30 and position < 0.7:
        return {"action": "buy", "size": 0.5}
    if rsi > 70 and position > 0.1:
        return {"action": "sell", "size": 0.5}
    if regime == 2 and position > 0.3:
        return {"action": "sell", "size": 0.5}
    if regime == 1 and position < 0.3:
        return {"action": "buy", "size": 0.5}
    return {"action": "hold", "size": 0.0}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


ACTION_MAP = {"hold": 0, "buy": 1, "sell": 2}


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    """Print a structured line for the grader to parse."""
    sys.stdout.write(f"[{tag}] {json.dumps(payload, default=float)}\n")
    sys.stdout.flush()


def _run_task(task_name: str, policy: LLMPolicy) -> Dict[str, Any]:
    env = TradingEnvironment(task_name=task_name)
    obs = env.reset(task_name=task_name)

    _emit(
        "START",
        {
            "task": task_name,
            "difficulty": TASKS[task_name].difficulty,
            "model": policy.model if policy.available else "rule-based",
            "max_steps": env.max_steps,
        },
    )

    total_reward = 0.0
    step = 0
    done = False
    last_obs = obs
    while not done and step < env.max_steps:
        decision = policy.decide(obs, task_name, step)
        action_type = ACTION_MAP.get(decision["action"], 0)
        size = float(decision["size"])
        action = TradeAction(action_type=action_type, size=size)
        obs = env.step(action)
        reward = float(getattr(obs, "reward", 0.0) or 0.0)
        total_reward += reward

        _emit(
            "STEP",
            {
                "task": task_name,
                "step": step,
                "action": decision["action"],
                "size": size,
                "reward": reward,
                "pv": float(getattr(obs, "portfolio_value", 0.0) or 0.0),
            },
        )

        done = bool(getattr(obs, "done", False))
        last_obs = obs
        step += 1

    summary = (last_obs.metadata or {}).get("episode_summary") or {}
    task_score = (last_obs.metadata or {}).get("task_score")
    if task_score is None:
        task_score = TASKS[task_name].grade(summary)
    task_score = float(task_score)

    _emit(
        "END",
        {
            "task": task_name,
            "score": task_score,
            "total_reward": total_reward,
            "summary": {
                k: float(summary.get(k, 0.0))
                for k in (
                    "total_return",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "volatility",
                )
            },
            "num_trades": int(summary.get("num_trades", 0)),
            "steps": step,
        },
    )
    return {
        "task": task_name,
        "score": task_score,
        "summary": summary,
        "steps": step,
    }


def main(task_names: Optional[List[str]] = None) -> int:
    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    base_url = os.environ.get("API_BASE_URL", DEFAULT_API_BASE)
    policy = LLMPolicy(model=model, base_url=base_url)

    if not policy.available:
        logger.warning("LLM client unavailable — using rule-based fallback policy")

    selected = task_names or list(TASKS.keys())
    results: List[Dict[str, Any]] = []
    for name in selected:
        if name not in TASKS:
            logger.warning("skipping unknown task: %s", name)
            continue
        results.append(_run_task(name, policy))
        time.sleep(0.1)

    if results:
        mean_score = sum(r["score"] for r in results) / len(results)
        _emit(
            "FINAL",
            {
                "mean_score": mean_score,
                "per_task": {r["task"]: r["score"] for r in results},
                "model": policy.model if policy.available else "rule-based",
            },
        )
    return 0


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a]
    sys.exit(main(task_names=argv or None))
