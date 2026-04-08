"""Baseline inference runner for the Self-Evolving Trading Agent.

Hackathon-required entry point. Runs the three graded tasks against an
in-process TradingEnvironment and reports per-task scores.

Uses an OpenAI-compatible chat endpoint (configurable via env vars) with a
~2B-parameter LLM as the decision model. Defaults to deterministic rule-based
policy when no API key is set.

Environment variables
---------------------
API_BASE_URL   (default: https://router.huggingface.co/v1)
MODEL_NAME     (default: Qwen/Qwen2.5-1.5B-Instruct)
API_KEY / HF_TOKEN  — API key for the inference endpoint

Output format (exact hackathon spec)
-------------------------------------
[START] task=<name> env=trading_env model=<name>
[STEP] step=<n> action=<verb>(<size>) reward=<x.xx> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<x.xx> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from models import TradeAction  # noqa: E402
from server.trading_environment import TradingEnvironment  # noqa: E402
from tasks import TASKS  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("inference")

ENV_NAME = "trading_env"
DEFAULT_API_BASE = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SUCCESS_THRESHOLD = 0.5

# Optional: used by from_docker_image() when pulling a local image instead of HF Space
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

# ---------------------------------------------------------------------------
# output helpers — key=value format, NOT JSON
# ---------------------------------------------------------------------------

def _out(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

def _start(task: str, model: str) -> None:
    _out(f"[START] task={task} env={ENV_NAME} model={model}")

def _step(step: int, action: str, size: float, reward: float,
          done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    _out(
        f"[STEP] step={step} action={action}({size:.2f}) "
        f"reward={reward:.2f} done={str(done).lower()} error={err}"
    )

def _end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    _out(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}"
    )

# ---------------------------------------------------------------------------
# LLM policy
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a disciplined trading agent. Each step you receive market "
    "features and your portfolio state. Respond with ONE JSON object "
    'like {"action": "buy", "size": 0.5} and nothing else. '
    "action must be buy, sell, or hold. size must be 0.25, 0.5, 0.75, or 1.0."
)


def _obs_to_prompt(obs: Any, task: str, step: int) -> str:
    return (
        f"task={task} step={step} "
        f"rsi={float(getattr(obs,'rsi',50)):.1f} "
        f"macd={float(getattr(obs,'macd',0)):.4f} "
        f"regime={int(getattr(obs,'regime',0))} "
        f"(0=sideways 1=bull 2=bear) "
        f"cash_ratio={float(getattr(obs,'cash_ratio',1)):.2f} "
        f"position={float(getattr(obs,'position_ratio',0)):.2f} "
        f"unrealized_pnl={float(getattr(obs,'unrealized_pnl',0)):.4f}"
    )


class LLMPolicy:
    def __init__(self, model: str, base_url: str, api_key: Optional[str]) -> None:
        self.model = model
        self._client = None
        self._failures = 0
        self._disabled = False
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key or "sk-none",
                timeout=20.0,
            )
        except Exception:
            pass

    @property
    def active(self) -> bool:
        return self._client is not None and not self._disabled

    def decide(self, obs: Any, task: str, step: int) -> Dict[str, Any]:
        if not self.active:
            return _rule_based(obs, task)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _obs_to_prompt(obs, task, step)},
                ],
                temperature=0,
                max_tokens=32,
                seed=42,
            )
            text = resp.choices[0].message.content or ""
            self._failures = 0
            return _parse_llm(text)
        except Exception as e:
            self._failures += 1
            if self._failures >= 3:
                self._disabled = True
            logger.warning("LLM failed (%s); rule-based fallback", e)
            return _rule_based(obs, task)


def _parse_llm(text: str) -> Dict[str, Any]:
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            d = json.loads(text[s:e + 1])
            action = str(d.get("action", "hold")).lower()
            if action not in {"buy", "sell", "hold"}:
                action = "hold"
            size = max(0.0, min(1.0, float(d.get("size", 0.5))))
            return {"action": action, "size": size}
        except Exception:
            pass
    return {"action": "hold", "size": 0.0}


def _rule_based(obs: Any, task: str = "") -> Dict[str, Any]:
    """Deterministic task-aware policy — no randomness, fully reproducible."""
    rsi = float(getattr(obs, "rsi", 50.0))
    pos = float(getattr(obs, "position_ratio", 0.0))
    regime = int(getattr(obs, "regime", 0))
    step = int(getattr(obs, "step_num", 0))

    # ── Bear market: capital preservation — stay in cash ─────────────────
    if task == "bear_market_survival":
        if pos > 0.02:
            return {"action": "sell", "size": 1.0}   # liquidate everything
        return {"action": "hold", "size": 0.0}

    # ── Bull trend: buy early and ride ───────────────────────────────────
    if task == "trend_following":
        if pos < 0.75 and (step < 40 or rsi < 50):
            return {"action": "buy", "size": 0.5}
        if rsi > 82 and pos > 0.3:                   # severe overbought → trim
            return {"action": "sell", "size": 0.25}
        return {"action": "hold", "size": 0.0}

    # ── Volatility control: active mean-reversion ────────────────────────
    if task == "volatility_control":
        if rsi < 35 and pos < 0.75:
            return {"action": "buy", "size": 0.5}
        if rsi > 65 and pos > 0.1:
            return {"action": "sell", "size": 0.5}
        return {"action": "hold", "size": 0.0}

    # ── Default generic policy ────────────────────────────────────────────
    if rsi < 30 and pos < 0.7:
        return {"action": "buy", "size": 0.5}
    if rsi > 70 and pos > 0.1:
        return {"action": "sell", "size": 0.5}
    if regime == 2 and pos > 0.1:   # bear — exit entirely
        return {"action": "sell", "size": 1.0}
    if regime == 1 and pos < 0.5:   # bull — add exposure
        return {"action": "buy", "size": 0.5}
    return {"action": "hold", "size": 0.0}


# ---------------------------------------------------------------------------
# task runner
# ---------------------------------------------------------------------------

_ACTION_MAP = {"hold": 0, "buy": 1, "sell": 2}


def _run_task(task_name: str, policy: LLMPolicy) -> Dict[str, Any]:
    task = TASKS[task_name]
    env = TradingEnvironment(task_name=task_name)
    obs = env.reset(task_name=task_name, seed=task.seed)

    model_label = policy.model if policy.active else "rule-based"
    _start(task_name, model_label)

    rewards: List[float] = []
    step = 0
    done = False
    last_obs = obs
    task_error: Optional[str] = None

    while not done and step < env.max_steps:
        decision = policy.decide(obs, task_name, step)
        action_verb = decision["action"]
        size = float(decision["size"])
        action = TradeAction(action_type=_ACTION_MAP.get(action_verb, 0), size=size)

        try:
            obs = env.step(action)
            reward = float(getattr(obs, "reward", 0.0) or 0.0)
            done = bool(getattr(obs, "done", False))
            step_error = getattr(obs, "error", None)
        except Exception as exc:
            reward = 0.0
            done = True
            step_error = str(exc)
            task_error = step_error

        rewards.append(reward)
        _step(step, action_verb, size, reward, done, step_error)

        last_obs = obs
        step += 1

    summary = (getattr(last_obs, "metadata", None) or {}).get("episode_summary") or {}
    task_score = (getattr(last_obs, "metadata", None) or {}).get("task_score")
    if task_score is None:
        task_score = task.grade(summary)
    task_score = float(task_score)
    success = task_score >= SUCCESS_THRESHOLD

    _end(success, step, task_score, rewards)
    return {"task": task_name, "score": task_score, "success": success}


def main(task_names: Optional[List[str]] = None) -> int:
    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    base_url = os.environ.get("API_BASE_URL", DEFAULT_API_BASE)
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
    policy = LLMPolicy(model=model, base_url=base_url, api_key=api_key)

    selected = task_names or list(TASKS.keys())
    for name in selected:
        if name not in TASKS:
            logger.warning("skipping unknown task: %s", name)
            continue
        _run_task(name, policy)
    return 0


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a]
    sys.exit(main(task_names=argv or None))
