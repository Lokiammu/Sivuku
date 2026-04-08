---
title: Self-Evolving Trading Agent
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: bsd-3-clause
---

# Self-Evolving Trading Agent

An **OpenEnv** environment where a DQN trading agent paper-trades historical
OHLCV data and the **reward function itself evolves** between episodes. A
critic inspects each trajectory and rewrites the rubric weights — so the
environment doesn't just grade the agent, it *teaches* the agent what to
care about.

Built for the OpenEnv Hackathon submission.

- **Environment:** `server/trading_environment.py` — single-asset paper trading
- **Rubric:** `rubrics/trading_rubric.py` — adaptive reward function, JSON-persisted weights
- **Agent baseline:** `agents/dqn_trader.py` — vanilla DQN
- **Self-evolution critic:** `agents/evolution_critic.py` — rule-based / LLM backends
- **Inference entry point:** `inference.py` — runs the three graded tasks

## Action space

`TradeAction{ action_type: int, size: float }`

| action_type | meaning |
|---|---|
| 0 | hold |
| 1 | buy (spend `size` × cash) |
| 2 | sell (sell `size` × position) |

`size` is clamped to `[0, 1]`. A flat 0.1% fee is charged per trade.

## Observation space

`MarketObservation` — a fixed-length float vector:

- `ohlcv_window`: last 20 candles × `[O, H, L, C, V]` (normalised, 100 floats)
- `rsi`, `macd`, `macd_signal`, `bb_upper`, `bb_lower`, `bb_mid`
- portfolio state: `portfolio_value`, `cash_ratio`, `position_ratio`, `unrealized_pnl`
- market context: `regime` ∈ {0=sideways, 1=bull, 2=bear}, `step_num`

## Tasks (hackathon-graded)

Three deterministic tasks are defined in [tasks.py](tasks.py).
Each task uses a seeded synthetic scenario so grading is reproducible across runs.

| Task | Difficulty | Scenario | Grader target |
|---|---|---|---|
| `trend_following` | easy | bullish trend | ≥ +5% return, max DD ≤ 15% |
| `volatility_control` | medium | choppy sideways | Sharpe ≥ 1.0, max DD ≤ 10%, some trading activity |
| `bear_market_survival` | hard | declining market | lose ≤ 2%, max DD ≤ 20% |

Each grader returns a score in `[0, 1]` attached to the last observation as
`metadata["task_score"]`.

## Running the baseline

```bash
# default: Qwen/Qwen2.5-1.5B-Instruct via HuggingFace Inference Router
python inference.py

# or pick a specific subset of tasks
python inference.py trend_following volatility_control

# override the LLM backend with any OpenAI-compatible endpoint
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
export HF_TOKEN=hf_xxx   # or API_KEY
python inference.py
```

When no API key is set the script uses a deterministic rule-based policy —
output is identical across runs. Output format:

```
[START] task=<name> env=trading_env model=<name>
[STEP]  step=<n> action=<verb>(<size>) reward=<x.xx> done=<true|false> error=<null>
[END]   success=<true|false> steps=<n> score=<x.xx> rewards=<r1,r2,...>
```

## Dev setup

```bash
# install deps
uv lock
uv sync

# run the FastAPI + WebSocket server
uv run server
# or:
python -m server.app
```

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Send action, get observation |
| `/state` | GET | Current portfolio state |
| `/schema` | GET | Action / observation schemas |
| `/docs` | GET | Interactive API docs (Swagger) |

## Architecture

```
┌─────────────┐    TradeAction    ┌──────────────────────┐
│  DQN Agent  │ ────────────────▶ │  TradingEnvironment  │
│  (PyTorch)  │ ◀──────────────── │  (OpenEnv)           │
└─────────────┘  MarketObservation└──────────┬───────────┘
                                             │ episode_summary
                                             ▼
                                  ┌──────────────────────┐
                                  │  AdaptiveRubric      │◀──┐
                                  │  (weights on disk)   │   │
                                  └──────────┬───────────┘   │ new
                                             │ stats          │ weights
                                             ▼                │
                                  ┌──────────────────────┐   │
                                  │  Evolution Critic    │───┘
                                  │  (LLM or rule-based) │
                                  └──────────────────────┘
```

The self-evolution loop respects OpenEnv invariants:

- rewards are computed inside `step()` via the `Rubric.forward()` call
- the critic does **not** modify agent or env code — it only rewrites the
  rubric weights stored in `rubric_config.json`
- on `reset()` the environment reloads the weights, so the next episode
  automatically uses the critic's updates

## License

BSD-3-Clause (inherits from upstream OpenEnv).
