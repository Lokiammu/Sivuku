# Self-Evolving Trading Agent

An **OpenEnv** environment where a DQN trading agent paper-trades historical
OHLCV data and the **reward function itself evolves** between episodes. A
critic inspects each trajectory and rewrites the rubric weights — so the
environment doesn't just grade the agent, it *teaches* the agent what to
care about.

Built for the OpenEnv Hackathon submission.

- **Environment:** `envs/trading_env/` — single-asset paper trading
- **Rubric:** `rubrics/trading_rubric.py` — adaptive reward function, JSON-persisted weights
- **Agent baseline:** `agents/dqn_trader.py` — vanilla DQN
- **Self-evolution critic:** `agents/evolution_critic.py` — rule-based / transformers / OpenAI backends
- **Dashboard:** `dashboard/gradio_app.py` — HuggingFace Spaces UI
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

Three deterministic tasks are defined in [envs/trading_env/tasks.py](envs/trading_env/tasks.py).
Each task uses a seeded synthetic scenario so grading is reproducible across runs.

| Task | Difficulty | Scenario | Grader target |
|---|---|---|---|
| `trend_following` | easy | bullish trend | ≥ +5% return, max DD ≤ 15% |
| `volatility_control` | medium | choppy sideways | Sharpe ≥ 1.0, max DD ≤ 10%, some trading activity |
| `bear_market_survival` | hard | declining market | lose ≤ 2%, max DD ≤ 20% |

Each grader returns a score in `[0, 1]` attached to the last observation as
`metadata["task_score"]`.

## Running the baseline

The hackathon expects a single `inference.py` that produces per-task scores.

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
[START] {"env": "trading_env", "task": "...", "model": "...", "seed": N}
[STEP]  {"env": "trading_env", "task": "...", "step": i, "action": {...}, "reward": x, "done": false, "error": null}
[END]   {"env": "trading_env", "task": "...", "score": s, "success": true/false, "rewards": [...], "summary": {...}, "error": null}
```

### Baseline scores (rule-based, deterministic seeds 17/42/99)

| Task | Score | Success (≥0.5) |
|---|---|---|
| trend_following | 0.361 | no |
| volatility_control | 0.150 | no |
| bear_market_survival | 0.000 | no |

## Dev setup

```bash
# install deps (env is at repo root)
uv lock
uv sync

# run the FastAPI + WebSocket server
uv run server
# or:
python -m server.app
```

## Verifying the environment

```bash
# from repo root — no path argument needed
python -m openenv.cli validate .
```

Should print `[OK] : Ready for multi-mode deployment`.

## HuggingFace Spaces dashboard

The `dashboard/` directory is a self-contained Gradio Space that trains the
DQN agent in a background thread and shows:

- live price chart with buy/sell markers
- portfolio equity curve ($10,000 paper money)
- per-episode reward + total return
- live reward-function weights (updated by the critic each episode)
- the critic's natural-language reasoning for each weight change

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
