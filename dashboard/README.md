---
title: Self-Evolving Trading Agent
emoji: 📈
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.35.0"
app_file: app.py
pinned: false
license: bsd-3-clause
---

# Self-Evolving Trading Agent

DQN trader that paper-trades historical OHLCV (yfinance). Between episodes,
a critic rewrites the reward-function weights — this is the **self-evolution**
loop.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## What you'll see

- Live price chart with buy/sell markers
- Portfolio value curve ($10,000 paper money)
- Episode reward + return
- Live reward-function weights (updated by the critic each episode)
- Critic's internal monologue — explains *why* it changed each weight

## Critic backends

- `heuristic` — rule-based, zero dependencies, always works
- `transformers` — local SmolLM2-360M (CPU-friendly, ~400MB)
- `openai` — gpt-4o-mini (needs `OPENAI_API_KEY`)
- `auto` — picks whichever is available
