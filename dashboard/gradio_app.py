"""Gradio dashboard for the self-evolving trading agent.

This is the HuggingFace Spaces entry point. It runs the environment and DQN
agent *in-process* (no separate FastAPI server) to fit within the free-tier
2 vCPU / 16GB Space.

Panels:
    - Live candlestick chart with trade markers
    - Portfolio value curve
    - Rolling reward / epsilon
    - Current reward weights (updated by the self-evolution critic)
    - Critic's "internal monologue" log
    - Training controls: Start / Stop / Reset
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make repo root importable when running from /dashboard
_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "envs", _REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dashboard")


# ----------------------------------------------------------------------
# Training state (shared between background thread and Gradio callbacks)
# ----------------------------------------------------------------------


class TrainingState:
    """All mutable dashboard state lives here, behind a single lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.stop_flag = False
        self.thread: Optional[threading.Thread] = None

        self.episode: int = 0
        self.total_steps: int = 0
        self.epsilon: float = 1.0

        # rolling charts (keep recent history capped to avoid memory blow-up)
        self.portfolio_curve: deque[float] = deque(maxlen=5000)
        self.episode_rewards: deque[float] = deque(maxlen=500)
        self.episode_returns: deque[float] = deque(maxlen=500)

        # per-step price stream (trailing window used by candlestick chart)
        self.price_history: deque[Dict[str, Any]] = deque(maxlen=300)
        self.trade_markers: deque[Dict[str, Any]] = deque(maxlen=300)

        # critic
        self.current_weights: Dict[str, float] = {}
        self.evolution_log: List[Dict[str, Any]] = []

        self.latest_summary: Dict[str, Any] = {}
        self.log_lines: deque[str] = deque(maxlen=200)

    def log(self, msg: str) -> None:
        logger.info(msg)
        with self.lock:
            self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


STATE = TrainingState()


# ----------------------------------------------------------------------
# Training loop (runs in background thread)
# ----------------------------------------------------------------------


def _run_training(
    ticker: str,
    interval: str,
    period: str,
    num_episodes: int,
    max_steps: int,
    critic_backend: str,
) -> None:
    """Main training loop — runs until STATE.stop_flag is set."""
    try:
        # Lazy imports so the Gradio app can still start even if one piece fails
        from trading_env.models import TradeAction
        from trading_env.server.trading_environment import TradingEnvironment
        from agents.dqn_trader import DQNTrader, obs_to_vector, ACTION_SPACE
        from agents.evolution_critic import EvolutionCritic
        from rubrics.trading_rubric import AdaptiveTradingRubric
    except Exception as e:
        STATE.log(f"[fatal] import error: {e}")
        with STATE.lock:
            STATE.running = False
        return

    STATE.log(f"booting env ticker={ticker} interval={interval} period={period}")
    try:
        env = TradingEnvironment(
            ticker=ticker, interval=interval, period=period, max_steps=max_steps
        )
    except Exception as e:
        STATE.log(f"[fatal] env init failed: {e}")
        with STATE.lock:
            STATE.running = False
        return

    agent = DQNTrader()
    critic = EvolutionCritic(backend=critic_backend)
    STATE.log(f"critic backend = {critic.backend}")

    with STATE.lock:
        if isinstance(env.rubric, AdaptiveTradingRubric):
            STATE.current_weights = dict(env.rubric.weights)

    for ep in range(num_episodes):
        if STATE.stop_flag:
            STATE.log("stop flag set — exiting")
            break

        obs = env.reset()
        with STATE.lock:
            STATE.episode = ep + 1
            STATE.portfolio_curve.clear()
            STATE.price_history.clear()
            STATE.trade_markers.clear()
            STATE.portfolio_curve.append(obs.portfolio_value)

        ep_reward = 0.0
        step_idx = 0

        while not obs.done and not STATE.stop_flag:
            state_vec = obs_to_vector(obs)
            action_idx, (at, sz) = agent.select_action(obs)
            action = TradeAction(action_type=at, size=sz)

            next_obs = env.step(action)
            reward = float(next_obs.reward or 0.0)
            ep_reward += reward

            agent.observe(obs, action_idx, reward, next_obs, next_obs.done)
            loss = agent.train_step()

            with STATE.lock:
                STATE.total_steps += 1
                STATE.portfolio_curve.append(next_obs.portfolio_value)
                price = float(next_obs.metadata.get("price", 0.0))
                STATE.price_history.append(
                    {"step": step_idx, "price": price, "value": next_obs.portfolio_value}
                )
                if at in (1, 2) and sz > 0:
                    STATE.trade_markers.append(
                        {"step": step_idx, "price": price, "side": "buy" if at == 1 else "sell"}
                    )

            obs = next_obs
            step_idx += 1

        # ---- end of episode: run the self-evolution critic ----
        summary: Dict[str, Any] = {}
        if env.rubric is not None and hasattr(env.rubric, "episode_summary"):
            summary = env.rubric.episode_summary()

        with STATE.lock:
            STATE.episode_rewards.append(ep_reward)
            STATE.episode_returns.append(float(summary.get("total_return", 0.0)))
            STATE.latest_summary = summary
            STATE.epsilon = getattr(agent, "epsilon", 0.0)

        if isinstance(env.rubric, AdaptiveTradingRubric):
            try:
                decision = critic.apply(env.rubric, summary)
                with STATE.lock:
                    STATE.current_weights = dict(env.rubric.weights)
                    STATE.evolution_log.append(
                        {
                            "episode": ep + 1,
                            "summary": summary,
                            "new_weights": dict(env.rubric.weights),
                            "reasoning": decision.reasoning,
                        }
                    )
                STATE.log(
                    f"ep {ep+1} done | ret={summary.get('total_return', 0):+.2%} "
                    f"sharpe={summary.get('sharpe', 0):.2f} trades={summary.get('num_trades', 0)} "
                    f"→ {decision.reasoning[:80]}"
                )
            except Exception as e:
                STATE.log(f"critic error: {e}")
        else:
            STATE.log(f"ep {ep+1} done (no rubric)")

    with STATE.lock:
        STATE.running = False
        STATE.stop_flag = False
    STATE.log("training loop exited")


# ----------------------------------------------------------------------
# Gradio callbacks
# ----------------------------------------------------------------------


def start_training(
    ticker: str,
    interval: str,
    period: str,
    num_episodes: int,
    max_steps: int,
    critic_backend: str,
) -> str:
    with STATE.lock:
        if STATE.running:
            return "Training already running."
        STATE.running = True
        STATE.stop_flag = False

    t = threading.Thread(
        target=_run_training,
        args=(ticker, interval, period, int(num_episodes), int(max_steps), critic_backend),
        daemon=True,
    )
    STATE.thread = t
    t.start()
    return f"Started training: {ticker} / {interval} / {period}"


def stop_training() -> str:
    with STATE.lock:
        if not STATE.running:
            return "Training is not running."
        STATE.stop_flag = True
    return "Stop signal sent — will finish current step."


def reset_all() -> str:
    with STATE.lock:
        if STATE.running:
            return "Stop training first."
        STATE.episode = 0
        STATE.total_steps = 0
        STATE.portfolio_curve.clear()
        STATE.price_history.clear()
        STATE.trade_markers.clear()
        STATE.episode_rewards.clear()
        STATE.episode_returns.clear()
        STATE.current_weights.clear()
        STATE.evolution_log.clear()
        STATE.latest_summary = {}
        STATE.log_lines.clear()
    return "State cleared."


# ----------------------------------------------------------------------
# Figure builders
# ----------------------------------------------------------------------


def _price_figure() -> go.Figure:
    with STATE.lock:
        history = list(STATE.price_history)
        trades = list(STATE.trade_markers)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price & Trades", "Portfolio Value ($)"),
    )
    if history:
        df = pd.DataFrame(history)
        fig.add_trace(
            go.Scatter(x=df["step"], y=df["price"], mode="lines", name="Price",
                       line=dict(color="#1f77b4", width=1.5)),
            row=1, col=1,
        )
        if trades:
            t_df = pd.DataFrame(trades)
            buys = t_df[t_df["side"] == "buy"]
            sells = t_df[t_df["side"] == "sell"]
            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys["step"], y=buys["price"], mode="markers", name="Buy",
                        marker=dict(symbol="triangle-up", size=10, color="#2ca02c"),
                    ),
                    row=1, col=1,
                )
            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells["step"], y=sells["price"], mode="markers", name="Sell",
                        marker=dict(symbol="triangle-down", size=10, color="#d62728"),
                    ),
                    row=1, col=1,
                )
        fig.add_trace(
            go.Scatter(x=df["step"], y=df["value"], mode="lines", name="Portfolio",
                       line=dict(color="#ff7f0e", width=1.5)),
            row=2, col=1,
        )
    fig.update_layout(
        height=520, margin=dict(l=40, r=20, t=40, b=30), showlegend=True,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
    )
    return fig


def _rewards_figure() -> go.Figure:
    with STATE.lock:
        rewards = list(STATE.episode_rewards)
        returns = list(STATE.episode_returns)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Episode Reward", "Episode Return %"),
        horizontal_spacing=0.12,
    )
    if rewards:
        fig.add_trace(
            go.Scatter(y=rewards, mode="lines+markers", name="reward",
                       line=dict(color="#9467bd")),
            row=1, col=1,
        )
    if returns:
        fig.add_trace(
            go.Scatter(y=[r * 100 for r in returns], mode="lines+markers", name="return %",
                       line=dict(color="#17becf")),
            row=1, col=2,
        )
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=40, b=30), showlegend=False,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
    )
    return fig


def _weights_figure() -> go.Figure:
    with STATE.lock:
        weights = dict(STATE.current_weights)

    if not weights:
        return go.Figure()
    fig = go.Figure(
        go.Bar(
            x=list(weights.keys()),
            y=list(weights.values()),
            marker=dict(color="#2ca02c"),
        )
    )
    fig.update_layout(
        title="Reward Function Weights (live)",
        height=280, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
    )
    return fig


def _status_text() -> str:
    with STATE.lock:
        s = STATE.latest_summary
        lines = [
            f"**Running:** {STATE.running}",
            f"**Episode:** {STATE.episode}",
            f"**Total steps:** {STATE.total_steps}",
            f"**Epsilon:** {STATE.epsilon:.3f}",
        ]
        if s:
            lines += [
                f"**Last return:** {s.get('total_return', 0):+.2%}",
                f"**Sharpe:** {s.get('sharpe', 0):.2f}",
                f"**Max drawdown:** {s.get('max_drawdown', 0):.2%}",
                f"**Trades:** {s.get('num_trades', 0)}",
            ]
    return "  \n".join(lines)


def _evolution_log_text() -> str:
    with STATE.lock:
        log = list(STATE.evolution_log[-20:])
    if not log:
        return "_No evolution events yet — the critic fires at the end of each episode._"
    chunks = []
    for e in reversed(log):
        chunks.append(
            f"**Episode {e['episode']}** — return {e['summary'].get('total_return', 0):+.2%}, "
            f"sharpe {e['summary'].get('sharpe', 0):.2f}\n"
            f"> {e['reasoning']}\n"
        )
    return "\n".join(chunks)


def _activity_log_text() -> str:
    with STATE.lock:
        return "\n".join(list(STATE.log_lines)[-40:])


def refresh_all():
    return (
        _price_figure(),
        _rewards_figure(),
        _weights_figure(),
        _status_text(),
        _evolution_log_text(),
        _activity_log_text(),
    )


# ----------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Self-Evolving Trading Agent", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# Self-Evolving Trading Agent\n"
            "DQN trader on historical OHLCV (yfinance). Between episodes, a critic "
            "rewrites the reward-function weights to adapt the agent's behaviour. "
            "All trades are paper money ($10,000 virtual)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                ticker = gr.Textbox("AAPL", label="Ticker")
                interval = gr.Dropdown(
                    ["1d", "1h", "30m", "15m"], value="1d", label="Interval"
                )
                period = gr.Dropdown(
                    ["1y", "2y", "5y", "10y", "max"], value="5y", label="Period"
                )
                num_episodes = gr.Slider(1, 500, value=50, step=1, label="Episodes")
                max_steps = gr.Slider(50, 2000, value=300, step=10, label="Max steps / episode")
                critic_backend = gr.Dropdown(
                    ["heuristic", "transformers", "openai", "auto"],
                    value="heuristic",
                    label="Evolution critic backend",
                )
                start_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop")
                reset_btn = gr.Button("Reset")
                status_msg = gr.Markdown("")
            with gr.Column(scale=3):
                status = gr.Markdown()
                price_plot = gr.Plot(label="Market & Portfolio")

        with gr.Row():
            rewards_plot = gr.Plot(label="Episode Reward / Return")
            weights_plot = gr.Plot(label="Reward Weights")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Self-Evolution Log (Critic's Internal Monologue)")
                evolution_log = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Activity Log")
                activity_log = gr.Textbox(lines=15, interactive=False, show_label=False)

        start_btn.click(
            start_training,
            inputs=[ticker, interval, period, num_episodes, max_steps, critic_backend],
            outputs=status_msg,
        )
        stop_btn.click(stop_training, outputs=status_msg)
        reset_btn.click(reset_all, outputs=status_msg)

        timer = gr.Timer(2.0)
        timer.tick(
            refresh_all,
            outputs=[price_plot, rewards_plot, weights_plot, status, evolution_log, activity_log],
        )

    return ui


def main() -> None:
    ui = build_ui()
    port = int(os.environ.get("PORT", "7860"))
    ui.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
