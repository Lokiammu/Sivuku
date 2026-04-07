"""DQN trading agent.

Works against any env that exposes ``reset()`` and ``step(action)`` and returns
a MarketObservation-shaped object. The agent is device-aware (cuda or cpu) and
safe to run on the free HuggingFace Spaces CPU tier.
"""

from __future__ import annotations

import logging
import os
import random
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action space: multi-discrete (3 action types × 4 sizes), with hold as a single
# entry. Total = 9 discrete actions.
# ---------------------------------------------------------------------------

ACTION_SPACE: List[Tuple[int, float]] = [
    (0, 0.0),                                             # hold
    (1, 0.25), (1, 0.5), (1, 0.75), (1, 1.0),             # buy variants
    (2, 0.25), (2, 0.5), (2, 0.75), (2, 1.0),             # sell variants
]
N_ACTIONS: int = len(ACTION_SPACE)


# ---------------------------------------------------------------------------
# Observation → flat vector
# ---------------------------------------------------------------------------


def _pad_or_truncate(values: Any, target_len: int) -> List[float]:
    if values is None:
        return [0.0] * target_len
    try:
        seq = list(values)
    except TypeError:
        return [0.0] * target_len
    if len(seq) >= target_len:
        return [float(x) for x in seq[:target_len]]
    return [float(x) for x in seq] + [0.0] * (target_len - len(seq))


def obs_to_vector(obs: Any) -> np.ndarray:
    """Flatten a MarketObservation into a fixed-length float32 vector.

    Layout (total = 114):
        - 100 floats  ohlcv_window (padded/truncated)
        - 1   float   rsi / 100
        - 5   floats  macd, macd_signal, bb_upper, bb_lower, bb_mid
        - 4   floats  portfolio_value, cash_ratio, position_ratio, unrealized_pnl
        - 3   floats  regime one-hot
        - 1   float   step_num
    """
    ohlcv = _pad_or_truncate(getattr(obs, "ohlcv_window", None), 100)

    rsi = float(getattr(obs, "rsi", 50.0) or 0.0) / 100.0

    indicators = [
        float(getattr(obs, "macd", 0.0) or 0.0),
        float(getattr(obs, "macd_signal", 0.0) or 0.0),
        float(getattr(obs, "bb_upper", 0.0) or 0.0),
        float(getattr(obs, "bb_lower", 0.0) or 0.0),
        float(getattr(obs, "bb_mid", 0.0) or 0.0),
    ]

    portfolio = [
        float(getattr(obs, "portfolio_value", 0.0) or 0.0),
        float(getattr(obs, "cash_ratio", 1.0) or 0.0),
        float(getattr(obs, "position_ratio", 0.0) or 0.0),
        float(getattr(obs, "unrealized_pnl", 0.0) or 0.0),
    ]

    regime = int(getattr(obs, "regime", 0) or 0)
    regime_onehot = [0.0, 0.0, 0.0]
    if 0 <= regime < 3:
        regime_onehot[regime] = 1.0

    step_num = float(getattr(obs, "step_num", 0) or 0)

    vec = np.array(
        ohlcv + [rsi] + indicators + portfolio + regime_onehot + [step_num],
        dtype=np.float32,
    )
    # Final sanity: replace any non-finite values
    vec = np.where(np.isfinite(vec), vec, 0.0).astype(np.float32)
    return vec


def _infer_state_dim() -> int:
    dummy = SimpleNamespace(
        ohlcv_window=[0.0] * 100,
        rsi=50.0,
        macd=0.0,
        macd_signal=0.0,
        bb_upper=0.0,
        bb_lower=0.0,
        bb_mid=0.0,
        portfolio_value=0.0,
        cash_ratio=1.0,
        position_ratio=0.0,
        unrealized_pnl=0.0,
        regime=0,
        step_num=0,
    )
    return int(obs_to_vector(dummy).shape[0])


STATE_DIM: int = _infer_state_dim()


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, n_actions)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Trader
# ---------------------------------------------------------------------------


class DQNTrader:
    def __init__(
        self,
        state_dim: Optional[int] = None,
        n_actions: int = N_ACTIONS,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 500,
        buffer_capacity: int = 50_000,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = int(state_dim if state_dim is not None else STATE_DIM)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.policy_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self.steps_done: int = 0

    # ------------------------------------------------------------------

    def _action_idx(self, obs_vec: np.ndarray, greedy: bool) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.from_numpy(obs_vec).float().unsqueeze(0).to(self.device)
            q = self.policy_net(x)
            return int(q.argmax(dim=1).item())

    def select_action(self, obs: Any, greedy: bool = False) -> Tuple[int, Tuple[int, float]]:
        """Return ``(action_index, (action_type, size))``."""
        vec = obs_to_vector(obs)
        idx = self._action_idx(vec, greedy=greedy)
        return idx, ACTION_SPACE[idx]

    def observe(
        self,
        obs: Any,
        action_idx: int,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        self.buffer.push(
            obs_to_vector(obs),
            int(action_idx),
            float(reward),
            obs_to_vector(next_obs),
            bool(done),
        )

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_all = self.policy_net(states_t)
        q_taken = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1

        # target network hard copy
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
            },
            str(p),
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(str(path), map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = int(ckpt.get("steps_done", 0))
        self.epsilon = float(ckpt.get("epsilon", self.epsilon_end))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _resolve_trade_action() -> Callable[..., Any]:
    """Lazy import so the file works standalone without envs on PYTHONPATH."""
    try:
        from trading_env.models import TradeAction  # type: ignore
        return lambda action_type, size: TradeAction(action_type=int(action_type), size=float(size))
    except Exception:
        try:
            from envs.trading_env.models import TradeAction  # type: ignore
            return lambda action_type, size: TradeAction(action_type=int(action_type), size=float(size))
        except Exception:
            return lambda action_type, size: SimpleNamespace(
                action_type=int(action_type), size=float(size)
            )


def train_loop(
    env: Any,
    agent: DQNTrader,
    num_episodes: int = 100,
    max_steps: int = 500,
    on_episode_end: Optional[Callable[[int, dict], None]] = None,
    on_step: Optional[Callable[[int, Any, Any, float], None]] = None,
) -> List[dict]:
    """Run a vanilla DQN training loop against any env with ``reset/step``."""
    make_action = _resolve_trade_action()
    results: List[dict] = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        steps = 0

        for step_idx in range(max_steps):
            action_idx, (at, sz) = agent.select_action(obs)
            action = make_action(at, sz)
            next_obs = env.step(action)

            reward = float(getattr(next_obs, "reward", 0.0) or 0.0)
            done = bool(getattr(next_obs, "done", False))

            agent.observe(obs, action_idx, reward, next_obs, done)
            agent.train_step()

            total_reward += reward
            steps += 1

            if on_step is not None:
                try:
                    on_step(step_idx, obs, (at, sz), reward)
                except Exception as e:  # noqa: BLE001
                    logger.warning("on_step callback failed: %s", e)

            obs = next_obs
            if done:
                break

        stats = {
            "episode": episode + 1,
            "total_reward": float(total_reward),
            "epsilon": float(agent.epsilon),
            "steps": int(steps),
            "final_portfolio": float(getattr(obs, "portfolio_value", 0.0) or 0.0),
        }
        results.append(stats)

        if on_episode_end is not None:
            try:
                on_episode_end(episode, stats)
            except Exception as e:  # noqa: BLE001
                logger.warning("on_episode_end callback failed: %s", e)

    return results
