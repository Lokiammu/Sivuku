"""Market simulation + paper-trading portfolio.

Pure quant code — no OpenEnv imports. Loads historical OHLCV via yfinance,
computes technical indicators, and runs a simple paper portfolio with fees.

Falls back to synthetic GBM data if yfinance is unavailable, so the env
always boots (important on free HuggingFace Spaces where network calls
occasionally fail).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADE_FEE = 0.001   # 0.1% per trade
DEFAULT_WINDOW = 20


# ======================================================================
# Market simulator
# ======================================================================


class MarketSimulator:
    """Replays historical OHLCV data one candle at a time.

    Data source priority:
        1. cached CSV at {cache_dir}/{ticker}_{interval}_{period}.csv
        2. yfinance download
        3. synthetic GBM fallback
    """

    def __init__(
        self,
        ticker: str = "AAPL",
        interval: str = "1d",
        period: str = "5y",
        cache_dir: str = ".data_cache",
        window_size: int = DEFAULT_WINDOW,
    ) -> None:
        self.ticker = ticker
        self.interval = interval
        self.period = period
        self.window_size = window_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._df: pd.DataFrame = self._load_data()
        self._precompute_indicators()

        self._cursor: int = self.window_size   # first valid index

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{self.ticker}_{self.interval}_{self.period}.csv"

        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if len(df) > self.window_size + 50:
                    logger.info("loaded %d candles from cache: %s", len(df), cache_path)
                    return df
            except Exception as e:
                logger.warning("cache read failed: %s", e)

        try:
            import yfinance as yf  # noqa: WPS433

            df = yf.download(
                self.ticker,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                raise RuntimeError("yfinance returned empty dataframe")

            # yfinance can return a MultiIndex on columns — flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
            df = df.dropna()

            try:
                df.to_csv(cache_path)
                logger.info("cached %d candles to %s", len(df), cache_path)
            except Exception as e:
                logger.warning("cache write failed: %s", e)

            if len(df) < self.window_size + 50:
                raise RuntimeError(f"not enough candles: {len(df)}")
            return df

        except Exception as e:
            logger.warning("yfinance failed (%s) — using synthetic GBM data", e)
            return self._synthetic_data(n=1500)

    @staticmethod
    def _synthetic_data(n: int = 1500, start_price: float = 100.0) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        drift = 0.0003
        vol = 0.015
        returns = rng.normal(drift, vol, size=n)
        close = start_price * np.exp(np.cumsum(returns))

        # build OHLC around the close
        noise = rng.normal(0, vol / 2, size=n) * close
        high = close + np.abs(noise)
        low = close - np.abs(noise)
        open_ = np.roll(close, 1)
        open_[0] = start_price
        volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)

        index = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
        return pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )

    # ------------------------------------------------------------------
    # indicators (vectorised, computed once at load)
    # ------------------------------------------------------------------

    def _precompute_indicators(self) -> None:
        close = self._df["Close"].astype(float)

        # RSI(14) — Wilder's
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, 1e-9)
        self._df["RSI"] = 100 - (100 / (1 + rs))

        # MACD(12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        self._df["MACD"] = macd
        self._df["MACD_SIG"] = macd.ewm(span=9, adjust=False).mean()

        # Bollinger Bands (20, 2σ)
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        self._df["BB_MID"] = sma
        self._df["BB_UPPER"] = sma + 2 * std
        self._df["BB_LOWER"] = sma - 2 * std

        # Market regime from rolling 20-period return + volatility
        rets = close.pct_change()
        mean20 = rets.rolling(20).mean()
        std20 = rets.rolling(20).std()
        regime = pd.Series(0, index=close.index, dtype=int)
        regime[mean20 > 0.005] = 1   # bull
        regime[mean20 < -0.005] = 2  # bear
        self._df["REGIME"] = regime
        self._df["VOL20"] = std20

        self._df = self._df.ffill().fillna(0.0)

    # ------------------------------------------------------------------
    # replay API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, start_idx: Optional[int] = None) -> int:
        """Start a new episode at a random (or given) index."""
        rng = random.Random(seed)
        min_idx = self.window_size
        max_idx = max(min_idx + 1, len(self._df) - 200)
        if start_idx is None:
            self._cursor = rng.randint(min_idx, max_idx)
        else:
            self._cursor = max(min_idx, min(start_idx, max_idx))
        return self._cursor

    def step_forward(self) -> bool:
        """Advance one candle. Returns False if we've hit the end."""
        if self._cursor + 1 >= len(self._df):
            return False
        self._cursor += 1
        return True

    @property
    def current_price(self) -> float:
        return float(self._df["Close"].iloc[self._cursor])

    @property
    def current_timestamp(self):
        return self._df.index[self._cursor]

    @property
    def done(self) -> bool:
        return self._cursor + 1 >= len(self._df)

    def get_features(self) -> Dict[str, Any]:
        """Build the feature dict consumed by the environment."""
        end = self._cursor + 1
        start = end - self.window_size
        window = self._df.iloc[start:end]

        close_last = float(window["Close"].iloc[-1])
        if close_last <= 0:
            close_last = 1.0

        # Normalise OHLC as relative change from the window's last close
        ohlcv: List[float] = []
        vol_mean = float(window["Volume"].mean()) or 1.0
        for _, row in window.iterrows():
            ohlcv.extend(
                [
                    float(row["Open"]) / close_last - 1.0,
                    float(row["High"]) / close_last - 1.0,
                    float(row["Low"]) / close_last - 1.0,
                    float(row["Close"]) / close_last - 1.0,
                    float(row["Volume"]) / vol_mean - 1.0,
                ]
            )
        # Replace any NaN/Inf with 0.0
        ohlcv = [0.0 if not np.isfinite(x) else x for x in ohlcv]

        cur = self._df.iloc[self._cursor]
        rsi = float(cur.get("RSI", 50.0))
        macd = float(cur.get("MACD", 0.0))
        macd_sig = float(cur.get("MACD_SIG", 0.0))
        bb_upper = float(cur.get("BB_UPPER", close_last))
        bb_lower = float(cur.get("BB_LOWER", close_last))
        bb_mid = float(cur.get("BB_MID", close_last))

        # Return bands as relative position vs current price
        bb_upper_rel = (bb_upper - close_last) / close_last
        bb_lower_rel = (bb_lower - close_last) / close_last
        bb_mid_rel = (bb_mid - close_last) / close_last

        regime = int(cur.get("REGIME", 0))

        features = {
            "ohlcv_window": ohlcv,
            "rsi": rsi if np.isfinite(rsi) else 50.0,
            "macd": macd if np.isfinite(macd) else 0.0,
            "macd_signal": macd_sig if np.isfinite(macd_sig) else 0.0,
            "bb_upper": bb_upper_rel if np.isfinite(bb_upper_rel) else 0.0,
            "bb_lower": bb_lower_rel if np.isfinite(bb_lower_rel) else 0.0,
            "bb_mid": bb_mid_rel if np.isfinite(bb_mid_rel) else 0.0,
            "regime": regime,
            "current_price": close_last,
        }
        return features


# ======================================================================
# Paper-trading portfolio
# ======================================================================


class Portfolio:
    """Single-asset paper portfolio. Long-only for now."""

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self.cash: float = float(initial_cash)
        self.shares: float = 0.0
        self.initial_value: float = float(initial_cash)

        self.trade_history: List[Dict[str, Any]] = []
        self.returns: List[float] = []   # per-step portfolio returns
        self._last_value: float = float(initial_cash)
        self._last_price: float = 0.0

    # ------------------------------------------------------------------
    # trading
    # ------------------------------------------------------------------

    def execute(
        self,
        action_type: int,
        size: float,
        price: float,
        timestamp: Any = None,
    ) -> Dict[str, Any]:
        """Execute a trade. Returns a dict summarising what happened."""
        size = max(0.0, min(1.0, float(size)))
        price = float(price)
        if price <= 0:
            return {"info": "invalid price", "action": "noop", "size": 0.0}

        record: Dict[str, Any] = {
            "timestamp": str(timestamp) if timestamp is not None else None,
            "price": price,
            "action_type": int(action_type),
            "size": size,
            "cash_before": self.cash,
            "shares_before": self.shares,
        }

        if action_type == 1 and size > 0 and self.cash > 1.0:
            spend = self.cash * size
            fee = spend * TRADE_FEE
            shares_bought = (spend - fee) / price
            self.cash -= spend
            self.shares += shares_bought
            record.update(
                {
                    "action": "buy",
                    "fee": fee,
                    "shares_delta": shares_bought,
                    "info": f"BUY {shares_bought:.4f} @ {price:.2f}",
                }
            )
            self.trade_history.append(record)
            return record

        if action_type == 2 and size > 0 and self.shares > 1e-9:
            shares_sold = self.shares * size
            proceeds = shares_sold * price
            fee = proceeds * TRADE_FEE
            self.cash += proceeds - fee
            self.shares -= shares_sold
            record.update(
                {
                    "action": "sell",
                    "fee": fee,
                    "shares_delta": -shares_sold,
                    "info": f"SELL {shares_sold:.4f} @ {price:.2f}",
                }
            )
            self.trade_history.append(record)
            return record

        record.update({"action": "hold", "info": "HOLD"})
        return record

    def mark_to_market(self, price: float) -> float:
        """Update portfolio value at the new price and record the return."""
        price = float(price)
        if price <= 0:
            price = self._last_price or 1.0
        self._last_price = price

        value = self.cash + self.shares * price
        if self._last_value > 0:
            step_return = (value - self._last_value) / self._last_value
        else:
            step_return = 0.0
        self.returns.append(step_return)
        self._last_value = value
        return value

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        return self.cash + self.shares * self._last_price

    @property
    def cash_ratio(self) -> float:
        pv = self.portfolio_value
        return float(self.cash / pv) if pv > 0 else 1.0

    @property
    def position_ratio(self) -> float:
        pv = self.portfolio_value
        return float((self.shares * self._last_price) / pv) if pv > 0 else 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.initial_value <= 0:
            return 0.0
        return (self.portfolio_value - self.initial_value) / self.initial_value

    # ------------------------------------------------------------------
    # episode stats (consumed by the critic)
    # ------------------------------------------------------------------

    def episode_stats(self) -> Dict[str, float]:
        returns = self.returns
        n = len(returns)
        if n == 0:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "num_trades": len(self.trade_history),
                "win_rate": 0.0,
                "volatility": 0.0,
            }

        arr = np.asarray(returns, dtype=np.float64)
        mean_r = float(arr.mean())
        std_r = float(arr.std())
        sharpe = mean_r / std_r * np.sqrt(252) if std_r > 1e-9 else 0.0

        neg = arr[arr < 0]
        if neg.size > 1:
            d_std = float(neg.std())
            sortino = mean_r / d_std * np.sqrt(252) if d_std > 1e-9 else 0.0
        else:
            sortino = float(sharpe)

        cum = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(cum)
        drawdown = (peak - cum) / peak
        max_dd = float(drawdown.max())

        total_return = float(cum[-1] - 1.0)

        # win rate over completed round-trip trades
        executed = [t for t in self.trade_history if t.get("action") in ("buy", "sell")]
        win_rate = 0.0
        if executed:
            wins = sum(1 for r in arr if r > 0)
            win_rate = float(wins / max(1, n))

        return {
            "total_return": total_return,
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": max_dd,
            "num_trades": len(executed),
            "win_rate": win_rate,
            "volatility": float(std_r),
        }
