"""
Binance Trading Bot — Confluence Strategy
===========================================
The brain of the bot. Evaluates multiple technical indicators
and produces a weighted confluence score to determine trade signals.
"""

from enum import Enum
from dataclasses import dataclass

import pandas as pd

import config
from core.utils import get_logger

logger = get_logger("Strategy")


class SignalType(Enum):
    """Trade signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """A trade signal with confidence score and reasoning."""
    type: SignalType
    score: float                # 0.0 to 1.0 (0 = strong sell, 1 = strong buy)
    reasons: list[str]          # Human-readable explanation for each indicator
    indicator_scores: dict      # Individual indicator contributions


class ConfluenceStrategy:
    """
    Multi-indicator confluence scoring strategy.

    Each indicator produces a signal score between 0.0 (strong sell) and 1.0 (strong buy).
    These scores are combined using configurable weights to produce a composite score.

    Score ≥ buy_threshold  → BUY
    Score ≤ sell_threshold → SELL
    Otherwise              → HOLD
    """

    def __init__(self):
        self.weights = config.INDICATOR_WEIGHTS
        self.buy_threshold = config.CONFLUENCE_BUY_THRESHOLD
        self.sell_threshold = config.CONFLUENCE_SELL_THRESHOLD

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "⚠ Indicator weights sum to %.2f (should be 1.0). Normalizing.", total
            )
            for key in self.weights:
                self.weights[key] /= total

    def evaluate(self, df: pd.DataFrame) -> Signal:
        """
        Evaluate all indicators and produce a confluent trade signal.

        Args:
            df: DataFrame with all indicators already calculated.

        Returns:
            Signal with type (BUY/SELL/HOLD), confidence score, and reasoning.
        """
        if df.empty or len(df) < 2:
            return Signal(
                type=SignalType.HOLD,
                score=0.5,
                reasons=["Insufficient data"],
                indicator_scores={},
            )

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        scores = {}
        reasons = []

        # ── RSI ──────────────────────────────────
        scores["rsi"], rsi_reason = self._score_rsi(latest, prev)
        reasons.append(rsi_reason)

        # ── MACD ─────────────────────────────────
        scores["macd"], macd_reason = self._score_macd(latest, prev)
        reasons.append(macd_reason)

        # ── Bollinger Bands ──────────────────────
        scores["bollinger"], bb_reason = self._score_bollinger(latest)
        reasons.append(bb_reason)

        # ── EMA Crossover ────────────────────────
        scores["ema_cross"], ema_reason = self._score_ema_cross(latest, prev)
        reasons.append(ema_reason)

        # ── Volume ───────────────────────────────
        scores["volume"], vol_reason = self._score_volume(latest)
        reasons.append(vol_reason)

        # ── VWAP ─────────────────────────────────
        scores["vwap"], vwap_reason = self._score_vwap(latest)
        reasons.append(vwap_reason)

        # ── Calculate weighted composite score ───
        composite = 0.0
        for key, weight in self.weights.items():
            composite += scores.get(key, 0.5) * weight

        # Determine signal type
        if composite >= self.buy_threshold:
            signal_type = SignalType.BUY
        elif composite <= self.sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        signal = Signal(
            type=signal_type,
            score=round(composite, 4),
            reasons=reasons,
            indicator_scores=scores,
        )

        # Log the decision
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        logger.info(
            "%s Signal: %s (score: %.4f / threshold: %.2f)",
            emoji.get(signal_type.value, "⚪"),
            signal_type.value,
            composite,
            self.buy_threshold,
        )
        for reason in reasons:
            logger.debug("   • %s", reason)

        return signal

    # ──────────────────────────────────────────────
    # INDIVIDUAL INDICATOR SCORERS
    # ──────────────────────────────────────────────
    # Each returns (score, reason) where:
    #   score: 0.0 = strong sell, 0.5 = neutral, 1.0 = strong buy

    def _score_rsi(self, latest: pd.Series, prev: pd.Series) -> tuple[float, str]:
        """Score RSI: Oversold → bullish, Overbought → bearish."""
        rsi = latest.get("rsi")
        if rsi is None or pd.isna(rsi):
            return 0.5, "RSI: No data (neutral)"

        if rsi <= config.RSI_OVERSOLD:
            score = 0.8 + (config.RSI_OVERSOLD - rsi) / config.RSI_OVERSOLD * 0.2
            return min(score, 1.0), f"RSI: {rsi:.1f} — Oversold ↗ (bullish)"
        elif rsi >= config.RSI_OVERBOUGHT:
            score = 0.2 - (rsi - config.RSI_OVERBOUGHT) / (100 - config.RSI_OVERBOUGHT) * 0.2
            return max(score, 0.0), f"RSI: {rsi:.1f} — Overbought ↘ (bearish)"
        else:
            # Normalize between 30-70 to 0.35-0.65 (mild lean)
            normalized = 0.65 - (rsi - config.RSI_OVERSOLD) / (config.RSI_OVERBOUGHT - config.RSI_OVERSOLD) * 0.3
            return normalized, f"RSI: {rsi:.1f} — Neutral zone"

    def _score_macd(self, latest: pd.Series, prev: pd.Series) -> tuple[float, str]:
        """Score MACD: Bullish crossover → buy, Bearish crossover → sell."""
        macd_line = latest.get("MACD_line")
        macd_signal = latest.get("MACD_signal")
        prev_macd = prev.get("MACD_line")
        prev_signal = prev.get("MACD_signal")
        histogram = latest.get("MACD_histogram")

        if any(x is None or (isinstance(x, float) and pd.isna(x)) for x in [macd_line, macd_signal, prev_macd, prev_signal]):
            return 0.5, "MACD: No data (neutral)"

        # Detect crossover
        bullish_cross = prev_macd <= prev_signal and macd_line > macd_signal
        bearish_cross = prev_macd >= prev_signal and macd_line < macd_signal

        if bullish_cross:
            return 0.90, "MACD: Bullish crossover ↗ (strong buy)"
        elif bearish_cross:
            return 0.10, "MACD: Bearish crossover ↘ (strong sell)"
        elif macd_line > macd_signal:
            # Already above signal — bullish but less strong
            strength = min(abs(histogram) / 50, 0.15) if histogram else 0
            return 0.65 + strength, f"MACD: Above signal line ↗ (bullish, hist: {histogram:.2f})"
        else:
            strength = min(abs(histogram) / 50, 0.15) if histogram else 0
            return 0.35 - strength, f"MACD: Below signal line ↘ (bearish, hist: {histogram:.2f})"

    def _score_bollinger(self, latest: pd.Series) -> tuple[float, str]:
        """Score Bollinger Bands: Price near lower → buy, near upper → sell."""
        bb_pct = latest.get("BB_pct")
        close = latest.get("close")
        bb_lower = latest.get("BB_lower")
        bb_upper = latest.get("BB_upper")

        if any(x is None or (isinstance(x, float) and pd.isna(x)) for x in [bb_pct, close, bb_lower, bb_upper]):
            return 0.5, "Bollinger: No data (neutral)"

        if bb_pct <= 0:
            # Price is below lower band
            return 0.90, f"Bollinger: Price BELOW lower band ({close:.2f} < {bb_lower:.2f}) — oversold"
        elif bb_pct >= 1:
            # Price is above upper band
            return 0.10, f"Bollinger: Price ABOVE upper band ({close:.2f} > {bb_upper:.2f}) — overbought"
        elif bb_pct < 0.2:
            return 0.75, f"Bollinger: Near lower band (BB%: {bb_pct:.2f}) — bullish zone"
        elif bb_pct > 0.8:
            return 0.25, f"Bollinger: Near upper band (BB%: {bb_pct:.2f}) — bearish zone"
        else:
            return 0.5, f"Bollinger: Mid-range (BB%: {bb_pct:.2f}) — neutral"

    def _score_ema_cross(self, latest: pd.Series, prev: pd.Series) -> tuple[float, str]:
        """Score EMA crossover: Short EMA above long → bullish trend."""
        ema_short = latest.get("EMA_short")
        ema_long = latest.get("EMA_long")
        prev_short = prev.get("EMA_short")
        prev_long = prev.get("EMA_long")

        if any(x is None or (isinstance(x, float) and pd.isna(x)) for x in [ema_short, ema_long, prev_short, prev_long]):
            return 0.5, "EMA: No data (neutral)"

        bullish_cross = prev_short <= prev_long and ema_short > ema_long
        bearish_cross = prev_short >= prev_long and ema_short < ema_long

        if bullish_cross:
            return 0.85, "EMA: Bullish crossover (EMA9 crossed above EMA21) ↗"
        elif bearish_cross:
            return 0.15, "EMA: Bearish crossover (EMA9 crossed below EMA21) ↘"
        elif ema_short > ema_long:
            spread = (ema_short - ema_long) / ema_long * 100
            return 0.65, f"EMA: Uptrend (spread: {spread:.3f}%)"
        else:
            spread = (ema_long - ema_short) / ema_long * 100
            return 0.35, f"EMA: Downtrend (spread: {spread:.3f}%)"

    def _score_volume(self, latest: pd.Series) -> tuple[float, str]:
        """Score volume: High volume confirms moves, low volume suggests caution."""
        vol_ratio = latest.get("vol_ratio")
        vol_spike = latest.get("vol_spike")
        close = latest.get("close")
        open_price = latest.get("open")

        if vol_ratio is None or (isinstance(vol_ratio, float) and pd.isna(vol_ratio)):
            return 0.5, "Volume: No data (neutral)"

        is_bullish_candle = close > open_price if (close and open_price) else False

        if vol_spike:
            if is_bullish_candle:
                return 0.80, f"Volume: Spike ({vol_ratio:.1f}x avg) on bullish candle ↗"
            else:
                return 0.20, f"Volume: Spike ({vol_ratio:.1f}x avg) on bearish candle ↘"
        elif vol_ratio > 1.0:
            return 0.55 if is_bullish_candle else 0.45, f"Volume: Above average ({vol_ratio:.1f}x)"
        else:
            return 0.50, f"Volume: Below average ({vol_ratio:.1f}x) — low conviction"

    def _score_vwap(self, latest: pd.Series) -> tuple[float, str]:
        """Score VWAP: Price below VWAP → undervalued, above → overvalued."""
        if not config.VWAP_ENABLED:
            return 0.5, "VWAP: Disabled"

        vwap = latest.get("VWAP")
        close = latest.get("close")

        if any(x is None or (isinstance(x, float) and pd.isna(x)) for x in [vwap, close]):
            return 0.5, "VWAP: No data (neutral)"

        deviation = (close - vwap) / vwap * 100

        if deviation < -1.5:
            return 0.80, f"VWAP: Price {deviation:.2f}% below VWAP — undervalued ↗"
        elif deviation < -0.5:
            return 0.65, f"VWAP: Price {deviation:.2f}% below VWAP — slight discount"
        elif deviation > 1.5:
            return 0.20, f"VWAP: Price {deviation:.2f}% above VWAP — overvalued ↘"
        elif deviation > 0.5:
            return 0.35, f"VWAP: Price {deviation:.2f}% above VWAP — slight premium"
        else:
            return 0.50, f"VWAP: Price at fair value (dev: {deviation:.2f}%)"
