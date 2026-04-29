"""
Binance Trading Bot — Technical Indicators
============================================
Calculates RSI, MACD, Bollinger Bands, EMA, VWAP, and Volume Analysis
using pandas-ta for reliable, vectorized computations.
"""

import pandas as pd
import pandas_ta as ta

import config
from core.utils import get_logger

logger = get_logger("Indicators")


def calculate_rsi(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures momentum — values below 30 indicate oversold (buy zone),
    values above 70 indicate overbought (sell zone).
    """
    period = period or config.RSI_PERIOD
    rsi = ta.rsi(df["close"], length=period)
    if rsi is not None:
        rsi.name = "rsi"
    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast: int = None,
    slow: int = None,
    signal: int = None,
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Returns DataFrame with columns: MACD_line, MACD_signal, MACD_histogram.
    - Bullish: MACD line crosses above signal line
    - Bearish: MACD line crosses below signal line
    """
    fast = fast or config.MACD_FAST
    slow = slow or config.MACD_SLOW
    signal = signal or config.MACD_SIGNAL

    macd_result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if macd_result is not None:
        macd_result.columns = ["MACD_line", "MACD_histogram", "MACD_signal"]
    return macd_result


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = None,
    std: float = None,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Returns DataFrame with columns: BB_lower, BB_mid, BB_upper, BB_bandwidth, BB_pct.
    - Price below lower band → potential buy
    - Price above upper band → potential sell
    """
    period = period or config.BB_PERIOD
    std = std or config.BB_STD

    bbands = ta.bbands(df["close"], length=period, std=std)
    if bbands is not None:
        cols = bbands.columns.tolist()
        # pandas-ta returns: BBL, BBM, BBU, BBB, BBP
        rename_map = {}
        for c in cols:
            if "BBL" in c:
                rename_map[c] = "BB_lower"
            elif "BBM" in c:
                rename_map[c] = "BB_mid"
            elif "BBU" in c:
                rename_map[c] = "BB_upper"
            elif "BBB" in c:
                rename_map[c] = "BB_bandwidth"
            elif "BBP" in c:
                rename_map[c] = "BB_pct"
        bbands = bbands.rename(columns=rename_map)
    return bbands


def calculate_ema(df: pd.DataFrame, short: int = None, long: int = None) -> pd.DataFrame:
    """
    Calculate EMA crossover signals.

    Returns DataFrame with columns: EMA_short, EMA_long.
    - Bullish: EMA_short > EMA_long (short-term trend is up)
    - Bearish: EMA_short < EMA_long (short-term trend is down)
    """
    short = short or config.EMA_SHORT
    long = long or config.EMA_LONG

    ema_short = ta.ema(df["close"], length=short)
    ema_long = ta.ema(df["close"], length=long)

    result = pd.DataFrame({"EMA_short": ema_short, "EMA_long": ema_long})
    return result


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP represents the average price weighted by volume — used by institutions.
    - Price below VWAP → potentially undervalued (buy zone)
    - Price above VWAP → potentially overvalued (sell zone)
    """
    # VWAP formula: cumulative(typical_price * volume) / cumulative(volume)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    vwap.name = "VWAP"
    return vwap


def calculate_volume_analysis(df: pd.DataFrame, multiplier: float = None) -> pd.DataFrame:
    """
    Analyze volume patterns.

    Returns DataFrame with columns: vol_sma, vol_ratio, vol_spike.
    - vol_spike = True when current volume > multiplier × average volume
    """
    multiplier = multiplier or config.VOLUME_SPIKE_MULTIPLIER
    period = 20

    vol_sma = df["volume"].rolling(window=period).mean()
    vol_ratio = df["volume"] / vol_sma
    vol_spike = vol_ratio > multiplier

    result = pd.DataFrame({
        "vol_sma": vol_sma,
        "vol_ratio": vol_ratio,
        "vol_spike": vol_spike,
    })
    return result


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ALL technical indicators and append them to the DataFrame.

    This is the main function called by the strategy engine.
    Returns the original DataFrame with all indicator columns added.
    """
    if df.empty:
        logger.warning("⚠ Empty DataFrame — cannot calculate indicators")
        return df

    result = df.copy()

    # RSI
    rsi = calculate_rsi(result)
    if rsi is not None:
        result["rsi"] = rsi

    # MACD
    macd = calculate_macd(result)
    if macd is not None:
        result = pd.concat([result, macd], axis=1)

    # Bollinger Bands
    bb = calculate_bollinger_bands(result)
    if bb is not None:
        result = pd.concat([result, bb], axis=1)

    # EMA Crossover
    ema = calculate_ema(result)
    if ema is not None:
        result = pd.concat([result, ema], axis=1)

    # VWAP
    if config.VWAP_ENABLED:
        vwap = calculate_vwap(result)
        result["VWAP"] = vwap

    # Volume Analysis
    vol = calculate_volume_analysis(result)
    result = pd.concat([result, vol], axis=1)

    logger.debug(
        "📐 Indicators calculated — RSI: %.1f | MACD: %.2f | BB_pct: %.2f",
        result["rsi"].iloc[-1] if "rsi" in result.columns else 0,
        result["MACD_line"].iloc[-1] if "MACD_line" in result.columns else 0,
        result["BB_pct"].iloc[-1] if "BB_pct" in result.columns else 0,
    )

    return result
