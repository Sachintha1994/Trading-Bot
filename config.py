"""
Binance Trading Bot — Configuration
====================================
All tunable parameters in one place.
Adjust these values to change the bot's behavior.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# API CREDENTIALS
# ──────────────────────────────────────────────
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ──────────────────────────────────────────────
# AI ANALYST (LLM) SETTINGS
# ──────────────────────────────────────────────
LLM_ANALYST_ENABLED = True
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
LLM_ANALYST_THRESHOLD = 0.50  # Only trade if AI confidence is > 50%

# ──────────────────────────────────────────────
# TRADING MODE
# ──────────────────────────────────────────────
USE_TESTNET = True  # True = Testnet, False = LIVE
DRY_RUN = True      # True = DEMO MODE (Local simulation, zero risk)
TESTNET_BASE_URL = "https://testnet.binance.vision"
LIVE_BASE_URL = "https://api.binance.com"

# ──────────────────────────────────────────────
# MARKET SETTINGS
# ──────────────────────────────────────────────
TRADING_PAIR = "BNBUSDT"
BASE_ASSET = "BNB"       # The asset you're buying/selling
QUOTE_ASSET = "USDT"     # The asset you're paying with
TIMEFRAME = "1h"         # Candle interval: 1m, 5m, 15m, 1h, 4h, 1d
KLINE_LIMIT = 100        # Number of candles to fetch for analysis

# ──────────────────────────────────────────────
# RISK MANAGEMENT (LOW-RISK PROFILE)
# ──────────────────────────────────────────────
RISK_PER_TRADE = 0.01          # 1% of portfolio per trade
MAX_DAILY_DRAWDOWN = 0.03      # 3% max daily loss — bot halts after this
STOP_LOSS_PCT = 0.02          # Initial 2% SL
TAKE_PROFIT_LADDER = [0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00] # 3% to 100%
MAX_OPEN_POSITIONS = 1         # Max 1 position at a time
COOLDOWN_CANDLES = 2           # Skip N candles after a losing trade
MIN_ORDER_VALUE_USDT = 10.0    # Binance minimum order value

# ──────────────────────────────────────────────
# CONFLUENCE STRATEGY SETTINGS
# ──────────────────────────────────────────────
CONFLUENCE_BUY_THRESHOLD = 0.65   # Score ≥ 0.65 → BUY signal
CONFLUENCE_SELL_THRESHOLD = 0.35  # Score ≤ 0.35 → SELL signal

# Indicator weights (must sum to 1.0)
INDICATOR_WEIGHTS = {
    "rsi": 0.20,
    "macd": 0.25,
    "bollinger": 0.20,
    "ema_cross": 0.15,
    "volume": 0.10,
    "vwap": 0.10,
}

# ──────────────────────────────────────────────
# TECHNICAL INDICATOR PARAMETERS
# ──────────────────────────────────────────────
# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# EMA Crossover
EMA_SHORT = 9
EMA_LONG = 21

# Volume
VOLUME_SPIKE_MULTIPLIER = 1.5  # Volume must be 1.5x average to confirm

# VWAP
VWAP_ENABLED = True

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
LOG_DIR = "logs"
TRADE_JOURNAL_FILE = os.path.join(LOG_DIR, "trade_journal.csv")
BOT_LOG_FILE = os.path.join(LOG_DIR, "bot.log")
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ──────────────────────────────────────────────
# DATA CACHE
# ──────────────────────────────────────────────
DATA_DIR = "data"

# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────
DASHBOARD_REFRESH_SECONDS = 10
