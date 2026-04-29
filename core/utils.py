"""
Binance Trading Bot — Utilities
================================
Logging, trade journaling, and helper functions.
"""

import os
import csv
import logging
from datetime import datetime, timezone
from colorama import Fore, Style, init as colorama_init

import config

colorama_init(autoreset=True)

# ──────────────────────────────────────────────
# LOGGER SETUP
# ──────────────────────────────────────────────

os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.DATA_DIR, exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def get_logger(name: str = "TradingBot") -> logging.Logger:
    """Create a logger with both colored console and file output."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # Console handler (colored)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_fmt = ColoredFormatter(
        "%(asctime)s │ %(levelname)-18s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (plain text)
    file_handler = logging.FileHandler(config.BOT_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()


# ──────────────────────────────────────────────
# TRADE JOURNAL
# ──────────────────────────────────────────────

JOURNAL_COLUMNS = [
    "timestamp", "side", "symbol", "entry_price", "quantity",
    "stop_loss", "take_profit", "exit_price", "pnl_pct",
    "pnl_usdt", "signal_score", "status", "notes",
]


def init_trade_journal():
    """Create the trade journal CSV file with headers if it doesn't exist."""
    if not os.path.exists(config.TRADE_JOURNAL_FILE):
        with open(config.TRADE_JOURNAL_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(JOURNAL_COLUMNS)
        logger.info("Trade journal initialized: %s", config.TRADE_JOURNAL_FILE)


def log_trade(trade_data: dict):
    """Append a trade record to the CSV journal."""
    init_trade_journal()
    row = [trade_data.get(col, "") for col in JOURNAL_COLUMNS]
    with open(config.TRADE_JOURNAL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    logger.info(
        "📝 Trade logged: %s %s @ %s | P&L: %s%%",
        trade_data.get("side", "?"),
        trade_data.get("symbol", "?"),
        trade_data.get("entry_price", "?"),
        trade_data.get("pnl_pct", "pending"),
    )


def load_trade_journal() -> list[dict]:
    """Load all trades from the journal CSV."""
    if not os.path.exists(config.TRADE_JOURNAL_FILE):
        return []
    trades = []
    with open(config.TRADE_JOURNAL_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return trades


# ──────────────────────────────────────────────
# FORMATTING HELPERS
# ──────────────────────────────────────────────

def format_price(price: float, decimals: int = 2) -> str:
    """Format a price value with commas and specified decimals."""
    return f"${price:,.{decimals}f}"


def format_pct(value: float) -> str:
    """Format a percentage value with sign and color hint."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_quantity(qty: float, decimals: int = 6) -> str:
    """Format a quantity with appropriate precision."""
    return f"{qty:.{decimals}f}"


def now_utc() -> datetime:
    """Get the current UTC datetime."""
    return datetime.now(timezone.utc)


def now_str() -> str:
    """Get the current UTC time as a formatted string."""
    return now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ──────────────────────────────────────────────
# PERFORMANCE CALCULATOR
# ──────────────────────────────────────────────

def calculate_performance(trades: list[dict]) -> dict:
    """Calculate performance metrics from a list of trade records."""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "total_pnl_usdt": 0.0,
            "avg_pnl_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
    if not closed_trades:
        return {
            "total_trades": len(trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "total_pnl_usdt": 0.0,
            "avg_pnl_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    pnl_values = []
    for t in closed_trades:
        try:
            pnl_values.append(float(t.get("pnl_pct", 0)))
        except (ValueError, TypeError):
            pnl_values.append(0.0)

    winning = [p for p in pnl_values if p > 0]
    losing = [p for p in pnl_values if p < 0]

    # Calculate max drawdown from cumulative P&L
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnl_values:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    total_pnl_usdt = 0.0
    for t in closed_trades:
        try:
            total_pnl_usdt += float(t.get("pnl_usdt", 0))
        except (ValueError, TypeError):
            pass

    return {
        "total_trades": len(closed_trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": (len(winning) / len(closed_trades)) * 100 if closed_trades else 0.0,
        "total_pnl_pct": sum(pnl_values),
        "total_pnl_usdt": total_pnl_usdt,
        "avg_pnl_pct": sum(pnl_values) / len(pnl_values) if pnl_values else 0.0,
        "best_trade_pct": max(pnl_values) if pnl_values else 0.0,
        "worst_trade_pct": min(pnl_values) if pnl_values else 0.0,
        "max_drawdown_pct": max_dd,
    }


# ──────────────────────────────────────────────
# STARTUP BANNER
# ──────────────────────────────────────────────

def print_banner():
    """Print a styled startup banner."""
    mode = f"{Fore.YELLOW}⚠ TESTNET{Style.RESET_ALL}" if config.USE_TESTNET else f"{Fore.RED}🔴 LIVE TRADING{Style.RESET_ALL}"
    banner = f"""
{Fore.CYAN}{'═' * 56}
  ╔══════════════════════════════════════════════════╗
  ║         🤖  BINANCE TRADING BOT  🤖              ║
  ║         Confluence Strategy Engine               ║
  ╚══════════════════════════════════════════════════╝
{'═' * 56}{Style.RESET_ALL}

  Mode          : {mode}
  Trading Pair  : {Fore.WHITE}{config.TRADING_PAIR}{Style.RESET_ALL}
  Timeframe     : {Fore.WHITE}{config.TIMEFRAME}{Style.RESET_ALL}
  Risk/Trade    : {Fore.GREEN}{config.RISK_PER_TRADE * 100:.1f}%{Style.RESET_ALL}
  Max Drawdown  : {Fore.YELLOW}{config.MAX_DAILY_DRAWDOWN * 100:.1f}%{Style.RESET_ALL}
  Stop Loss     : {Fore.RED}{config.STOP_LOSS_PCT * 100:.1f}%{Style.RESET_ALL}
  Take Profit   : {Fore.GREEN}{config.TAKE_PROFIT_LADDER[0]*100:.1f}% to {config.TAKE_PROFIT_LADDER[-1]*100:.1f}% (Ladder){Style.RESET_ALL}
  Buy Threshold : {Fore.CYAN}{config.CONFLUENCE_BUY_THRESHOLD}{Style.RESET_ALL}

{Fore.CYAN}{'═' * 56}{Style.RESET_ALL}
"""
    print(banner)
