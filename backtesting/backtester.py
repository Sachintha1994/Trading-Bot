"""
Binance Trading Bot — Backtester
=================================
Simulates the trading strategy against historical data
to evaluate performance before risking real capital.

Usage:
    python -m backtesting.backtester
    python -m backtesting.backtester --symbol ETHUSDT --days 180
"""

import sys
import argparse
from datetime import datetime, timezone, timedelta
from tabulate import tabulate

import pandas as pd

import config
from core.data_engine import BinanceDataEngine
from core.indicators import calculate_all_indicators
from core.strategy import ConfluenceStrategy, SignalType
from core.utils import get_logger, format_price, format_pct

logger = get_logger("Backtester")


class Backtester:
    """
    Simulates trading strategy on historical data.

    Tracks portfolio value, trade outcomes, and performance metrics.
    """

    def __init__(
        self,
        symbol: str = None,
        initial_balance: float = 10000.0,
        risk_per_trade: float = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ):
        self.symbol = symbol or config.TRADING_PAIR
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade or config.RISK_PER_TRADE
        self.stop_loss_pct = stop_loss_pct or config.STOP_LOSS_PCT
        self.take_profit_pct = take_profit_pct or config.TAKE_PROFIT_PCT

        self.strategy = ConfluenceStrategy()
        self.data_engine = BinanceDataEngine()

        # Trade tracking
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.position = None  # None = no position

    def fetch_historical_data(self, days: int = 365, interval: str = None) -> pd.DataFrame:
        """
        Fetch historical kline data.

        Note: Binance limits kline requests to 1000 candles at a time,
        so we fetch in batches for longer periods.
        """
        interval = interval or config.TIMEFRAME
        logger.info("📥 Fetching %d days of %s data for %s...", days, interval, self.symbol)

        # Fetch max candles available (up to 1000)
        limit = min(days * 24, 1000)  # For 1h candles
        if interval == "15m":
            limit = min(days * 96, 1000)
        elif interval == "4h":
            limit = min(days * 6, 1000)
        elif interval == "1d":
            limit = min(days, 1000)

        df = self.data_engine.get_klines(
            symbol=self.symbol,
            interval=interval,
            limit=limit,
        )

        if df.empty:
            logger.error("❌ No historical data received")
            return df

        logger.info("📊 Loaded %d candles from %s to %s",
                    len(df), df.index[0], df.index[-1])
        return df

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run the backtest simulation on the provided DataFrame.

        Returns a dictionary of performance metrics.
        """
        if df.empty or len(df) < 30:
            logger.error("❌ Need at least 30 candles for backtesting")
            return {}

        logger.info("🚀 Starting backtest simulation...")
        logger.info("   Initial balance: %s", format_price(self.initial_balance))
        logger.info("   Risk per trade: %s", format_pct(self.risk_per_trade * 100))
        logger.info("   Stop loss: %s", format_pct(self.stop_loss_pct * 100))
        logger.info("   Take profit: %s", format_pct(self.take_profit_pct * 100))

        # Calculate all indicators once
        df = calculate_all_indicators(df)

        # Iterate through each candle (starting after enough data for indicators)
        start_idx = 30  # Skip initial candles where indicators aren't ready

        for i in range(start_idx, len(df)):
            # Get a slice up to the current candle (no look-ahead bias)
            window = df.iloc[:i + 1]
            current = df.iloc[i]
            current_price = float(current["close"])
            current_time = df.index[i]

            # Check existing position for SL/TP
            if self.position:
                hit = self._check_sl_tp(current)
                if hit:
                    self._close_position(hit["exit_price"], hit["reason"], current_time)

            # Get signal from strategy
            if not self.position:
                signal = self.strategy.evaluate(window)

                if signal.type == SignalType.BUY:
                    self._open_position("BUY", current_price, current_time, signal.score)
                elif signal.type == SignalType.SELL:
                    self._open_position("SELL", current_price, current_time, signal.score)

            # Record equity
            equity = self.balance
            if self.position:
                unrealized = self._calc_unrealized_pnl(current_price)
                equity += unrealized

            self.equity_curve.append({
                "time": current_time,
                "equity": equity,
                "price": current_price,
            })

        # Close any remaining open position at the last price
        if self.position:
            last_price = float(df["close"].iloc[-1])
            self._close_position(last_price, "BACKTEST_END", df.index[-1])

        # Calculate final metrics
        metrics = self._calculate_metrics()

        # Print results
        self._print_results(metrics)

        return metrics

    def _open_position(self, side: str, price: float, time: datetime, score: float):
        """Open a simulated position."""
        risk_amount = self.balance * self.risk_per_trade
        stop_distance = price * self.stop_loss_pct
        quantity = risk_amount / stop_distance

        if side == "BUY":
            sl = price * (1 - self.stop_loss_pct)
            tp = price * (1 + self.take_profit_pct)
        else:
            sl = price * (1 + self.stop_loss_pct)
            tp = price * (1 - self.take_profit_pct)

        self.position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "stop_loss": sl,
            "take_profit": tp,
            "entry_time": time,
            "signal_score": score,
        }

    def _close_position(self, exit_price: float, reason: str, time: datetime):
        """Close the current position and record the trade."""
        if not self.position:
            return

        pos = self.position
        entry = pos["entry_price"]
        qty = pos["quantity"]

        if pos["side"] == "BUY":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        pnl_usdt = pnl_pct / 100 * (entry * qty)
        self.balance += pnl_usdt

        self.trades.append({
            "side": pos["side"],
            "entry_price": entry,
            "exit_price": exit_price,
            "entry_time": pos["entry_time"],
            "exit_time": time,
            "quantity": qty,
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usdt": round(pnl_usdt, 4),
            "reason": reason,
            "signal_score": pos["signal_score"],
        })

        self.position = None

    def _check_sl_tp(self, candle: pd.Series) -> dict | None:
        """Check if the current candle hit SL or TP."""
        pos = self.position
        high = float(candle["high"])
        low = float(candle["low"])

        if pos["side"] == "BUY":
            if low <= pos["stop_loss"]:
                return {"exit_price": pos["stop_loss"], "reason": "STOP_LOSS"}
            if high >= pos["take_profit"]:
                return {"exit_price": pos["take_profit"], "reason": "TAKE_PROFIT"}
        else:
            if high >= pos["stop_loss"]:
                return {"exit_price": pos["stop_loss"], "reason": "STOP_LOSS"}
            if low <= pos["take_profit"]:
                return {"exit_price": pos["take_profit"], "reason": "TAKE_PROFIT"}

        return None

    def _calc_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for open position."""
        if not self.position:
            return 0.0
        pos = self.position
        if pos["side"] == "BUY":
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]
        return pnl_pct * (pos["entry_price"] * pos["quantity"])

    def _calculate_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "total_return_pct": 0,
                "final_balance": self.balance,
            }

        pnl_list = [t["pnl_pct"] for t in self.trades]
        usdt_list = [t["pnl_usdt"] for t in self.trades]
        winners = [p for p in pnl_list if p > 0]
        losers = [p for p in pnl_list if p <= 0]

        # Max drawdown from equity curve
        max_dd = 0
        peak = self.initial_balance
        for point in self.equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            dd = (peak - point["equity"]) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        gross_profit = sum(p for p in usdt_list if p > 0)
        gross_loss = abs(sum(p for p in usdt_list if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe ratio (simplified — annualized)
        import numpy as np
        returns = np.array(pnl_list)
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

        # Trade duration
        durations = []
        for t in self.trades:
            if t["entry_time"] and t["exit_time"]:
                dur = t["exit_time"] - t["entry_time"]
                durations.append(dur.total_seconds() / 3600)  # hours

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(self.trades) * 100,
            "total_return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100,
            "total_return_usdt": self.balance - self.initial_balance,
            "final_balance": self.balance,
            "avg_win_pct": sum(winners) / len(winners) if winners else 0,
            "avg_loss_pct": sum(losers) / len(losers) if losers else 0,
            "best_trade_pct": max(pnl_list),
            "worst_trade_pct": min(pnl_list),
            "max_drawdown_pct": max_dd,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "avg_trade_duration_hrs": sum(durations) / len(durations) if durations else 0,
            "sl_hits": sum(1 for t in self.trades if t["reason"] == "STOP_LOSS"),
            "tp_hits": sum(1 for t in self.trades if t["reason"] == "TAKE_PROFIT"),
        }

    def _print_results(self, metrics: dict):
        """Print a formatted backtest results table."""
        # Force UTF-8 output on Windows
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        if not metrics or metrics.get("total_trades", 0) == 0:
            logger.info("No trades were executed during the backtest period.")
            return

        print("\n")
        print("=" * 56)
        print("  BACKTEST RESULTS")
        print("=" * 56)

        table_data = [
            ["Initial Balance", format_price(self.initial_balance)],
            ["Final Balance", format_price(metrics["final_balance"])],
            ["Total Return", format_pct(metrics["total_return_pct"])],
            ["", ""],
            ["Total Trades", str(metrics["total_trades"])],
            ["Winning Trades", f"{metrics['winning_trades']} ({metrics['win_rate']:.1f}%)"],
            ["Losing Trades", str(metrics["losing_trades"])],
            ["", ""],
            ["Avg Win", format_pct(metrics["avg_win_pct"])],
            ["Avg Loss", format_pct(metrics["avg_loss_pct"])],
            ["Best Trade", format_pct(metrics["best_trade_pct"])],
            ["Worst Trade", format_pct(metrics["worst_trade_pct"])],
            ["", ""],
            ["Max Drawdown", format_pct(metrics["max_drawdown_pct"])],
            ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
            ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"],
            ["", ""],
            ["Stop-Loss Hits", str(metrics["sl_hits"])],
            ["Take-Profit Hits", str(metrics["tp_hits"])],
            ["Avg Duration", f"{metrics['avg_trade_duration_hrs']:.1f} hours"],
        ]

        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
        print("=" * 56)

        # Print recent trades
        if self.trades:
            print("\n  RECENT TRADES (last 10)")
            print("-" * 56)
            recent = self.trades[-10:]
            trade_table = []
            for t in recent:
                marker = "WIN" if t["pnl_pct"] > 0 else "LOSS"
                trade_table.append([
                    marker,
                    t["side"],
                    format_price(t["entry_price"]),
                    format_price(t["exit_price"]),
                    format_pct(t["pnl_pct"]),
                    t["reason"],
                ])
            print(tabulate(
                trade_table,
                headers=["", "Side", "Entry", "Exit", "P&L", "Reason"],
                tablefmt="grid",
            ))

        print("\n")

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get the equity curve as a DataFrame for plotting."""
        return pd.DataFrame(self.equity_curve)

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all trades as a DataFrame."""
        return pd.DataFrame(self.trades)


def main():
    """CLI entry point for the backtester."""
    parser = argparse.ArgumentParser(description="Backtest the trading strategy")
    parser.add_argument("--symbol", default=config.TRADING_PAIR, help="Trading pair")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--interval", default=config.TIMEFRAME, help="Candle interval")
    parser.add_argument("--balance", type=float, default=10000.0, help="Starting balance")
    args = parser.parse_args()

    bt = Backtester(symbol=args.symbol, initial_balance=args.balance)
    df = bt.fetch_historical_data(days=args.days, interval=args.interval)

    if not df.empty:
        bt.run(df)


if __name__ == "__main__":
    main()
