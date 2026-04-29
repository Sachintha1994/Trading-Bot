"""
Binance Trading Bot — Main Entry Point
========================================
Runs the trading loop: fetch data → analyze → decide → execute.
Press Ctrl+C to stop gracefully.
"""

import sys
import time
from datetime import datetime, timezone

import config
from core.data_engine import BinanceDataEngine
from core.indicators import calculate_all_indicators
from core.strategy import ConfluenceStrategy, SignalType
from core.risk_manager import RiskManager
from core.order_executor import OrderExecutor
from core.llm_analyst import get_analyst
from core.utils import get_logger, print_banner, format_price, format_pct

logger = get_logger("Main")

# Timeframe to seconds mapping
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
}


def run_bot():
    """Main trading bot loop."""

    print_banner()

    # ── Initialize components ────────────────
    logger.info("🔧 Initializing trading bot components...")

    data_engine = BinanceDataEngine()
    strategy = ConfluenceStrategy()
    risk_manager = RiskManager()
    executor = OrderExecutor(data_engine)

    if config.DRY_RUN:
        logger.info("🛡️ DRY RUN MODE ENABLED — No real trades will be placed.")
        data_engine.authenticated = False  # Force simulation mode

    # Check API connectivity
    if not data_engine.ping():
        logger.error("❌ Cannot reach Binance API. Check your internet connection.")
        sys.exit(1)

    logger.info("✅ All systems online")

    # Get initial balance
    try:
        if data_engine.authenticated:
            balance = data_engine.get_account_balance()
            logger.info("💰 Starting balance: %s %s", format_price(balance), config.QUOTE_ASSET)
        else:
            balance = 5.0  # Simulated balance
            logger.info("💰 Simulated balance: %s (no API key)", format_price(balance))
    except Exception as e:
        balance = 5.0
        logger.warning("💰 Could not fetch real balance. Using simulated $5.00 USDT (Note: Binance minimum is 10 USDT)")

    # Calculate sleep interval
    interval_seconds = INTERVAL_SECONDS.get(config.TIMEFRAME, 3600)
    logger.info("⏱ Analysis interval: %s (%ds)", config.TIMEFRAME, interval_seconds)

    # Track the last analyzed candle timestamp
    last_candle_time = None
    cycle_count = 0

    # ── Main loop ────────────────────────────
    logger.info("🚀 Bot is now running. Press Ctrl+C to stop.\n")

    try:
        while True:
            cycle_count += 1
            logger.info("━" * 50)
            logger.info("📊 Analysis Cycle #%d — %s", cycle_count,
                       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

            # ── Step 1: Fetch market data ────
            try:
                df = data_engine.get_klines()
                if df.empty:
                    logger.warning("⚠ No market data received. Retrying in 30s...")
                    time.sleep(30)
                    continue
            except Exception as e:
                logger.error("📡 Network Error: %s. Retrying in 10s...", e)
                time.sleep(10)
                continue

            current_price = float(df["close"].iloc[-1])
            current_candle_time = df.index[-1]

            # Skip if we already analyzed this candle
            if last_candle_time and current_candle_time == last_candle_time:
                logger.debug("⏳ Same candle, waiting for next close...")
                time.sleep(min(interval_seconds // 6, 60))
                continue

            last_candle_time = current_candle_time
            logger.info("💲 Current %s price: %s", config.TRADING_PAIR, format_price(current_price))

            # ── Step 2: Calculate indicators ─
            df = calculate_all_indicators(df)

            # ── Step 3: Check existing position SL/TP ─
            sl_tp_hit = risk_manager.check_stop_loss_take_profit(current_price)
            if sl_tp_hit == "TP_STAGE_UP":
                # Don't close, just log (SL was already moved in risk_manager)
                pass 
            elif sl_tp_hit:
                logger.info("⚡ %s triggered — closing position", sl_tp_hit)

                pos = risk_manager.state.open_position
                close_side = "SELL" if pos["side"] == "BUY" else "BUY"

                executor.place_close_order(close_side, pos["quantity"])
                result = risk_manager.close_position(current_price)

                if result:
                    logger.info(
                        "📊 Trade result: %s P&L | %s (%s)",
                        format_pct(result["pnl_pct"]),
                        format_price(result["pnl_usdt"]),
                        sl_tp_hit,
                    )

                # Update balance
                if data_engine.authenticated:
                    balance = data_engine.get_account_balance()
                else:
                    balance += result.get("pnl_usdt", 0)

            # ── Step 4: Run strategy ─────────
            signal = strategy.evaluate(df)

            # ── Step 5: Execute if signal is actionable ─
            if signal.type in (SignalType.BUY, SignalType.SELL):
                side = signal.type.value

                approved, order_params, reason = risk_manager.validate_trade(
                    side=side,
                    current_price=current_price,
                    available_balance=balance,
                    signal_score=signal.score,
                )

                if approved and order_params:
                    # ── AI Analyst Validation ──
                    analyst = get_analyst()
                    ai_verdict = analyst.analyze_trade(
                        symbol=config.TRADING_PAIR,
                        side=side,
                        current_price=current_price,
                        df=df,
                        technical_reasons=signal.reasons,
                        confluence_score=signal.score
                    )
                    
                    if ai_verdict.get("action") == "HOLD" or ai_verdict.get("confidence", 1.0) < config.LLM_ANALYST_THRESHOLD:
                        logger.warning("🚫 Trade VETOED by AI Analyst: %s (Confidence: %.2f)", 
                                       ai_verdict.get("reasoning"), ai_verdict.get("confidence", 0))
                        approved = False
                    else:
                        logger.info("🧠 AI Analyst CONFIRMED: %s", ai_verdict.get("reasoning"))

                if approved and order_params:
                    logger.info("🎯 Executing %s trade (confidence: %.2f)", side, signal.score)

                    response = executor.place_market_order(order_params)

                    if response:
                        risk_manager.register_open_position(order_params)
                        logger.info("✅ Trade executed successfully!")
                    else:
                        logger.error("❌ Order execution failed")
                else:
                    logger.info("🚫 Trade rejected by risk manager: %s", reason)

            else:
                logger.info("🟡 HOLD — No trade signal (score: %.4f)", signal.score)

            # ── Step 6: Print summary ────────
            risk_status = risk_manager.get_status()
            logger.info(
                "📈 Status: Balance=%s | Daily P&L=%s | Trades today=%d | Position=%s",
                format_price(balance),
                format_price(risk_status["daily_pnl"]),
                risk_status["daily_trades"],
                "OPEN" if risk_status["has_open_position"] else "NONE",
            )

            # ── Step 7: Daily reset check ────
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute < (interval_seconds // 60):
                risk_manager.reset_daily()

            # ── Step 8: Sleep until next candle ─
            # Sleep for a fraction of the interval to check SL/TP more frequently
            sleep_time = min(interval_seconds // 4, 300)  # Max 5 minutes
            logger.info("⏳ Sleeping %ds until next check...\n", sleep_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\n🛑 Bot stopped by user (Ctrl+C)")
        logger.info("📊 Final balance: %s", format_price(balance))
        risk_status = risk_manager.get_status()
        logger.info("📊 Today's P&L: %s over %d trades",
                    format_price(risk_status["daily_pnl"]),
                    risk_status["daily_trades"])
        logger.info("👋 Goodbye!")

    except Exception as e:
        logger.critical("💥 Unexpected error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    run_bot()
