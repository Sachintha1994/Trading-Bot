"""
Binance Trading Bot — Risk Manager
====================================
Enforces position sizing, stop-loss/take-profit, drawdown limits,
and cooldown periods to protect capital.
"""

from datetime import datetime, timezone
from dataclasses import dataclass, field

import config
from core.utils import get_logger, load_trade_journal

logger = get_logger("RiskManager")


@dataclass
class OrderParams:
    """Validated order parameters ready for execution."""
    symbol: str
    side: str           # "BUY" or "SELL"
    quantity: float
    entry_price: float
    stop_loss: float
    tp_ladder: list[float]
    risk_usdt: float    # Moved here (non-default)
    current_stage: int = 0 # Default follows


@dataclass
class RiskState:
    """Tracks the current risk state of the bot."""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    last_trade_time: datetime = None
    is_halted: bool = False
    halt_reason: str = ""
    open_position: dict = field(default_factory=dict)
    cooldown_remaining: int = 0


class RiskManager:
    """
    Enforces all risk management rules before any trade is placed.

    Rules:
    1. Position sizing based on RISK_PER_TRADE
    2. Automatic stop-loss and take-profit calculation
    3. Max daily drawdown limit
    4. Cooldown after consecutive losses
    5. Max open positions limit
    6. Minimum order value check
    """

    def __init__(self):
        self.state = RiskState()
        self._load_daily_state()

    def _load_daily_state(self):
        """Load today's P&L from the trade journal."""
        trades = load_trade_journal()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        daily_pnl = 0.0
        daily_trades = 0
        consecutive_losses = 0

        for trade in reversed(trades):
            trade_date = trade.get("timestamp", "")[:10]
            if trade_date == today and trade.get("status") == "CLOSED":
                daily_trades += 1
                try:
                    pnl = float(trade.get("pnl_usdt", 0))
                    daily_pnl += pnl
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0  # Reset on a win
                except (ValueError, TypeError):
                    pass

        self.state.daily_pnl = daily_pnl
        self.state.daily_trades = daily_trades
        self.state.consecutive_losses = consecutive_losses

        if daily_pnl < 0:
            logger.info("📊 Today's P&L: -$%.2f (%d trades)", abs(daily_pnl), daily_trades)
        else:
            logger.info("📊 Today's P&L: +$%.2f (%d trades)", daily_pnl, daily_trades)

    def validate_trade(
        self,
        side: str,
        current_price: float,
        available_balance: float,
        signal_score: float,
    ) -> tuple[bool, OrderParams | None, str]:
        """
        Validate whether a trade should be executed.

        Args:
            side: "BUY" or "SELL"
            current_price: Current market price
            available_balance: Available USDT balance
            signal_score: Confluence score (0.0-1.0)

        Returns:
            (approved, order_params, reason)
        """
        # ── Check 1: Is the bot halted? ─────────
        if self.state.is_halted:
            return False, None, f"Bot is HALTED: {self.state.halt_reason}"

        # ── Check 2: Daily drawdown limit ───────
        max_loss = available_balance * config.MAX_DAILY_DRAWDOWN
        if self.state.daily_pnl < 0 and abs(self.state.daily_pnl) >= max_loss:
            self.state.is_halted = True
            self.state.halt_reason = (
                f"Max daily drawdown reached: -${abs(self.state.daily_pnl):.2f} "
                f"(limit: -${max_loss:.2f})"
            )
            logger.error("🛑 %s", self.state.halt_reason)
            return False, None, self.state.halt_reason

        # ── Check 3: Max open positions ─────────
        if self.state.open_position:
            return False, None, "Already have an open position — max 1 at a time"

        # ── Check 4: Cooldown period ────────────
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return False, None, (
                f"Cooldown active: {self.state.cooldown_remaining + 1} candles remaining "
                f"(after {self.state.consecutive_losses} consecutive losses)"
            )

        # ── Check 5: Calculate position size ────
        risk_amount = available_balance * config.RISK_PER_TRADE
        stop_distance = current_price * config.STOP_LOSS_PCT

        if stop_distance <= 0:
            return False, None, "Invalid stop distance"

        quantity = risk_amount / stop_distance

        # Calculate order value
        order_value = quantity * current_price

        # ── Check 6: Minimum order value ────────
        if order_value < config.MIN_ORDER_VALUE_USDT:
            return False, None, (
                f"Order value too small: ${order_value:.2f} "
                f"(minimum: ${config.MIN_ORDER_VALUE_USDT})"
            )

        # ── Check 7: Sufficient balance ─────────
        if order_value > available_balance:
            # Scale down to available balance
            quantity = (available_balance * 0.95) / current_price  # 95% to leave buffer
            order_value = quantity * current_price
            if order_value < config.MIN_ORDER_VALUE_USDT:
                return False, None, f"Insufficient balance: ${available_balance:.2f}"

        # ── Calculate SL/TP Ladder ──────────────
        tp_ladder_prices = []
        for pct in config.TAKE_PROFIT_LADDER:
            if side == "BUY":
                tp_ladder_prices.append(round(current_price * (1 + pct), 2))
            else:
                tp_ladder_prices.append(round(current_price * (1 - pct), 2))

        if side == "BUY":
            stop_loss = current_price * (1 - config.STOP_LOSS_PCT)
        else:
            stop_loss = current_price * (1 + config.STOP_LOSS_PCT)

        order_params = OrderParams(
            symbol=config.TRADING_PAIR,
            side=side,
            quantity=round(quantity, 6),
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            tp_ladder=tp_ladder_prices,
            current_stage=0,
            risk_usdt=round(risk_amount, 2)
        )

        logger.info(
            "✅ Trade validated: %s %.6f %s @ $%.2f | SL: $%.2f | TP: $%.2f | Risk: $%.2f",
            side, quantity, config.TRADING_PAIR, current_price,
            order_params.stop_loss, order_params.take_profit, risk_amount,
        )

        return True, order_params, "Trade approved"

    def register_open_position(self, order_params: OrderParams):
        """Record that a position has been opened."""
        self.state.open_position = {
            "side": order_params.side,
            "entry_price": order_params.entry_price,
            "quantity": order_params.quantity,
            "stop_loss": order_params.stop_loss,
            "tp_ladder": order_params.tp_ladder,
            "current_stage": 0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("📌 Position opened: %s", self.state.open_position)

    def close_position(self, exit_price: float) -> dict:
        """
        Close the current position and calculate P&L.

        Returns trade result dict.
        """
        pos = self.state.open_position
        if not pos:
            return {}

        entry = pos["entry_price"]
        qty = pos["quantity"]
        side = pos["side"]

        if side == "BUY":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        pnl_usdt = pnl_pct / 100 * (entry * qty)

        result = {
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "quantity": qty,
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usdt": round(pnl_usdt, 4),
        }

        # Update daily P&L
        self.state.daily_pnl += pnl_usdt
        self.state.daily_trades += 1

        # Track consecutive losses
        if pnl_usdt < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= 2:
                self.state.cooldown_remaining = config.COOLDOWN_CANDLES
                logger.warning(
                    "⏸ Cooldown activated: %d candles after %d consecutive losses",
                    config.COOLDOWN_CANDLES, self.state.consecutive_losses,
                )
        else:
            self.state.consecutive_losses = 0

        # Clear position
        self.state.open_position = {}

        emoji = "🟢" if pnl_usdt >= 0 else "🔴"
        logger.info(
            "%s Position closed: %s P&L: %+.2f%% ($%+.2f) | Daily: $%+.2f",
            emoji, side, pnl_pct, pnl_usdt, self.state.daily_pnl,
        )

        return result

    def check_stop_loss_take_profit(self, current_price: float) -> str | None:
        """
        Check if the current price has hit SL or any TP stage in the dynamic ladder.
        """
        pos = self.state.open_position
        if not pos:
            return None

        side = pos["side"]
        entry = pos["entry_price"]
        sl = pos["stop_loss"]
        ladder = pos.get("tp_ladder", [])
        stage = pos.get("current_stage", 0)

        if not ladder:
            return None

        if side == "BUY":
            # --- Check Ladder Stages ---
            for i in range(stage, len(ladder)):
                target_price = ladder[i]
                if current_price >= target_price:
                    # Milestone hit!
                    new_stage = i + 1
                    
                    if new_stage == len(ladder):
                        logger.info("🎯 FINAL LADDER TARGET hit at $%.2f. Closing trade!", current_price)
                        return "TAKE_PROFIT"
                    
                    # Move SL to previous milestone (or entry if first milestone)
                    new_sl = ladder[i-1] if i > 0 else entry
                    logger.info("🚀 Ladder Milestone %d hit! Moving Stop Loss to $%.2f.", new_stage, new_sl)
                    
                    pos["current_stage"] = new_stage
                    pos["stop_loss"] = new_sl
                    return "TP_STAGE_UP"
                else:
                    # Haven't reached this or any further milestone
                    break

            # --- Check SL (Dynamic) ---
            if current_price <= sl:
                logger.warning("🛑 STOP LOSS hit at $%.2f (Current SL: $%.2f)", current_price, sl)
                return "STOP_LOSS"

        else:  # SELL
            # --- Check Ladder Stages ---
            for i in range(stage, len(ladder)):
                target_price = ladder[i]
                if current_price <= target_price:
                    # Milestone hit!
                    new_stage = i + 1
                    
                    if new_stage == len(ladder):
                        logger.info("🎯 FINAL LADDER TARGET hit at $%.2f. Closing trade!", current_price)
                        return "TAKE_PROFIT"
                    
                    # Move SL to previous milestone (or entry if first milestone)
                    new_sl = ladder[i-1] if i > 0 else entry
                    logger.info("🚀 Ladder Milestone %d hit! Moving Stop Loss to $%.2f.", new_stage, new_sl)
                    
                    pos["current_stage"] = new_stage
                    pos["stop_loss"] = new_sl
                    return "TP_STAGE_UP"
                else:
                    break

            # --- Check SL (Dynamic) ---
            if current_price >= sl:
                logger.warning("🛑 STOP LOSS hit at $%.2f (Current SL: $%.2f)", current_price, sl)
                return "STOP_LOSS"

        return None

    def reset_daily(self):
        """Reset daily counters (call at midnight UTC)."""
        logger.info("🔄 Daily risk counters reset")
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.is_halted = False
        self.state.halt_reason = ""

    def get_status(self) -> dict:
        """Get current risk state as a dictionary."""
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "consecutive_losses": self.state.consecutive_losses,
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "has_open_position": bool(self.state.open_position),
            "open_position": self.state.open_position,
            "cooldown_remaining": self.state.cooldown_remaining,
        }
