"""
Binance Trading Bot — Order Executor
======================================
Handles order placement, cancellation, and status checking on Binance.
All orders are logged to the trade journal.
"""

import time
from binance.spot import Spot
from binance.error import ClientError, ServerError

import config
from core.data_engine import BinanceDataEngine
from core.risk_manager import OrderParams
from core.utils import get_logger, log_trade, now_str

logger = get_logger("OrderExecutor")


class OrderExecutor:
    """
    Executes trades on Binance with retry logic and order logging.
    """

    def __init__(self, data_engine: BinanceDataEngine):
        self.engine = data_engine
        self.client = data_engine.client

        # Get symbol precision for proper quantity formatting
        self.step_size, self.qty_precision = data_engine.get_step_size()
        logger.info(
            "📐 Order precision: step_size=%.8f, qty_decimals=%d",
            self.step_size, self.qty_precision,
        )

    def _round_quantity(self, quantity: float) -> float:
        """Round quantity to the symbol's allowed step size."""
        return round(quantity, self.qty_precision)

    def place_market_order(self, order_params: OrderParams) -> dict:
        """
        Place a MARKET order on Binance.

        Returns the order response dict, or empty dict on failure.
        """
        if not self.engine.authenticated:
            logger.warning("⚠ Cannot place orders without API keys — simulating order")
            return self._simulate_order(order_params)

        qty = self._round_quantity(order_params.quantity)

        try:
            logger.info(
                "🚀 Placing MARKET %s order: %s %s",
                order_params.side, f"{qty:.{self.qty_precision}f}", order_params.symbol,
            )

            response = self.client.new_order(
                symbol=order_params.symbol,
                side=order_params.side,
                type="MARKET",
                quantity=qty,
            )

            order_id = response.get("orderId", "N/A")
            status = response.get("status", "UNKNOWN")
            filled_price = self._get_avg_fill_price(response)

            logger.info(
                "✅ Order filled: ID=%s | Status=%s | Avg Price: $%.2f",
                order_id, status, filled_price,
            )

            # Log to trade journal
            log_trade({
                "timestamp": now_str(),
                "side": order_params.side,
                "symbol": order_params.symbol,
                "entry_price": filled_price or order_params.entry_price,
                "quantity": qty,
                "stop_loss": order_params.stop_loss,
                "take_profit": order_params.take_profit,
                "signal_score": "",
                "status": "OPEN",
                "notes": f"OrderID: {order_id}",
            })

            return response

        except ClientError as e:
            logger.error("❌ Client error placing order: %s (code: %s)", e.error_message, e.error_code)
            return {}
        except ServerError as e:
            logger.error("❌ Server error placing order: %s", e)
            return self._retry_order(order_params, retries=2)

    def place_close_order(self, side: str, quantity: float, symbol: str = None) -> dict:
        """
        Place a closing order (opposite side of the position).

        Args:
            side: "BUY" or "SELL" (the closing side)
            quantity: Amount to close
            symbol: Trading pair
        """
        symbol = symbol or config.TRADING_PAIR
        qty = self._round_quantity(quantity)

        if not self.engine.authenticated:
            logger.info("📝 Simulated close order: %s %.6f %s", side, qty, symbol)
            return {"simulated": True, "side": side, "quantity": qty}

        try:
            response = self.client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty,
            )
            logger.info("✅ Close order placed: %s %.6f %s", side, qty, symbol)
            return response

        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to place close order: %s", e)
            return {}

    def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all open orders for a symbol."""
        symbol = symbol or config.TRADING_PAIR
        if not self.engine.authenticated:
            return True

        try:
            self.client.cancel_open_orders(symbol=symbol)
            logger.info("🗑 All open orders cancelled for %s", symbol)
            return True
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to cancel orders: %s", e)
            return False

    def get_order_status(self, order_id: int, symbol: str = None) -> dict:
        """Check the status of a specific order."""
        symbol = symbol or config.TRADING_PAIR
        if not self.engine.authenticated:
            return {}

        try:
            return self.client.get_order(symbol=symbol, orderId=order_id)
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to get order status: %s", e)
            return {}

    def _get_avg_fill_price(self, response: dict) -> float:
        """Extract average fill price from order response."""
        fills = response.get("fills", [])
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
            return total_cost / total_qty if total_qty > 0 else 0
        # Fallback to cummulativeQuoteQty / executedQty
        cum_quote = float(response.get("cummulativeQuoteQty", 0))
        exec_qty = float(response.get("executedQty", 0))
        return cum_quote / exec_qty if exec_qty > 0 else 0

    def _simulate_order(self, order_params: OrderParams) -> dict:
        """Simulate an order for unauthenticated mode (for testing logic)."""
        logger.info(
            "📝 SIMULATED %s order: %.6f %s @ $%.2f | SL: $%.2f | TP: $%.2f",
            order_params.side,
            order_params.quantity,
            order_params.symbol,
            order_params.entry_price,
            order_params.stop_loss,
            order_params.take_profit,
        )

        log_trade({
            "timestamp": now_str(),
            "side": order_params.side,
            "symbol": order_params.symbol,
            "entry_price": order_params.entry_price,
            "quantity": order_params.quantity,
            "stop_loss": order_params.stop_loss,
            "take_profit": order_params.take_profit,
            "signal_score": "",
            "status": "OPEN (SIM)",
            "notes": "Simulated — no API key",
        })

        return {
            "simulated": True,
            "side": order_params.side,
            "quantity": order_params.quantity,
            "price": order_params.entry_price,
        }

    def _retry_order(self, order_params: OrderParams, retries: int = 2) -> dict:
        """Retry placing an order after a server error."""
        for attempt in range(1, retries + 1):
            logger.warning("🔄 Retrying order (attempt %d/%d)...", attempt, retries)
            time.sleep(2 * attempt)  # exponential backoff
            try:
                qty = self._round_quantity(order_params.quantity)
                response = self.client.new_order(
                    symbol=order_params.symbol,
                    side=order_params.side,
                    type="MARKET",
                    quantity=qty,
                )
                logger.info("✅ Retry successful on attempt %d", attempt)
                return response
            except (ClientError, ServerError) as e:
                logger.error("❌ Retry %d failed: %s", attempt, e)

        logger.error("❌ All retry attempts exhausted")
        return {}
