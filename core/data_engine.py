"""
Binance Trading Bot — Data Engine
==================================
Handles all Binance API interaction: market data, account info, and order queries.
"""

import pandas as pd
from binance.spot import Spot
from binance.error import ClientError, ServerError

import config
from core.utils import get_logger, ms_to_datetime

logger = get_logger("DataEngine")


class BinanceDataEngine:
    """Manages connection and data retrieval from Binance API."""

    def __init__(self):
        base_url = config.TESTNET_BASE_URL if config.USE_TESTNET else config.LIVE_BASE_URL

        if not config.BINANCE_API_KEY or config.BINANCE_API_KEY == "your_api_key_here":
            logger.warning(
                "⚠ No API key configured! Running in public-data-only mode. "
                "Edit .env to add your keys."
            )
            self.client = Spot(base_url=base_url)
            self.authenticated = False
        else:
            self.client = Spot(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_API_SECRET,
                base_url=base_url,
            )
            self.authenticated = True

        mode = "TESTNET" if config.USE_TESTNET else "LIVE"
        logger.info("🔗 Connected to Binance (%s)", mode)

    def get_klines(
        self,
        symbol: str = None,
        interval: str = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data from Binance.

        Returns a DataFrame with columns:
            open_time, open, high, low, close, volume, close_time
        """
        symbol = symbol or config.TRADING_PAIR
        interval = interval or config.TIMEFRAME
        limit = limit or config.KLINE_LIMIT

        try:
            raw = self.client.klines(symbol=symbol, interval=interval, limit=limit)
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to fetch klines: %s", e)
            return pd.DataFrame()

        if not raw:
            logger.warning("⚠ No kline data returned for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades_count",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])

        # Convert types
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["open_time"] = df["open_time"].apply(ms_to_datetime)
        df["close_time"] = df["close_time"].apply(ms_to_datetime)

        # Keep only relevant columns
        df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
        df.set_index("open_time", inplace=True)

        logger.debug("📊 Fetched %d candles for %s (%s)", len(df), symbol, interval)
        return df

    def get_current_price(self, symbol: str = None) -> float:
        """Get the latest ticker price for a symbol."""
        symbol = symbol or config.TRADING_PAIR
        try:
            ticker = self.client.ticker_price(symbol=symbol)
            price = float(ticker["price"])
            logger.debug("💲 %s price: %.2f", symbol, price)
            return price
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to get price: %s", e)
            return 0.0

    def get_account_balance(self, asset: str = None) -> float:
        """Get the available balance for a specific asset."""
        if not self.authenticated:
            logger.warning("⚠ Cannot fetch balance without API keys")
            return 0.0

        asset = asset or config.QUOTE_ASSET
        try:
            account = self.client.account()
            for balance in account.get("balances", []):
                if balance["asset"] == asset:
                    free = float(balance["free"])
                    logger.debug("💰 %s balance: %.4f", asset, free)
                    return free
            logger.warning("⚠ Asset %s not found in account", asset)
            return 0.0
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to fetch balance: %s", e)
            raise e

    def get_all_balances(self) -> dict:
        """Get all non-zero balances from the account."""
        if not self.authenticated:
            return {}

        try:
            account = self.client.account()
            balances = {}
            for b in account.get("balances", []):
                free = float(b["free"])
                locked = float(b["locked"])
                if free > 0 or locked > 0:
                    balances[b["asset"]] = {"free": free, "locked": locked}
            return balances
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to fetch balances: %s", e)
            return {}

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open orders for a symbol."""
        if not self.authenticated:
            return []

        symbol = symbol or config.TRADING_PAIR
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            logger.debug("📋 Open orders for %s: %d", symbol, len(orders))
            return orders
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to fetch open orders: %s", e)
            return []

    def get_symbol_info(self, symbol: str = None) -> dict:
        """Get exchange info for a symbol (precision, min qty, etc.)."""
        symbol = symbol or config.TRADING_PAIR
        try:
            info = self.client.exchange_info(symbol=symbol)
            for s in info.get("symbols", []):
                if s["symbol"] == symbol:
                    return s
            return {}
        except (ClientError, ServerError) as e:
            logger.error("❌ Failed to fetch symbol info: %s", e)
            return {}

    def get_step_size(self, symbol: str = None) -> tuple[float, int]:
        """
        Get the minimum step size and precision for order quantity.

        Returns:
            (step_size, precision) — e.g. (0.00001, 5)
        """
        info = self.get_symbol_info(symbol)
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                precision = len(f["stepSize"].rstrip("0").split(".")[-1])
                return step, precision
        return 0.00001, 5  # safe defaults

    def ping(self) -> bool:
        """Check if the Binance API is reachable."""
        try:
            self.client.ping()
            return True
        except Exception:
            return False
