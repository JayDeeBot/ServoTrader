# servo_trader_interface.py

"""
ServoTraderInterface

This module defines the abstract base class for the ServoTrader trading system.
It specifies the required interface for any exchange integration used in the
ServoCrypto platform. The purpose of this interface is to enable modular and
interchangeable API support for multiple crypto trading platforms (e.g., Alpaca,
Binance, Kraken), facilitating scalability and commercial readiness.

All concrete implementations must inherit from this class and provide
exchange-specific logic for the defined trading operations.

Assumptions:
- All trades (buys and sells) are executed using market orders for real-time performance.
- Trading is conducted in fractional quantities.
- Implementations should handle connection retries and error management internally.

Usage:
    class ServoTraderBinance(ServoTraderInterface):
        def get_portfolio(self): ...
        def get_cash_balance(self): ...
        def execute_buy(self, symbol): ...
        def execute_sell(self, symbol, fractional_qty): ...
        def cancel_order(self, order_id): ...
        def get_order_by_id(self, order_id): ...
        def get_saleable_quantity(self, symbol): ...
        def round_step_size(quantity, step_size): ...
        def _load_step_sizes(self): ...

Author: Jarred Deluca
Project: ServoCrypto
Created: 2025-06-07
"""

from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_DOWN, getcontext

class ServoTraderInterface(ABC):
    """
    Abstract base class defining the trading interface for crypto platforms.
    This interface assumes market buy and sell orders for fast execution in RL environments.
    """

    @abstractmethod
    def get_portfolio(self) -> dict:
        """
        Retrieve the current portfolio holdings.

        Returns:
            dict: A mapping of crypto symbols to quantity and market value.
        """
        pass

    @abstractmethod
    def get_cash_balance(self) -> float:
        """
        Get the current available cash balance in the trading account.

        Returns:
            float: The available cash.
        """
        pass

    @abstractmethod
    def execute_buy(self, symbol: str) -> str:
        """
        Execute a market buy order for the given symbol using available cash.

        Args:
            symbol (str): The crypto symbol to buy.

        Returns:
            str: The unique order ID of the buy order.
        """
        pass

    @abstractmethod
    def execute_sell(self, symbol: str, fractional_qty: float) -> str:
        """
        Execute a market sell order for the given symbol.

        Args:
            symbol (str): The crypto symbol to sell.
            fractional_qty (float): The quantity to sell.

        Returns:
            str: The unique order ID of the market sell order.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """
        Cancel an existing order by its ID.

        Args:
            order_id (str): The unique ID of the order to cancel.
        """
        pass

    @abstractmethod
    def get_order_by_id(self, order_id: str):
        """
        Retrieve the status/details of an order by its ID.

        Args:
            order_id (str): The unique ID of the order.

        Returns:
            Any: A platform-specific order object or status dictionary.
        """
        pass

    @abstractmethod
    def get_saleable_quantity(self, symbol: str) -> float:
        """
        Calculate the amount of a given crypto symbol that is currently saleable,
        rounded to meet exchange constraints (e.g., step size).

        Args:
            symbol (str): The crypto trading symbol (e.g., 'BTCUSDT').

        Returns:
            float: The valid, saleable quantity of the crypto asset.
        """
        pass
    
    @abstractmethod
    def _load_step_sizes(self):
        """
        Preload step sizes for all tradeable symbols.
        Returns:
            dict: A mapping of symbols to their step sizes.
        """
        pass

    @staticmethod
    def round_step_size(quantity, step_size):
        """
        Rounds a quantity down to the nearest allowed step size.

        Args:
            quantity (float): The amount to round.
            step_size (str): The step size as a string, e.g., '0.00000001'.

        Returns:
            float: The rounded quantity.
        """
        getcontext().prec = 20  # ensure precision
        quantity = Decimal(str(quantity))
        step_size = Decimal(str(step_size))
        rounded = (quantity // step_size) * step_size  # floor to step multiple
        return float(rounded)