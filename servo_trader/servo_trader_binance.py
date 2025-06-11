"""
servo_trader_binance.py

This module implements the ServoTraderBinance class, a concrete subclass of ServoTraderInterface.
It provides a complete Binance-specific implementation of all required trading functions,
including market buy/sell execution, order status tracking, cancellation, and asset rounding logic
based on Binance's trading filters (e.g., LOT_SIZE).

The class is designed to integrate with the ServoCrypto trading platform and supports high-frequency,
short-term trading strategies on live markets using the Binance REST API.

Key Features:
- Executes market buy/sell orders with real-time balance checks
- Loads and uses Binance step sizes to round order quantities for compliance
- Tracks open orders and maps their IDs to trading symbols for status/cancellation
- Provides utility for calculating the maximum saleable quantity for any crypto symbol
- Uses robust error handling and internal logging for fault tolerance

Dependencies:
- binance.client.Client (from the `python-binance` package)
- Requires a valid Binance API key and secret

Author: Jarred Deluca
Project: ServoCrypto
Created: 2025-06-07
"""

# servo_trader_binance.py

from binance.client import Client # Import the binance API
from .servo_trader_interface import ServoTraderInterface # Import the parent class (trading interface)
import time

class ServoTraderBinance(ServoTraderInterface):
    """
    ServoTrader implementation class for the Binance API. 
    This class implements all virtual functions in the ServoTraderInterface class for application in the main.
    Inherits from ServoTraderInterface.
    """
    RED_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"

    def __init__(self, api_key: str, secret_key: str):
        """
        Constructor for the trader object.
        Initialises the client for connection to Binance, fetched the current portfolio and current cash balance.

        Args:
            api_key (str): Binance unique api key for users account
            secret_key (str): Binance unique secret key for users account
        """
        self.client = Client(api_key, secret_key) # Init the API Client - connection to binance
        self.portfolio = self.get_portfolio() # Fetch the current portfolio - all positions
        self.cash_balance = self.get_cash_balance() # Fetch the current cash balance
        self.step_sizes = self._load_step_sizes() # Load the step sizes for all available cryptos (assists with calculating the saleable qty)
        self.order_symbol_map = {} # Init an order symbol map to store our pending orders

    def get_portfolio(self) -> dict:
        """
        Retrieve the current portfolio holdings.

        Returns:
            dict: A mapping of crypto symbols to quantity and market value.
        """
        try:
            account_info = self.client.get_account() # Fetch the entire account status
            holdings = { # Perform dictionary comprehension and save the result to holdings
                asset['asset']: { # Grab the assets ID
                    "qty": float(asset['free']) + float(asset['locked']), # Calculate the held qty of the asset
                }
                for asset in account_info['balances'] # Loop through all asset entries in account info
                if float(asset['free']) + float(asset['locked']) > 0 # Only save the entry if the qty > 0
            }
            return holdings # Return the refined dictionary
        except Exception as e:
            print(f"{self.RED_COLOR}Error fetching portfolio: {e}{self.RESET_COLOR}") # Report error
            return {} # If there is an error return nothing

    def get_cash_balance(self) -> float:
        """
        Get the current available cash balance in the trading account.

        Returns:
            float: The available cash.
        """
        try:
            account_info = self.client.get_account() # Fetch the entire account status
            usdt_balance = next((float(a['free']) for a in account_info['balances'] if a['asset'] == 'USDT'), 0.0) # Grab USDT balance
            return usdt_balance # Return USDT balance
        except Exception as e:
            print(f"{self.RED_COLOR}Error fetching USDT balance: {e}{self.RESET_COLOR}") # Report error
            return 0.0 # If there is an error return 0.0

    def execute_buy(self, symbol: str) -> str:
        """
        Execute a market buy order for the given symbol using available cash.

        Args:
            symbol (str): The crypto symbol to buy.

        Returns:
            str: The unique order ID of the buy order.
        """
        try:
            cash = self.get_cash_balance() # Fetch the accounts cash balance
            if cash <= 0: # If the cash balance is 0 or less we cannot buy
                print("No funds available for market buy.")
                return None # Return none as the buy order id

            print(f"Placing market buy for {symbol} with ${cash:.2f}")
            order = self.client.create_order( # Place the buy order
                symbol=symbol, # Crypto symbol to buy
                side='BUY', # Set side as 'BUY' for buy order
                type='MARKET', # Set type as 'MARKET' for a buy order at market value
                quoteOrderQty=cash # Set the order qty as our fetced cash balance
            )
            self.order_symbol_map[str(order['orderId'])] = symbol # Save the order ID to member
            return str(order['orderId']) # Return the order ID
        except Exception as e:
            print(f"{self.RED_COLOR}Error placing market buy: {e}{self.RESET_COLOR}") # Report error
            return None # If there is an error return none
 
    def execute_sell(self, symbol: str, fractional_qty: float) -> str:
        """
        Execute a market sell order for the given symbol.

        Args:
            symbol (str): The crypto symbol to sell.
            fractional_qty (float): The quantity to sell.

        Returns:
            str: The unique order ID of the market sell order.
        """
        try:
            if fractional_qty <= 0: # If the fractional qty argument is 0 or below we cannot sell
                print("Cannot sell zero or negative quantity.")
                return None # Return none as the sell order ID

            print(f"Placing market sell for {fractional_qty:.8f} of {symbol}")
            order = self.client.create_order( # Place the sell order
                symbol=symbol, # Crypto symbol to buy
                side='SELL', # Set side as 'SELL' for buy order
                type='MARKET', # Set type as 'MARKET' for a sell order at market value
                quantity=fractional_qty # Set the order qty as the fractional qty argument supplied
            )
            self.order_symbol_map[str(order['orderId'])] = symbol # Save the order ID to member
        except Exception as e:
            print(f"{self.RED_COLOR}Error placing market sell: {e}{self.RESET_COLOR}") # Report error
            return None # If there is an error return none

    def cancel_order(self, order_id: str) -> None:
        """
        Cancel an existing order by its ID.

        Args:
            order_id (str): The unique ID of the order to cancel.
        """
        try:
            symbol = self.order_symbol_map.get(order_id) # Grab the symbol associated with the order ID argument from symbol map member
            if not symbol: # If there is no symbol found in the map for this ID report the issue
                print(f"{self.RED_COLOR}Symbol for order {order_id} not found. Cannot cancel.{self.RESET_COLOR}")
                return # Return immediately

            self.client.cancel_order(symbol=symbol, orderId=int(order_id)) # Cancel the order using the symbol and ID
            print(f"Order {order_id} for {symbol} cancelled successfully.")
        except Exception as e:
            print(f"{self.RED_COLOR}Error cancelling order: {e}{self.RESET_COLOR}") # Report error

    def get_order_by_id(self, order_id: str):
        """
        Retrieve the status/details of an order by its ID.

        Args:
            order_id (str): The unique ID of the order.

        Returns:
            Any: A platform-specific order object or status dictionary.
        """
        try:
            symbol = self.order_symbol_map.get(order_id) # Grab the symbol associated with the order ID argument from symbol map member
            if not symbol: # If there is no symbol found in the map for this ID report the issue
                print(f"{self.RED_COLOR}Symbol for order {order_id} not found. Cannot retrieve status.{self.RESET_COLOR}")
                return None # Return none

            order = self.client.get_order(symbol=symbol, orderId=int(order_id)) # Fetch the order details using the symbol and ID
            print(f"Order status: {order['status']}")
            return order # Return the status
        except Exception as e:
            print(f"{self.RED_COLOR}Error fetching order: {e}{self.RESET_COLOR}") # Report error
            return None # If there is an error return none

    def _load_step_sizes(self):
        """
        Preload step sizes for all tradeable symbols.
        Returns:
            dict: A mapping of symbols to their step sizes.
        """
        step_map = {} # Init a step map for saving the step size for each symbol
        try:
            exchange_info = self.client.get_exchange_info() # Fetch the exchange info
            for symbol_data in exchange_info['symbols']: # Loop through all symbols available in the exchange
                symbol = symbol_data['symbol'] # Save the symbol for the crypto in question
                lot_filter = next((f for f in symbol_data['filters'] if f['filterType'] == 'LOT_SIZE'), None) # Loop through the available data and grab the 'LOT_SIZE'
                if lot_filter: # If a lot size was found save it to the step map and label it with the saved symbol
                    step_map[symbol] = lot_filter['stepSize']
        except Exception as e:
            print(f"{self.RED_COLOR}Error loading step sizes: {e}{self.RESET_COLOR}") # Report error
        return step_map # If there is an error return the incomplete step map
    
    def get_saleable_quantity(self, symbol: str) -> float:
        """
        Return the saleable quantity for a given symbol, rounded to step size.
        """
        try:
            asset = symbol.replace("USDT", "")  # crude parsing
            raw_qty = self.portfolio.get(asset, {}).get("qty", 0.0)
            step_size = self.step_sizes.get(symbol, '0.00000001')  # fallback
            rounded_qty = self.round_step_size(raw_qty, step_size)

            print(f"[Debug] Raw qty: {raw_qty}, Step size: {step_size}, Rounded: {rounded_qty}")
            return rounded_qty
        except Exception as e:
            print(f"{self.RED_COLOR}Error calculating saleable quantity for {symbol}: {e}{self.RESET_COLOR}")
            return 0.0