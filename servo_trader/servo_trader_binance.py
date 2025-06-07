# servo_trader_binance.py

from binance.client import Client # Import the binance API
from servo_trader_interface import ServoTraderInterface # Import the parent class (trading interface)
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
        self.client = Client(api_key, secret_key) # Init the API Client - connection to binance.
        self.portfolio = self.get_portfolio() # Fetch the current portfolio - all positions
        self.cash_balance = self.get_cash_balance() # Fetch the current cash balance

    def get_portfolio(self) -> dict:
        """
        Retrieve the current portfolio holdings.

        Returns:
            dict: A mapping of crypto symbols to quantity and market value.
        """
        try: # Try and catch any exceptions
            account_info = self.client.get_account() # Fetch the account info - including all positions
            holdings = { 
                asset['asset']: {
                    "qty": float(asset['free']) + float(asset['locked']),
                    "value": None  # Market value requires price lookup
                }
                for asset in account_info['balances']
                if float(asset['free']) + float(asset['locked']) > 0
            }
            return holdings
        except Exception as e:
            print(f"{self.RED_COLOR}Error fetching portfolio: {e}{self.RESET_COLOR}")
            return {}

    def get_cash_balance(self) -> float:
        """
        Get the current available cash balance in the trading account.

        Returns:
            float: The available cash.
        """
        try:
            account_info = self.client.get_account()
            usdt_balance = next((float(a['free']) for a in account_info['balances'] if a['asset'] == 'USDT'), 0.0)
            return usdt_balance
        except Exception as e:
            print(f"{self.RED_COLOR}Error fetching USDT balance: {e}{self.RESET_COLOR}")
            return 0.0

    def execute_buy(self, symbol: str) -> str:
        """
        Execute a market buy order for the given symbol using available cash.

        Args:
            symbol (str): The crypto symbol to buy.

        Returns:
            str: The unique order ID of the buy order.
        """
        try:
            cash = self.get_cash_balance()
            if cash <= 0:
                print("No funds available for market buy.")
                return None

            print(f"Placing market buy for {symbol} with ${cash:.2f}")
            order = self.client.create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quoteOrderQty=cash
            )
            return str(order['orderId'])
        except Exception as e:
            print(f"{self.RED_COLOR}Error placing market buy: {e}{self.RESET_COLOR}")
            return None

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
            if fractional_qty <= 0:
                print("Cannot sell zero or negative quantity.")
                return None

            print(f"Placing market sell for {fractional_qty:.8f} of {symbol}")
            order = self.client.create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=fractional_qty
            )
            return str(order['orderId'])
        except Exception as e:
            print(f"{self.RED_COLOR}Error placing market sell: {e}{self.RESET_COLOR}")
            return None

    def cancel_order(self, order_id: str) -> None:
        """
        Cancel an existing order by its ID.

        Args:
            order_id (str): The unique ID of the order to cancel.
        """
        try:
            # Binance requires the symbol to cancel the order
            # You may need to store order-symbol mappings if cancelling later
            print(f"{self.RED_COLOR}Canceling order {order_id}: Binance requires symbol context — not implemented here.{self.RESET_COLOR}")
        except Exception as e:
            print(f"{self.RED_COLOR}Error cancelling order: {e}{self.RESET_COLOR}")

    def get_order_by_id(self, order_id: str):
        """
        Retrieve the status/details of an order by its ID.

        Args:
            order_id (str): The unique ID of the order.

        Returns:
            Any: A platform-specific order object or status dictionary.
        """
        try:
            print(f"{self.RED_COLOR}Fetching order by ID requires symbol context in Binance. Not implemented.{self.RESET_COLOR}")
        except Exception as e:
            print(f"{self.RED_COLOR}Error fetching order: {e}{self.RESET_COLOR}")
