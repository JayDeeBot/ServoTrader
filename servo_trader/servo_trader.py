import time
import pandas as pd
from requests.exceptions import ConnectionError, Timeout
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class ServoTrader:
    RED_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"

    def __init__(self, api_key, secret_key, prediction, paper=False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.recommended_crypto, self.expected_growth = prediction
        self.portfolio = self.get_portfolio()
        self.cash_balance = self.get_cash_balance()

    def retry_with_backoff(self, func, *args, max_retries=5, **kwargs):
        retries = 0
        delay = 2  # Start with 2 seconds delay

        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except (ConnectionError, Timeout) as e:
                print(f"{self.RED_COLOR}Network error: {e}. Retrying in {delay}s...{self.RESET_COLOR}")
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    print(f"{self.RED_COLOR}Rate limit exceeded. Retrying in {delay}s...{self.RESET_COLOR}")
                else:
                    print(f"{self.RED_COLOR}Unexpected error: {e}{self.RESET_COLOR}")
                    break

            time.sleep(delay)
            retries += 1
            delay *= 2  # Exponential backoff

        print(f"{self.RED_COLOR}Max retries reached. Operation failed.{self.RESET_COLOR}")
        return None

    # def get_portfolio(self):
    #     return self.retry_with_backoff(self.trading_client.get_all_positions)

    def get_portfolio(self):
        positions = self.retry_with_backoff(self.trading_client.get_all_positions)

        if positions is None:
            return {}  # Return empty dict if no positions available

        # Convert list of positions into a dictionary with symbol as the key
        portfolio_dict = {pos.symbol: {"qty": float(pos.qty), "value": float(pos.market_value)} for pos in positions}
        return portfolio_dict

    def get_cash_balance(self):
        account = self.retry_with_backoff(self.trading_client.get_account)
        return float(account.cash) if account else 0.0

    def execute_sell(self, symbol, fractional_qty, limit_price):
        if fractional_qty > 0:
            print(f"Selling {fractional_qty:.8f} of {symbol} at limit price ${limit_price}")
            
            sell_order = LimitOrderRequest(
                symbol=symbol,
                qty=fractional_qty,
                side=OrderSide.SELL,
                limit_price=limit_price,
                time_in_force=TimeInForce.GTC
            )

            response = self.retry_with_backoff(self.trading_client.submit_order, sell_order)

            if response:
                try:
                    order_id = str(response.id)  # Ensure it's a string representation of UUID
                    print(f"Sell Order Placed: {order_id}")
                    return order_id
                except AttributeError:
                    print(f"{self.RED_COLOR}Error: Sell order response missing 'id' attribute.{self.RESET_COLOR}")

        print("Invalid quantity. Cannot sell zero or negative amounts.")
        return None

    def execute_buy(self, symbol):
        if self.cash_balance > 0:
            try:
                crypto_data = pd.read_csv('crypto_data.csv')
                latest_price = crypto_data[symbol].iloc[-1]
                rounded_price = round(latest_price, 4)
                print(f"Buying {symbol} at ${rounded_price} with ${self.cash_balance} available.")

                buy_order = LimitOrderRequest(
                    symbol=symbol,
                    notional=self.cash_balance,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    limit_price=rounded_price
                )
                response = self.retry_with_backoff(self.trading_client.submit_order, buy_order)
                
                # Add this line to inspect the response
                # print(f"Order Response: {response}")

                if response and hasattr(response, 'id'):
                    return str(response.id)  # Ensure UUID string format
                else:
                    print(f"{self.RED_COLOR}Order did not return a valid ID.{self.RESET_COLOR}")
                    return None

            except (FileNotFoundError, KeyError, IndexError) as e:
                print(f"{self.RED_COLOR}Error with crypto_data.csv: {e}{self.RESET_COLOR}")
        else:
            print("Insufficient funds to make a trade.")
        return None
    
    def execute_market_sell(self, symbol):
        """
        Performs a market sell to quickly liquidate a crypto position.
        """
        self.portfolio = self.get_portfolio()  # Refresh portfolio to get real-time holdings
        fractional_qty = self.portfolio.get(symbol, {}).get("qty", 0)

        if fractional_qty > 0:
            print(f"{self.RED_COLOR}Performing MARKET SELL of {fractional_qty} {symbol} to quickly exit position.{self.RESET_COLOR}")
            market_sell_order = MarketOrderRequest(
                symbol=symbol,
                qty=fractional_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            return self.retry_with_backoff(self.trading_client.submit_order, market_sell_order)

        print(f"{self.RED_COLOR}No holdings available to market sell for {symbol}.{self.RESET_COLOR}")
        return None