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
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Tuple, List, Dict, Any
import ccxt


class ServoTraderBinance(ServoTraderInterface):
    """
    ServoTrader implementation class for the Binance API. 
    This class implements all virtual functions in the ServoTraderInterface class for application in the main.
    Inherits from ServoTraderInterface.
    """
    RED_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"

    def __init__(self, params_path: str = "/home/jarred/git/ServoTrader/servo_trader/config/params.yaml"):
        """
        Constructor for the trader object.
        Initialises the client for connection to Binance, fetched the current portfolio and current cash balance.

        Args:
            params_path (str): Path to the params YAML file (default: params.yaml)
        """
        # Load YAML parameters
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to load params from {params_path}: {e}")

        # Retrieve API credentials
        api_key = params.get("BINANCE_API_KEY")
        secret_key = params.get("BINANCE_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError("BINANCE_API_KEY and BINANCE_SECRET_KEY must be provided in params.yaml.")

        # Optional flags/settings (with sensible defaults)
        use_testnet: bool = bool(params.get("BINANCE_TESTNET", False))
        default_type: str = str(params.get("BINANCE_DEFAULT_TYPE", "spot"))  # ccxt option

        # --- Python-Binance client (existing flow) ---
        # Init the API Client - connection to binance
        self.client = Client(api_key, secret_key)

        # --- ccxt client (for unified fetch_balance/fetch_tickers, used by get_estimated_balance_usdt) ---
        try:
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": secret_key,
                "enableRateLimit": True,
                "options": {
                    "defaultType": default_type,  # 'spot' by default
                },
            })
            if use_testnet:
                # Put ccxt in sandbox if requested; ignore if not supported
                try:
                    self.exchange.set_sandbox_mode(True)
                except Exception:
                    pass
        except Exception as e:
            # Surface a clear error if ccxt cannot be initialized
            raise RuntimeError(f"Failed to initialize ccxt binance client: {e}")

        # Lightweight in-memory cache for pricing used by get_estimated_balance_usdt()
        # Structure: {"ts": unix_seconds, "tickers": {...}}
        self._price_cache = {"ts": 0.0, "tickers": {}}

        # Fetch the current portfolio & cash balance using your existing helpers
        self.portfolio = self.get_portfolio()     # Fetch the current portfolio - all positions
        self.cash_balance = self.get_cash_balance()  # Fetch the current cash balance

        # Init an order symbol map to store our pending orders
        self.order_symbol_map = {}

        # Trading limits (GUI can update these)
        self.max_buy_limit = 1e6           # Cap any buy orders to $1 mil max
        self.daily_cashout_percent = 0.0   # Percentage of the cash balance to cash out daily (start at 0%)

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
        Execute a market buy order for the given symbol using the lesser of:
        - The available account cash balance.
        - A predefined maximum buy limit set in self.max_buy_limit.

        Args:
            symbol (str): The crypto symbol to buy.

        Returns:
            str: The unique order ID of the buy order, or None if the order fails or funds are insufficient.
        """
        try:
            cash = self.get_cash_balance()  # Fetch the account's current cash balance
            buy_limit = getattr(self, 'max_buy_limit', cash)  # Use limit if it exists, otherwise allow full cash
            buy_amount = min(cash, buy_limit)  # Use the smaller of available cash and buy limit

            if buy_amount <= 0:
                print("No valid funds available for market buy.")
                return None

            print(f"Placing market buy for {symbol} with ${buy_amount:.2f}")
            order = self.client.create_order(  # Place the buy order
                symbol=symbol,                # Crypto symbol to buy
                side='BUY',                   # Set side as 'BUY' for buy order
                type='MARKET',                # Set type as 'MARKET' for a buy order at market value
                quoteOrderQty=buy_amount      # Set the order qty to the capped amount
            )
            self.order_symbol_map[str(order['orderId'])] = symbol  # Save the order ID to member
            return str(order['orderId'])  # Return the order ID
        except Exception as e:
            print(f"{self.RED_COLOR}Error placing market buy: {e}{self.RESET_COLOR}")  # Report error
            return None

    def execute_sell(self, symbol: str) -> str | None:
        """
        Execute a market sell order for the full available quantity of `symbol`,
        trying progressively coarser decimal precision **rounded down** until the
        exchange accepts the LOT_SIZE.

        Logic:
        1) Get free balance (e.g., 42.816).
        2) Try selling exactly that amount (same decimals).
        3) On LOT_SIZE failure, drop one decimal place and floor (42.81).
        4) On failure again, drop to one decimal (42.8).
        5) On failure again, drop to integer (42).
        6) If still failing, give up.

        Notes:
        - We avoid rounding up to ensure we never exceed available balance.
        - Quantity is sent as a string to avoid float representation issues.
        - This is intentionally minimal and aligns with the requested behavior.
        """
        try:
            # --- Resolve asset and free balance ---
            asset = symbol.replace("USDT", "")
            balance_info = self.client.get_asset_balance(asset=asset)
            if not balance_info:
                print(f"{self.RED_COLOR}Could not retrieve balance for {asset}{self.RESET_COLOR}")
                return None

            try:
                free_qty = Decimal(str(balance_info.get("free", "0")))
            except (InvalidOperation, TypeError):
                print(f"{self.RED_COLOR}Invalid balance for {asset}{self.RESET_COLOR}")
                return None

            if free_qty <= 0:
                print("Cannot sell zero or negative quantity.")
                return None

            # Determine how many decimal places are present in the free balance.
            # Example: 42.816 -> exponent = -3  => max_decimals = 3
            max_decimals = max(0, -free_qty.as_tuple().exponent)

            # Try with current decimals, then 1 fewer, etc., down to 0.
            for decimals in range(max_decimals, -1, -1):
                # Build a quantize exponent for this precision:
                # decimals=3 -> Decimal('1E-3'), decimals=1 -> Decimal('1E-1'), decimals=0 -> Decimal('1')
                quant = Decimal((0, (1,), -decimals))
                qty = free_qty.quantize(quant, rounding=ROUND_DOWN)

                # Skip if quantized to zero (shouldn't happen for your case, but safe-guard).
                if qty <= 0:
                    continue

                try:
                    qty_str = format(qty.normalize(), 'f')  # exact string (no scientific notation)
                    print(f"Placing market sell for {qty_str} of {symbol}")
                    order = self.client.create_order(
                        symbol=symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=qty_str
                    )
                    self.order_symbol_map[str(order["orderId"])] = symbol
                    return str(order["orderId"])
                except Exception as e:
                    # Common failure here is LOT_SIZE filter; we then try fewer decimals.
                    print(f"{self.RED_COLOR}Failed with {decimals} decimals: {e}{self.RESET_COLOR}")
                    continue

            print(f"{self.RED_COLOR}All attempts to place sell order failed for {symbol}.{self.RESET_COLOR}")
            return None

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
        Retrieve the status and executed price of an order by its ID.

        Args:
            order_id (str): The unique ID of the order.

        Returns:
            dict or None: Dictionary containing status, executedQty, avgPrice, and full order info if successful; None on error.
        """
        try:
            symbol = self.order_symbol_map.get(order_id)  # Get associated trading symbol
            if not symbol:
                print(f"{self.RED_COLOR}Symbol for order {order_id} not found. Cannot retrieve status.{self.RESET_COLOR}")
                return None

            order = self.client.get_order(symbol=symbol, orderId=int(order_id))
            print(f"Order status: {order['status']}")

            # Calculate average fill price if 'fills' field is available (note: not always available in get_order)
            fills = order.get('fills', [])  # Sometimes not returned from get_order
            if fills:
                total_cost = sum(float(fill['price']) * float(fill['qty']) for fill in fills)
                total_qty = sum(float(fill['qty']) for fill in fills)
                avg_price = total_cost / total_qty if total_qty > 0 else 0.0
            else:
                # If 'fills' are not available, fallback to cummulativeQuoteQty / executedQty
                total_cost = float(order.get('cummulativeQuoteQty', 0.0))
                total_qty = float(order.get('executedQty', 0.0))
                avg_price = total_cost / total_qty if total_qty > 0 else 0.0

            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": order['status'],
                "executedQty": order.get('executedQty'),
                "avgPrice": avg_price,
                "raw_order": order
            }
        except Exception as e:
            print(f"{self.RED_COLOR}Error fetching order: {e}{self.RESET_COLOR}")
            return None
        
    def liquify(self):
        """
        Liquidates all non-USDT holdings by selling them for USDT at market price.

        This is typically called at the start of a live trading session to ensure 
        the agent starts from a clean, cash-only portfolio.

        Returns:
            dict: A mapping of symbols successfully liquidated with executed sell order IDs.
        """
        print("💧 Liquifying portfolio...")

        liquidation_results = {}
        self.portfolio = self.get_portfolio()  # Refresh portfolio

        for asset, info in self.portfolio.items():
            if asset == "USDT":
                continue  # Skip base currency

            symbol = asset + "USDT"

            print(f"→ Selling {symbol}...")
            order_id = self.execute_sell(symbol)
            if order_id:
                liquidation_results[symbol] = order_id
            else:
                print(f"{self.RED_COLOR}Failed to sell {symbol}.{self.RESET_COLOR}")

        print("✅ Liquification complete.")
        return liquidation_results
    
    def set_max_buy_limit(self, amount: float) -> None:
        """
        Set the maximum USD amount allowed per market buy order.

        This function allows you to control risk by capping the maximum
        dollar amount that can be used in any single market buy. This limit
        will be applied in the `execute_buy()` method by taking the lesser
        of the account cash balance and this set limit.

        Args:
            amount (float): The maximum amount in USD to spend per market buy.
                            Must be a positive number greater than zero.

        Returns:
            None
        """
        # Validate input to prevent nonsensical values
        if amount <= 0:
            print(f"{self.RED_COLOR}Invalid buy limit. Must be a positive value greater than zero.{self.RESET_COLOR}")
            return

        # Set the instance variable to the specified value
        self.max_buy_limit = amount

        # Confirm that the buy limit has been set
        print(f"✅ Max buy limit successfully set to ${amount:.2f}")

    def set_daily_cashout_percent(self, percent: float) -> None:
        """
        Set the maximum percentage of the total portfolio value that can be
        cashed out (sold) per day.

        This setting is used to throttle how much of the portfolio is allowed
        to be converted to cash each day. It helps enforce conservative sell
        strategies and avoid dumping large portions of assets at once.

        Args:
            percent (float): A float between 0 and 100 representing the maximum
                            percentage of the total portfolio to sell in a day.
                            For example, 10.0 means at most 10% can be sold today.

        Returns:
            None
        """
        # Ensure the percentage value is valid (between 0 and 100)
        if percent <= 0 or percent > 100:
            print(f"{self.RED_COLOR}Invalid percentage. Must be between 0 and 100.{self.RESET_COLOR}")
            return

        # Set the internal member to the validated value
        self.daily_cashout_percent = percent

        # Confirm successful update
        print(f"✅ Daily cashout limit set to {percent:.2f}% of portfolio value.")

    def cash_out_to_bank(self) -> None:
        """
        Cash out funds to the linked bank account based on daily withdrawal rules.

        Cashout logic:
        - If `daily_cashout_percent` == 0:
            → Only cash out the amount that exceeds the `max_buy_limit`.
        - If `daily_cashout_percent` > 0:
            → Cash out either:
                (a) `daily_cashout_percent` of the cash balance, OR
                (b) All cash exceeding `max_buy_limit`, whichever is larger.

        This function is designed to preserve enough capital for trading while allowing
        periodic withdrawals or profit-taking based on your strategy.

        Returns:
            None
        """
        try:
            cash = self.get_cash_balance()  # Get current available cash
            max_limit = getattr(self, 'max_buy_limit', 1e6)  # Fallback to 1 mil if not set
            cashout_percent = getattr(self, 'daily_cashout_percent', 0)  # Fallback to 0 if not set

            # Case 1: No daily cashout percent — only cash out excess cash above the max buy limit
            if cashout_percent == 0:
                if cash > max_limit:
                    amount_to_cash_out = cash - max_limit  # Only take the surplus
                    print(f"Cashing out excess funds: ${amount_to_cash_out:.2f}")
                    self.transfer_to_bank(amount_to_cash_out)
                else:
                    print("✅ No excess funds to cash out today.")
                return

            # Case 2: Daily cashout percent is set (> 0)
            percent_limit = (cash * cashout_percent) / 100  # Calculate allowed withdrawal
            surplus_cash = cash - max_limit  # Cash above the buy limit (if any)

            # Use the larger of the two amounts
            amount_to_cash_out = max(percent_limit, surplus_cash) if surplus_cash > 0 else percent_limit

            if amount_to_cash_out <= 0:
                print("✅ No funds eligible for cash out under current rules.")
                return

            print(f"Initiating daily cashout of ${amount_to_cash_out:.2f}")
            self.transfer_to_bank(amount_to_cash_out)

        except Exception as e:
            print(f"{self.RED_COLOR}Error during cash out to bank: {e}{self.RESET_COLOR}")

    def transfer_to_bank(self, amount: float) -> None:
        """
        Placeholder for actual transfer logic to send funds to your bank.

        Args:
            amount (float): Amount in USD to transfer.

        Returns:
            None
        """
        print(f"🚀 [MOCK] Transferred ${amount:.2f} to your bank account.")

    def find_untradeable_codes(self, code_list: list[str]) -> list[str]:
        """
        Identifies crypto symbols from the provided list that are not currently tradeable on Binance.

        Optimized using multithreading for fast execution and includes a real-time
        percentage completion and ETA display in the console.

        A symbol is considered untradeable if:
        - It does not exist on Binance (i.e. `get_symbol_info()` returns None).
        - Its status is not 'TRADING'.

        Args:
            code_list (list[str]): A list of crypto trading symbols (e.g., ['BTCUSDT', 'ETHUSDT']).

        Returns:
            list[str]: A list of symbols that are not currently tradeable.
        """
        untradeable = []
        total = len(code_list)
        completed = 0
        lock = threading.Lock()
        start_time = time.perf_counter()

        def check_symbol(symbol: str) -> tuple[str, bool]:
            """
            Helper function to check a single symbol's tradeability.

            Returns:
                (symbol, is_untradeable)
            """
            try:
                info = self.client.get_symbol_info(symbol)
                if not info:
                    print(f"❌ {symbol} not found on Binance.")
                    return (symbol, True)

                status = info.get("status", "")
                if status != "TRADING":
                    print(f"⚠️ {symbol} has non-trading status: {status}")
                    return (symbol, True)

                return (symbol, False)

            except Exception as e:
                print(f"{self.RED_COLOR}Error checking {symbol}: {e}{self.RESET_COLOR}")
                return (symbol, True)

        # --- Multithreaded check ---
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(check_symbol, symbol): symbol for symbol in code_list}

            for future in as_completed(futures):
                symbol, is_untradeable = future.result()
                if is_untradeable:
                    untradeable.append(symbol)

                # Update progress and ETA safely
                with lock:
                    completed += 1
                    if completed % 5 == 0 or completed == total:
                        elapsed = time.perf_counter() - start_time
                        percent = (completed / total) * 100
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(f"[{completed}/{total}] {percent:.1f}% complete | ETA: {eta:.1f}s")

        return untradeable
    
    def find_viable_replacement_codes(self, replace_codes: list[str], current_codes: list[str]) -> list[str]:
        """
        Searches for new crypto trading pairs on Binance that can be used to replace outdated or invalid ones.

        The function ensures replacements:
        - Are not already in use (`replace_codes` or `current_codes`)
        - Are USDT pairs (e.g., BTCUSDT)
        - Are currently trading (`status == 'TRADING'`)
        - Are ranked by 24h quote volume to prefer highly liquid pairs

        Returns:
            list[str]: A list of new symbols (equal in length to `replace_codes`) that are safe to use.
        """
        try:
            # Step 1: Build exclusion set
            exclude_set = set(replace_codes + current_codes)
            replacements_needed = len(replace_codes)
            candidate_symbols = []

            # Step 2: Get exchange info to filter for viable symbols
            exchange_info = self.client.get_exchange_info()
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info.get('symbol')

                if (
                    symbol.endswith('USDT') and
                    symbol not in exclude_set and
                    symbol_info.get('status') == 'TRADING'
                ):
                    candidate_symbols.append(symbol)

            print(f"🔍 Found {len(candidate_symbols)} viable USDT symbols for replacement.")

            # Step 3: Get 24h volume for ranking
            tickers = self.client.get_ticker()
            ticker_map = {
                t['symbol']: float(t.get('quoteVolume', 0.0))
                for t in tickers if 'symbol' in t and t['symbol'] in candidate_symbols
            }

            ranked_candidates = sorted(
                ticker_map.keys(),
                key=lambda s: ticker_map[s],
                reverse=True
            )

            replacements = ranked_candidates[:replacements_needed]
            print(f"✅ Selected replacements: {replacements}")
            return replacements

        except Exception as e:
            print(f"{self.RED_COLOR}Error finding replacement codes: {e}{self.RESET_COLOR}")
            return []
        
    def get_estimated_balance_usdt(self, quote: str = "USDT", min_value_usd: float = 0.01, 
                                   price_cache_ttl_s: int = 10,) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Returns the estimated total account equity (spot) valued in `quote` (default USDT).

        Sums free+locked balances of each asset and converts to USDT using latest prices.
        Stablecoins are treated as ~1:1 to reduce API calls.

        Returns:
            (equity_usdt, breakdown)
        """
        try:
            if not hasattr(self, "exchange"):
                raise RuntimeError("ccxt client not initialized: self.exchange is missing")

            # 1) Fetch balances (unified)
            bal = self.exchange.fetch_balance()
            totals = bal.get("total", {}) or {}

            # 2) Lightweight ticker cache to reduce API calls
            now = time.time()
            use_cache = (now - float(self._price_cache.get("ts", 0))) < float(price_cache_ttl_s)
            if use_cache and self._price_cache.get("tickers"):
                tickers = self._price_cache["tickers"]
            else:
                tickers = self.exchange.fetch_tickers()
                self._price_cache = {"ts": now, "tickers": tickers}

            equity = 0.0
            breakdown: List[Dict[str, Any]] = []

            # treat these as 1:1 to USDT
            stablecoins = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI"}

            for raw_asset, amount in totals.items():
                if not amount or amount <= 0:
                    continue

                asset = str(raw_asset).upper()
                value = None

                # Direct 1:1 or same-quote
                if asset == quote or asset in stablecoins:
                    value = float(amount)

                else:
                    # Typical ccxt symbol format
                    symbol = f"{asset}/{quote}"
                    price = None

                    # Try cached ticker first
                    t = tickers.get(symbol)
                    if t:
                        price = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")

                    # Fallback to single fetch
                    if price is None:
                        try:
                            t2 = self.exchange.fetch_ticker(symbol)
                            price = t2.get("last") or t2.get("close") or t2.get("bid") or t2.get("ask")
                        except Exception:
                            price = None

                    if price is not None:
                        value = float(amount) * float(price)

                if value is not None and value >= float(min_value_usd):
                    equity += value
                    breakdown.append(
                        {"asset": asset, "amount": float(amount), "value_usdt": float(value)}
                    )

            return float(equity), breakdown

        except Exception as e:
            print(f"[ServoTraderBinance] get_estimated_balance_usdt error: {e}")
            return 0.0, []