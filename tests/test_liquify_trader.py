"""
Test Script: test_liquify_trader.py

This script tests the `liquify()` method of the ServoTraderBinance class.
It attempts to sell all non-USDT assets in the live Binance account,
and prints out the result of each liquidation attempt.

WARNING: This will place real market sell orders. Use on a testnet account or with caution.

Author: Jarred Deluca
Created: 2025
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from servo_trader.servo_trader_binance import ServoTraderBinance
import time

def test_liquify():
    print("🚀 Starting test for liquify() method...")

    trader = ServoTraderBinance("/home/jarred/git/ServoTrader/servo_trader/config/params.yaml")

    # Print initial portfolio
    print("\n📊 Initial Portfolio:")
    portfolio = trader.get_portfolio()
    for asset, info in portfolio.items():
        print(f"  {asset}: {info['qty']}")

    # Run the liquify method
    results = trader.liquify()

    # Print results
    print("\n🧾 Liquidation Results:")
    if results:
        for symbol, order_id in results.items():
            print(f"  ✔ Sold {symbol} (Order ID: {order_id})")
    else:
        print("  ❌ No assets were sold.")

    # Give time for orders to fill
    print("\n⏳ Waiting 10 seconds to confirm updated portfolio...")
    time.sleep(10)

    # Print updated portfolio
    print("\n📉 Updated Portfolio:")
    updated_portfolio = trader.get_portfolio()
    for asset, info in updated_portfolio.items():
        print(f"  {asset}: {info['qty']}")

    print("\n✅ Liquify test complete.\n")

if __name__ == "__main__":
    test_liquify()
