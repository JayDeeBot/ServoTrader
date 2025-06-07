# tests/test_servo_trader_binance.py

from servo_trader.servo_trader_binance import ServoTraderBinance
import os

# Replace with your actual keys or load them securely (e.g., from .env)
API_KEY = "6BZOFxkzIau3dqljZu8tbbKY5tZxnptRJkOfHq6Nx5jZDbvogxseqFkaQ3RnuaBE"
SECRET_KEY = "6vguZphjUuW9t6SdPZUxImWNzB2anPr91jAWHw9dwIASLeFAVnbQQMZi0iZVBdru"

def main():
    print("Initializing ServoTraderBinance...")
    trader = ServoTraderBinance(api_key=API_KEY, secret_key=SECRET_KEY)

    print("\n📦 Portfolio:")
    portfolio = trader.get_portfolio()
    for symbol, data in portfolio.items():
        print(f"{symbol}: Qty = {data['qty']}, Value = {data['value']}")

    print("\n💰 Cash Balance:")
    balance = trader.get_cash_balance()
    print(f"USDT: {balance:.4f}")

    # -------------------------
    # ⚠️ Optional Live Trade Tests
    # Only uncomment if you're ready to test real trades!
    # -------------------------

    # print("\n🛒 Testing Market Buy Order:")
    # buy_order_id = trader.execute_buy('BTCUSDT')  # Replace with any supported symbol
    # print(f"Buy Order ID: {buy_order_id}")

    # print("\n💸 Testing Market Sell Order:")
    # sell_order_id = trader.execute_sell('BTCUSDT', 0.0001)  # Replace with actual small qty you hold
    # print(f"Sell Order ID: {sell_order_id}")

    # -------------------------
    # Testing incomplete methods
    # -------------------------

    print("\n🛑 Testing get_order_by_id and cancel_order (placeholders):")
    trader.get_order_by_id("123456789")
    trader.cancel_order("123456789")

if __name__ == "__main__":
    main()
