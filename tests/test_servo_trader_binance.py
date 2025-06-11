# tests/test_servo_trader_binance.py
# /bin/python3.11 -m tests.test_servo_trader_binance

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from servo_trader.servo_trader_binance import ServoTraderBinance
import time

# Replace with your actual keys or load them securely (e.g., from .env)
API_KEY = "6BZOFxkzIau3dqljZu8tbbKY5tZxnptRJkOfHq6Nx5jZDbvogxseqFkaQ3RnuaBE"
SECRET_KEY = "6vguZphjUuW9t6SdPZUxImWNzB2anPr91jAWHw9dwIASLeFAVnbQQMZi0iZVBdru"

def main():
    print("Initializing ServoTraderBinance...")
    trader = ServoTraderBinance(api_key=API_KEY, secret_key=SECRET_KEY)

    ### --- PORTFOLIO/BALANCE RETRIEVAL TESTING --- ###

    # Check your portfolio
    print("\n📦 Portfolio:")
    portfolio = trader.get_portfolio()
    for symbol, data in portfolio.items():
        print(f"{symbol}: Qty = {data['qty']}, Value = {data['value']}")

    # Check your cash balance
    print("\n💰 Cash Balance:")
    balance = trader.get_cash_balance()
    print(f"USDT: {balance:.4f}")

    # -------------------------
    # ⚠️ Optional Live Trade Tests
    # Only uncomment if you're ready to test real trades!
    # -------------------------

    # ### --- BUY ORDER TESTING --- ###

    # # Make a buy order
    # print("\n🛒 Testing Market Buy Order:")
    # buy_order_id = trader.execute_buy('BTCUSDT')  # Buy a crypto and save the ID
    # print(f"Buy Order ID: {buy_order_id}")
    
    # # Check the status of your buy order
    # print("\n🛑 Testing Buy Order Status Retrieval:")
    # trader.get_order_by_id(buy_order_id) # Check the buy order status

    # ### --- SELL ORDER TESTING --- ###

    # Make a sell order
    print("\n💸 Testing Market Sell Order:")
    saleable_qty = trader.get_saleable_quantity("BTCUSDT") # Caculate the highest qty we can sell
    print(f"Saleable BTC: {saleable_qty}")
    sell_order_id = trader.execute_sell("BTCUSDT", saleable_qty) # Sell the crypto and save the ID
    print(f"Sell Order ID: {sell_order_id}")

    # Check the status of your sell order
    print("\n🛑 Testing Sell Order Status Retrieval:")
    trader.get_order_by_id(sell_order_id) # Check the sell order status

    ### -- CANCEL ORDER TESTING --- ###

    # # Place an LIMIT BUY order (below market)
    # print("\n🧪 Placing realistic LIMIT BUY order:")
    # try:
    #     ticker = trader.client.get_symbol_ticker(symbol="BTCUSDT")
    #     market_price = float(ticker["price"])
    #     price = round(market_price * 0.9, 2)  # 10% below market, safe for buy

    #     buy_order = trader.client.create_order(
    #         symbol='BTCUSDT',
    #         side='BUY',
    #         type='LIMIT',
    #         timeInForce='GTC',
    #         quantity=0.0001,
    #         price=str(price)
    #     )
    #     buy_order_id = str(buy_order['orderId'])
    #     trader.order_symbol_map[buy_order_id] = 'BTCUSDT'
    #     print(f"Limit Buy Order ID: {buy_order_id} at ${price}")
    # except Exception as e:
    #     print(f"Error placing limit buy: {e}")

    # # Cancel the BUY order
    # if buy_order_id:
    #     print("\n🛑 Canceling Limit Buy Order:")
    #     trader.cancel_order(buy_order_id)

    # # Place an LIMIT SELL order (above market)
    # print("\n🧪 Placing realistic LIMIT SELL order:")
    # try:
    #     qty = trader.get_saleable_quantity('BTCUSDT')
    #     ticker = trader.client.get_symbol_ticker(symbol="BTCUSDT")
    #     market_price = float(ticker["price"])
    #     price = round(market_price * 1.1, 2)  # 10% above market, safe for sell

    #     if qty >= 0.0001:
    #         sell_order = trader.client.create_order(
    #             symbol='BTCUSDT',
    #             side='SELL',
    #             type='LIMIT',
    #             timeInForce='GTC',
    #             quantity=qty,
    #             price=str(price)
    #         )
    #         sell_order_id = str(sell_order['orderId'])
    #         trader.order_symbol_map[sell_order_id] = 'BTCUSDT'
    #         print(f"Limit Sell Order ID: {sell_order_id} at ${price}")
    #     else:
    #         print("Not enough BTC to place test sell order.")
    # except Exception as e:
    #     print(f"Error placing limit sell: {e}")

    # # Wait briefly (optional)
    # time.sleep(1)

    # # Cancel the SELL order
    # if sell_order_id:
    #     print("\n🛑 Canceling Limit Sell Order:")
    #     trader.cancel_order(sell_order_id)

if __name__ == "__main__":
    main()
