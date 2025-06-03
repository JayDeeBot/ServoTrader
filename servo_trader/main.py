import time
import yaml
from crypto_database_init import CryptoDatabaseInitialiser
from servo_trader import ServoTrader
from servo_predictor import ServoPredictor
from concurrent.futures import ProcessPoolExecutor, as_completed

API_KEY = 'AKG8S8UZN25QTO2PQS6X'
SECRET_KEY = 'MllrezSqh5HWY8qmfhxcAWBqOBvxIU2OzZtIuyt8'

RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

loop_interval_minutes = params['loop_interval_minutes']
confidence_threshold = params['confidence_threshold'] / 100
timeframe = params['timeframe']
crypto_bars_to_analyse = params['crypto_bars_to_analyse']

print("Initialising Database...")
data_base = CryptoDatabaseInitialiser(
    csv_path='crypto_data.csv',
    json_path='crypto_codes.json',
    yaml_path='params.yaml'
)

predictor = ServoPredictor(csv_path='crypto_data.csv', json_path='crypto_codes.json', threshold=confidence_threshold, timeframe=timeframe)
servo_trader = None  # Delay initialization until a prediction is available

# State machine states
STATE_LOCATE_CRYPTO = "locatecrypto"
STATE_CHECK_BUY = "checkbuy"
STATE_SELL_CRYPTO = "sellcrypto"
STATE_CHECK_SELL = "checksell"

state = STATE_LOCATE_CRYPTO
current_crypto = None
last_sell_trade_id = None
last_buy_trade_id = None
last_buy_price = None

# Track sell order attempts
sell_order_attempts = 0
MAX_SELL_ATTEMPTS = 1

while True:
    loop_start = time.time()
    print("State: ", state)
    try:
        # print("Retrieving Latest Crypto Data...")
        # data_base = CryptoDatabaseInitialiser(
        #     csv_path='crypto_data.csv',
        #     json_path='crypto_codes.json',
        #     yaml_path='params.yaml'
        # )

        # print("Fitting models with multiprocessing...")
        # start_time = time.time()

        # # Temporary storage for models
        # model_results = []

        # with ProcessPoolExecutor(max_workers=8) as executor:
        #     futures = [executor.submit(predictor.fit_models, crypto_code) for crypto_code in predictor.crypto_codes]
            
        #     for future in as_completed(futures):
        #         try:
        #             crypto_code, model_data = future.result()
        #             if model_data:
        #                 model_results.append((crypto_code, model_data))  # Collect results safely
        #         except Exception as e:
        #             print(f"{RED_COLOR}Error fitting model in process: {e}{RESET_COLOR}")

        # # Now safely update the predictor.models dictionary after all processes are done
        # for crypto_code, model_data in model_results:
        #     predictor.models[crypto_code] = model_data

        # elapsed_time = time.time() - start_time
        # print(f"Model fitting completed in {elapsed_time:.2f} seconds.")
        # predictor.filter_models()

        if state == STATE_LOCATE_CRYPTO:
            print("Retrieving Latest Crypto Data...")
            data_base = CryptoDatabaseInitialiser(
                csv_path='crypto_data.csv',
                json_path='crypto_codes.json',
                yaml_path='params.yaml'
            )

            print("Fitting models with multiprocessing...")
            # start_time = time.time()

            # Temporary storage for models
            model_results = []

            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(predictor.fit_models, crypto_code) for crypto_code in predictor.crypto_codes]
                
                for future in as_completed(futures):
                    try:
                        crypto_code, model_data = future.result()
                        if model_data:
                            model_results.append((crypto_code, model_data))  # Collect results safely
                    except Exception as e:
                        print(f"{RED_COLOR}Error fitting model in process: {e}{RESET_COLOR}")

            # Now safely update the predictor.models dictionary after all processes are done
            for crypto_code, model_data in model_results:
                predictor.models[crypto_code] = model_data

            # elapsed_time = time.time() - start_time
            # print(f"Model fitting completed in {elapsed_time:.2f} seconds.")
            predictor.filter_models()
            best_crypto, expected_growth = predictor.predict_best_crypto(minutes_ahead=loop_interval_minutes)
            if best_crypto:
                servo_trader = ServoTrader(
                    api_key=API_KEY,
                    secret_key=SECRET_KEY,
                    prediction=(best_crypto, expected_growth),
                    paper=False
                )
                account = servo_trader.trading_client.get_account()
                # print(f"Crypto Status: {account.crypto_status}")
                last_buy_trade_id = servo_trader.execute_buy(best_crypto)
                if last_buy_trade_id:
                    current_crypto = best_crypto
                    state = STATE_CHECK_BUY
                else:
                    print(f"{RED_COLOR}Failed to execute buy trade for {best_crypto}.{RESET_COLOR}")
            else:
                print(f"{RED_COLOR}No valid crypto predictions available. Skipping trading this loop.{RESET_COLOR}")

        elif state == STATE_CHECK_BUY:
            if last_buy_trade_id:
                try:
                    buy_order = servo_trader.trading_client.get_order_by_id(last_buy_trade_id)
                    if buy_order.status == 'filled':
                        last_buy_price = float(buy_order.filled_avg_price)
                        print(f"Successfully bought crypto: {current_crypto}")
                        state = STATE_SELL_CRYPTO
                    else:
                        print(f"Waiting for buy order to be filled for {current_crypto}...")
                except Exception as e:
                    print(f"{RED_COLOR}Error fetching buy order: {e}{RESET_COLOR}")
                    state = STATE_LOCATE_CRYPTO  # Reset to avoid getting stuck

        elif state == STATE_SELL_CRYPTO:
            if last_buy_price:  # Ensure the last buy price is available
                servo_trader.portfolio = servo_trader.get_portfolio()  # Refresh portfolio
                limit_price = predictor.calculate_dynamic_limit_price(current_crypto, last_buy_price)
                fractional_qty = servo_trader.portfolio.get(current_crypto, {}).get("qty", 0)

                if fractional_qty > 0:  # Check for valid quantity
                    print(f"Selling crypto: {current_crypto} at dynamic limit price: ${limit_price:.2f}.")
                    last_sell_trade_id = servo_trader.execute_sell(
                        current_crypto, fractional_qty=fractional_qty, limit_price=limit_price
                    )
                    state = STATE_CHECK_SELL
                else:
                    print(f"{RED_COLOR}Invalid quantity. Cannot sell crypto: {current_crypto}.{RESET_COLOR}")
                    state = STATE_LOCATE_CRYPTO
            else:
                print(f"{RED_COLOR}No buy price available for {current_crypto}.{RESET_COLOR}")
                state = STATE_LOCATE_CRYPTO

        elif state == STATE_CHECK_SELL:
            if last_sell_trade_id:
                sell_order = servo_trader.trading_client.get_order_by_id(last_sell_trade_id)
                if sell_order.status == 'filled':
                    print(f"Successfully sold crypto: {current_crypto}")
                    current_crypto = None
                    servo_trader = None
                    sell_order_attempts = 0  # Reset attempts
                    state = STATE_LOCATE_CRYPTO
                else:
                    print(f"Waiting for sell order to be filled for {current_crypto}... Attempt {sell_order_attempts + 1}")
                    sell_order_attempts += 1

                    if sell_order_attempts >= MAX_SELL_ATTEMPTS:
                        print(f"{RED_COLOR}Sell order for {current_crypto} stuck. Canceling and re-placing...{RESET_COLOR}")
                        servo_trader.trading_client.cancel_order_by_id(last_sell_trade_id)

                        # # Re-calculate limit price before re-placing
                        # servo_trader.portfolio = servo_trader.get_portfolio()  # Refresh portfolio before re-placing order
                        # limit_price = predictor.calculate_dynamic_limit_price(current_crypto, last_buy_price)
                        # fractional_qty = servo_trader.portfolio.get(current_crypto, {}).get("qty", 0)

                        # last_sell_trade_id = servo_trader.execute_sell(current_crypto, fractional_qty, limit_price)
                        # sell_order_attempts = 0  # Reset attempts after re-placing

                        # Execute a market sell instead
                        servo_trader.execute_market_sell(current_crypto)

                        # Reset after forcing liquidation
                        current_crypto = None
                        servo_trader = None
                        sell_order_attempts = 0
                        state = STATE_LOCATE_CRYPTO

    except Exception as e:
        print(f"{RED_COLOR}An error occurred in the loop: {e}{RESET_COLOR}")

    loop_duration = time.time() - loop_start
    sleep_time = max(0, (loop_interval_minutes * 60) - loop_duration)
    print(f"Loop completed in {loop_duration:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.")

    for remaining in range(int(sleep_time), 0, -1):
        print(f"Sleeping... {remaining} seconds remaining", end="\r")
        time.sleep(1)
    print("Sleeping... 0 seconds remaining. Starting next loop.")