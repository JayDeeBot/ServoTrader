# ServoTrader
Automated Trading Algorithm

This repo consists of 5 classes that together make version 1 of the ServoTrader algorithm:

1. DatabaseInitialiser - Creates a csv of the initial database of stock close prices for a given set of stock codes, a given number of close prices, at a given interval.
2. ServoData - Retrieves the latest set of stock close prices for the given set of stock codes.
3. StockDataStore - Adds the latest data to the csv.
4. ServoPredictor - Repairs the data set, creates a series of regression model for each stock, filters the models by confidence level and predicts the growth.
5. ServoTrader - Trades to the stock with the strongest model (as long as at least one has a confidence level above the given threshold and is predicted to grow in the next loop iteration).

Furthermore, there is a main function that implements the members of each class in sequence and a series of unit tests to test the performance.

The program is designed to operate using the Alpaca API for stock trading. In theory it can be adapted to trade cryptos using the Alpaca API or to use other API's for either stock, cryptos or potentially currency.
