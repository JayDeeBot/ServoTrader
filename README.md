# ServoTrader

**ServoTrader** is an automated cryptocurrency trading bot that uses machine learning models (e.g., regression ensembles) to predict short-term price movements and execute limit trades for fast, small-profit selloffs. It leverages technical indicators and model accuracy filtering to make trading decisions dynamically based on the current market.

---

## 🚀 Features

- 📈 Predicts the best-performing crypto using over 20 regression models
- 🧠 Applies technical indicators (MACD, Bollinger Bands, ATR, etc.)
- 🤖 Automates buy/sell orders through Alpaca or Binance API
- 🕰️ Fully autonomous state machine for trading logic
- 📊 Dynamic limit pricing based on recent volatility
- 🔁 Loop-based design with continuous crypto selection and trade execution

---

## 📂 Project Structure

```
servo_trader/
├── crypto_database_init.py       # Downloads historical OHLCV data using Kraken API
├── servo_predictor.py            # Technical indicators + model training & prediction
├── servo_trader.py               # Trade execution logic via Alpaca
├── main.py                       # Full state machine logic (buy, hold, sell)
├── crypto_codes.json             # List of available crypto symbols
├── params.yaml                   # Configuration file for timeframes and confidence
├── LICENSE
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## 🔧 Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/servo_trader.git
cd servo_trader
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or with Poetry:

```bash
poetry install
```

---

## 🧪 Usage

1. Configure the `params.yaml` file with your desired loop interval, data size, and confidence threshold.

2. Add your API keys to `main.py` (or use environment variables for security):

```python
API_KEY = "your_binance_key"
SECRET_KEY = "your_binance_secret"
```

3. Run the bot:

```bash
python main.py
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

Created by [Jarred Deluca](mailto:jarred.g.deluca@student.uts.edu.au)

## TODO List

1. Test LSTM in small training sessions and evaluate the models.
2. Adjust the price selection in the crypto training environment to make it more stringent:
    (a) Buy Price = High @ step
    (b) Hold price = Low @ step
    (c) Sell price = Low @ step - epsilon (where e = some small value)
3. Repair the live trading env and scripts so that the logging is accurate, areas to repair:
    (a) The sell price
    (b) Episode return
    (c) Overall return
    (d) Average daily return
4. Review the buy and sell/hold training environments;
    (a) Buy:
        (i) The highest reward should be given to the buy decision that within the next 60 minutes will achieve the most profit
    (b) Sell/Hold:
        (i) Hold - Rewards should be positive if the crypto increases and negative if it decreases.
        (ii) Sell - Inverse of hold.
5. Download additional datasets for different cryptos - 2 more data sets of 100 new cryptos each. Attempt further training on the bulbasaur model and evaluate to determine whether the model has improved performance. 
