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
