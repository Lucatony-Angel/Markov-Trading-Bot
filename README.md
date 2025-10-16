# Markov Trading Bot

Neural-network assisted trading simulator that pairs a simple Markov-state strategy with an LSTM and an interactive Streamlit dashboard. The app downloads historical prices, trains a model to classify market regimes, backtests the resulting signals, and stores every run so you can review current and past performance.

## Features
- LSTM-based market state classifier built with TensorFlow/Keras.
- Momentum-aware signal generation and backtest metrics (cumulative return, max drawdown, win/loss).
- Streamlit UI to trigger new runs, filter by date range, and compare strategy vs. buy-and-hold.
- Automatic persistence of each run (`runs/`) for later inspection.

## Project Structure
- `trading_bot_core.py` – core pipeline: data download, preprocessing, training, backtesting, and persistence helpers.
- `streamlit_app.py` – Streamlit dashboard that calls the core logic, displays charts, and exposes controls.
- `requirements.txt` – Python dependencies (numpy, pandas, tensorflow, scikit-learn, matplotlib, yfinance, streamlit, etc.).
- `runs/` – auto-generated CSV/JSON outputs per backtest (git-ignored).
- `historical_prices.csv` – cached price data from the most recent download (git-ignored).

## Getting Started

### Prerequisites
- Python 3.9 or newer recommended.
- Optional: virtualenv tooling (`python -m venv` or `pipenv`) to isolate dependencies.

### Install
```bash
git clone https://github.com/Lucatony-Angel/Markov-Trading-Bot.git
cd "Markov's trading bot"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Command-line Backtest
```bash
python trading_bot_core.py
```
This downloads price data (or reuses `historical_prices.csv`), trains the LSTM, prints backtest metrics, plots cumulative returns, and saves the run under `runs/`.

### Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Then open `http://localhost:8501` in a browser. Use the sidebar to:
1. Select ticker and date range.
2. Launch a new backtest (results auto-save and the view refreshes).
3. Pick any stored run and zoom into historical windows via the slider.

## Notes & Limitations
- Training 1000-epoch LSTMs can take several minutes; lower the epoch count for quick experiments.
- Apple Silicon users may need `tensorflow-macos` and `tensorflow-metal` if the standard wheel fails.
- External data comes from Yahoo Finance (`yfinance`); network access is required for fresh downloads.

## Next Steps
- Add automated tests around preprocessing and signal generation.
- Introduce configurable hyperparameters inside the Streamlit UI.
- Package as a deployable app or schedule periodic backtests.***
