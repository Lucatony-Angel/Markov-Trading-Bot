import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime

def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start = start, end = end, auto_adjust = False)
    data.to_csv(("historical_prices.csv"))

    #print("File saved as historical_prices.csv")

    return data


def preprocess_data(data, window=20):
    """Preprocess data: calculate returns, define states."""
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Return'] = data['Close'].pct_change()
    data['RollingMean'] = data['Return'].rolling(window).mean()
    data['State'] = pd.cut(data['RollingMean'],
                           bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                           labels=[0, 1, 2, 3, 4]) 
    data.dropna(inplace=True)
    return data

def calculate_transition_matrix(states):
    """Calculate the Markov transition matrix."""
    n_states = len(states.unique())
    transition_matrix = np.zeros((n_states, n_states))

    for (s1, s2) in zip(states[:-1], states[1:]):
        transition_matrix[s1, s2] += 1

    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix

def structure_sequences(data, sequence_length=30):
    """Structure data as TensorFlow sequences."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data.iloc[i-sequence_length:i][['Return']].values)
        y.append(data.iloc[i]['State'])
    return np.array(X), np.array(y)


def build_model(sequence_length, feature_count, state_count=5):
    """Build a neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, feature_count)),
        tf.keras.layers.Dense(state_count, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_trading_signals_with_momentum(predictions, data, lookback=5):
    """Incorporate momentum into trading signals."""
    signals = []
    for i, state_probs in enumerate(predictions):
        most_likely_state = np.argmax(state_probs)
        recent_returns = data['Return'].iloc[max(0, i-lookback):i]
        momentum = recent_returns.sum()

        if most_likely_state in [0, 1]:  
            signals.append(-2 if momentum < -0.02 else -1)  
        elif most_likely_state == 2:  
            signals.append(0)  
        elif most_likely_state in [3, 4]:  
            signals.append(2 if momentum > 0.02 else 1)  
    return signals


def backtest(data, signals):
    """Backtest the trading strategy."""
    data['Signal'] = signals
    data['StrategyReturn'] = data['Signal'].shift(1) * data['Return']
    data['CumulativeStrategy'] = (1 + data['StrategyReturn']).cumprod()
    data['CumulativeBuyHold'] = (1 + data['Return']).cumprod()

    metrics = {
        "Cumulative Return": data['CumulativeStrategy'].iloc[-1],
        "Maximum Drawdown": (data['CumulativeStrategy'] / data['CumulativeStrategy'].cummax() - 1).min(),
        "Win/Loss Ratio": (data['StrategyReturn'] > 0).sum() / max((data['StrategyReturn'] < 0).sum(), 1)
    }
    return data, metrics

def save_run(test_data: pd.DataFrame, metrics: dict, output_dir: Path = Path("runs")) -> Path:
    output_dir.mkdir(parents=True, exist_ok= True)
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    data_path = output_dir / f"{run_id}_results.csv"
    metrics_path = output_dir / f"{run_id}_metrics.json"

    test_data.to_csv(data_path)
    pd.Series(metrics).to_json(metrics_path)

    return data_path


def run_backtest(ticker= "NVDA", start = "2020-01-01", end = "2024-12-31") -> tuple[pd.DataFrame, dict]: 
    csv_path = Path("historical_prices.csv")
    if csv_path.exists():
        data = pd.read_csv(csv_path, index_col = 0, parse_dates = True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                "_".join(col).strip() if isinstance(col, tuple) else col
                for col in data.columns
            ]
            
        if "Close" not in data.columns:
            raise KeyError(f"'Close' column missing; columns: {data.columns.tolist()}")
    else:
        data = download_prices(ticker, start, end)

    data = preprocess_data(data)

    transition_matrix = calculate_transition_matrix(data['State'].astype(int))

    sequence_length = 30
    X, y = structure_sequences(data, sequence_length)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_model(sequence_length, feature_count=X.shape[2])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32)

    predictions = model.predict(X_test)

    test_data = data.iloc[train_size + sequence_length:].copy() 
    signals = generate_trading_signals_with_momentum(predictions, test_data)

    test_data, performance_metrics = backtest(test_data, signals)
    #save_run(test_data, performance_metrics)

    return test_data, performance_metrics




if __name__ == "__main__":
    results, metrics = run_backtest()
    print("Backtest Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(results['CumulativeStrategy'], label="Strategy")
    plt.plot(results['CumulativeBuyHold'], label="Buy and Hold")
    plt.legend()
    plt.title("Backtest Results")
    plt.show()

    output_path = save_run(results, metrics)
    print(f"Saved latest run to {output_path}")