import streamlit as st
import pandas as pd
from pathlib import Path
from trading_bot_core import run_backtest, save_run

RUNS_DIR = Path("runs")


@st.cache_data
def load_runs(run_dir: Path = RUNS_DIR):
    results = []
    for csv_path in sorted(run_dir.glob("*_results.csv"), reverse = True):
        run_id = csv_path.stem.replace("_results", "")
        metrics_path = csv_path.with_name(f"{run_id}_metrics.json")
        metrics = {}
        if metrics_path.exists():
            metrics = pd.read_json(metrics_path, typ = "series").to_dict()
        df = pd.read_csv(csv_path, index_col = 0, parse_dates = True)
        results.append({"id": run_id, "data": df, "metrics": metrics})
    return results


st.sidebar.header("Run a New Backtest")
ticker = st.sidebar.text_input("Ticker", value = "NVDA")
start = st.sidebar.date_input("Start", value = pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End", value = pd.to_datetime("2024-12-31"))

if st.sidebar.button("Run backtest"):
    with st.sidebar:
        st.write("Running...")
    data, metrics = run_backtest(ticker, start.isoformat(), end.isoformat())
    save_run(data, metrics)
    st.cache_data.clear()
    st.rerun()

runs = load_runs()
if not runs:
    st.info("No backtests saved yet. Run one from the sidebar.")
    st.stop()
    
run_options = {
    f"{item['id']} ({item['data'].index.min().date()} → {item['data'].index.max().date()})": item
    for item in runs
}
label = st.selectbox("Select run", list(run_options.keys()))
selected = run_options[label]
df = selected["data"].copy()
metrics = selected["metrics"]

start_date, end_date = st.slider(
    "Date range",
    min_value = df.index.min().date(),
    max_value = df.index.max().date(),
    value = (df.index.min().date(), df.index.max().date()),
)
mask = (df.index.date >= start_date) & (df.index.date <= end_date)
filtered = df.loc[mask]

st.subheader("cumulative Returns")
st.line_chart(filtered[["CumulativeStrategy", "CumulativeBuyHold"]])
st.subheader("Key Metrics")

cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.2f}")
