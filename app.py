"""
╔══════════════════════════════════════════════════════════════════╗
║     AI-Based Stock Price Prediction System Using LSTM            ║
║     Built with Python, TensorFlow, and Streamlit                 ║
╚══════════════════════════════════════════════════════════════════╝
"""
import pandas as pd

def safe_float(value):
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    try:
        return float(value)
    except:
        return None
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
EarlyStopping = tf.keras.callbacks.EarlyStopping
import warnings
import math
import datetime

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

def safe_scalar(value):
    import numpy as np
    
    if value is None:
        return None
    
    if isinstance(value, (list, tuple, np.ndarray)):
        value = np.array(value).flatten()[0]
    
    if hasattr(value, "iloc"):
        value = value.iloc[-1]
    
    return float(value)

def safe_number(value):
    try:
        if value is None:
            return None
        if hasattr(value, "iloc"):
            value = value.iloc[-1]
        return float(value)
    except:
        return None

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# ─────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────

st.markdown("""
<style>

/* Sidebar Background */
section[data-testid="stSidebar"] {
    background: #071124;
    border-right: 1px solid rgba(0,255,255,0.2);
}

/* Remove Streamlit default padding */
.block-container {
    padding-top: 1rem;
}

/* Page background */
.stApp {
    background: linear-gradient(135deg, #0a0f1f, #111a35);
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CUSTOM CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* Full dark grid background */
.stApp {
    background-color: #050B18;
    background-image:
        linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
    background-size: 40px 40px;
}
    /* Title area */
    .main-title {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        animation: shine 3s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }

    /* Metric cards */
    .metric-card {
    background: rgba(10,20,40,0.6);
    border: 1px solid rgba(0,255,255,0.3);
    border-radius: 16px;
    padding: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 20px rgba(0,255,255,0.1);
    transition: 0.3s ease;
}

   .metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 30px rgba(0,255,255,0.4);
}

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }

    /* Signal box */
    .signal-buy {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        box-shadow: 0 0 30px rgba(56,239,125,0.4);
    }
    .signal-sell {
        background: linear-gradient(135deg, #fc4a1a, #f7b733);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        box-shadow: 0 0 30px rgba(252,74,26,0.4);
    }
    .signal-hold {
        background: linear-gradient(135deg, #485563, #29323c);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
    }

    /* Info box */
    .info-box {
        background: rgba(0,210,255,0.08);
        border-left: 4px solid #00d2ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }

    /* Sidebar */
    .css-1d391kg { background: rgba(15,12,41,0.95) !important; }

    /* Prediction table */
    .pred-row-up { color: #38ef7d; font-weight: 600; }
    .pred-row-dn { color: #fc4a1a; font-weight: 600; }

    /* Section header */
    .section-header {
        border-bottom: 2px solid #00d2ff;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        color: #00d2ff;
        font-size: 1.3rem;
        font-weight: 700;
    }
            
    .stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(0,255,255,0.08);
    border-radius: 12px;
    padding: 10px 20px;
    border: 1px solid rgba(0,255,255,0.2);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #00f5ff, #00c3ff);
    color: black !important;
    font-weight: 700;
}
            .stButton>button {
    background: linear-gradient(90deg,#00f5ff,#00c3ff);
    color: black;
    font-weight: 700;
    border-radius: 12px;
    border: none;
}

.stButton>button:hover {
    box-shadow: 0 0 20px #00f5ff;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    ticker = st.text_input(
        " Enter Stock Ticker",
        value="AAPL",
        help="Examples: AAPL, TSLA, GOOGL, RELIANCE.NS, TCS.NS"
    ).upper().strip()
    

    period = st.selectbox(
    "Historical Data Period",
    options=["1y", "2y"],
    index=0,
)
    st.markdown("---")
    st.markdown("### Model Settings")

    epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=5, step=1)

    batch_size = st.selectbox("Batch Size", [16, 32], index=1)

    seq_len = st.slider("Sequence Length (days)", min_value=10, max_value=60, value=30, step=5)
    st.markdown("---")
    run_button = st.button(" Run Prediction", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b> Popular Tickers</b><br>
    🇺🇸 AAPL · TSLA · GOOGL · MSFT · AMZN<br>
    🇮🇳 RELIANCE.NS · TCS.NS · INFY.NS
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <small style='color:#888'>
    ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. 
    Do not use as financial advice.
    </small>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 AI Stock Price Prediction System</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:1rem; margin-bottom:1rem;'>
    Powered by <b style='color:#00d2ff'>LSTM Deep Learning</b> · 
    Built with <b style='color:#00d2ff'>TensorFlow & Streamlit</b> · 
    Real-time data via <b style='color:#00d2ff'>Yahoo Finance</b>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, period: str):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")

        df.index = pd.to_datetime(df.index)

        stock = yf.Ticker(ticker)
        info = stock.info

        return df, info

    except Exception as e:
        raise ValueError(f"Failed to fetch data: {e}")

def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 50-day and 200-day Simple Moving Averages."""
    df = df.copy()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    return df


def prepare_lstm_data(series: np.ndarray, seq_len: int, split_ratio: float = 0.8):
    """
    Normalize data and create overlapping sequences for LSTM.
    Returns: X_train, y_train, X_test, y_test, scaler, train_size
    """
    # ── Normalize to [0, 1] ──
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    # ── Create sequences of shape (samples, seq_len, 1) ──
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # ── 80/20 train-test split ──
    train_size = int(len(X) * split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # ── Reshape for LSTM: (samples, time_steps, features) ──
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    return X_train, y_train, X_test, y_test, scaler, train_size


def build_lstm_model(seq_len: int) -> tf.keras.Sequential:
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(seq_len, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    """
    Architecture:
      LSTM(128) → Dropout(0.2)
      LSTM(64)  → Dropout(0.2)
      Dense(32) → Dense(1)
    """
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(seq_len, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_next_7_days(model, last_seq: np.ndarray, scaler, seq_len: int) -> np.ndarray:
    """
    Iteratively predict next 7 trading days using the trained LSTM.
    Each prediction is appended to the sequence for the next step.
    """
    predictions = []
    current_seq = last_seq.copy()  # shape (seq_len, 1)

    for _ in range(7):
        inp = current_seq.reshape(1, seq_len, 1)
        pred_scaled = model.predict(inp, verbose=0)[0][0]
        pred_price  = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_price)
        # Slide window forward
        current_seq = np.append(current_seq[1:], [[pred_scaled]], axis=0)

    return np.array(predictions)


def generate_signal(prices_7d: np.ndarray, current_price: float) -> tuple:
    """
    Simple rule-based Buy/Sell/Hold signal:
      - If avg predicted price > current * 1.02  → BUY
      - If avg predicted price < current * 0.98  → SELL
      - Otherwise                                → HOLD
    """
    avg_pred = np.mean(prices_7d)
    change_pct = ((avg_pred - current_price) / current_price) * 100

    if avg_pred > current_price * 1.02:
        return "🟢 BUY", "signal-buy", change_pct
    elif avg_pred < current_price * 0.98:
        return "🔴 SELL", "signal-sell", change_pct
    else:
        return "🟡 HOLD", "signal-hold", change_pct


def plot_price_history(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """Plot closing price with 50-day and 200-day moving averages."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Close"],  color="#00d2ff", linewidth=1.5, label="Close Price", alpha=0.9)
    ax.plot(df.index, df["MA50"],   color="#f7b733", linewidth=1.5, label="MA 50",       linestyle="--")
    ax.plot(df.index, df["MA200"],  color="#fc4a1a", linewidth=1.5, label="MA 200",      linestyle="--")
    close_values = df["Close"].values.flatten()

    ax.plot(df.index, close_values, color="#00d2ff", linewidth=1.5, label="Close Price", alpha=0.9)
    ax.fill_between(df.index, close_values, alpha=0.1, color="#00d2ff")
    ax.set_title(f"{ticker} – Historical Closing Price with Moving Averages", fontsize=14, color="white", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(facecolor="#1a1a2e", edgecolor="#00d2ff", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(df: pd.DataFrame, train_size: int, seq_len: int,
                              predictions: np.ndarray, ticker: str) -> plt.Figure:
    """Plot actual vs predicted stock prices on the test set."""
    # Align prediction dates
    pred_dates = df.index[train_size + seq_len: train_size + seq_len + len(predictions)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Close"], color="#aaaaaa", linewidth=1, label="All Actual", alpha=0.5)
    ax.plot(pred_dates, df["Close"].values[train_size + seq_len: train_size + seq_len + len(predictions)],
            color="#00d2ff", linewidth=2, label="Actual (Test Period)")
    ax.plot(pred_dates, predictions, color="#38ef7d", linewidth=2, label="LSTM Predicted", linestyle="--")
    ax.axvline(x=df.index[train_size + seq_len], color="#f7b733",
               linestyle=":", linewidth=2, label="Train/Test Split")
    ax.set_title(f"{ticker} – Actual vs LSTM Predicted Price", fontsize=14, color="white", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(facecolor="#1a1a2e", edgecolor="#00d2ff", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def plot_future_7_days(future_prices: np.ndarray, last_date, current_price: float, ticker: str) -> plt.Figure:
    """Plot 7-day future price predictions."""
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)
    colors = ["#38ef7d" if p >= current_price else "#fc4a1a" for p in future_prices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(y=current_price, color="#aaaaaa", linestyle="--", linewidth=1.5, label=f"Current: ${current_price:.2f}", alpha=0.7)
    ax.plot([last_date] + list(future_dates), [current_price] + list(future_prices),
            color="#00d2ff", linewidth=2, marker='o', markersize=8, zorder=5)
    for i, (d, p, c) in enumerate(zip(future_dates, future_prices, colors)):
        ax.scatter(d, p, color=c, s=120, zorder=6)
        ax.annotate(f"${p:.2f}", (d, p), textcoords="offset points",
                    xytext=(0, 12), ha='center', color=c, fontsize=10, fontweight='bold')
    ax.set_title(f"{ticker} – 7-Day Price Forecast", fontsize=14, color="white", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Predicted Price (USD)", fontsize=11)
    ax.legend(facecolor="#1a1a2e", edgecolor="#00d2ff", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.tight_layout()
    return fig


def plot_loss_curve(history) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'],     color="#00d2ff", linewidth=2, label="Training Loss")
    ax.plot(history.history['val_loss'], color="#fc4a1a", linewidth=2, label="Validation Loss")
    ax.set_title("Model Training Loss (MSE)", fontsize=13, color="white", pad=15)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(facecolor="#1a1a2e", edgecolor="#00d2ff", labelcolor="white")
    fig.tight_layout()
    return fig


def plot_volume(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """Plot trading volume as a bar chart."""
    fig, ax = plt.subplots(figsize=(14, 3))

    close_vals  = df["Close"].values.flatten()
    open_vals   = df["Open"].values.flatten()
    volume_vals = df["Volume"].values.flatten()

    colors = ["#38ef7d" if c >= o else "#fc4a1a"
              for c, o in zip(close_vals, open_vals)]

    ax.bar(df.index, volume_vals, color=colors, alpha=0.7, width=1)

    ax.set_title(f"{ticker} – Trading Volume", fontsize=13, color="white", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Volume", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  WELCOME STATE (before run)
# ─────────────────────────────────────────────
if not run_button:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size:2.5rem'></div>
            <div style='color:#00d2ff; font-size:1.1rem; font-weight:700; margin-top:0.5rem'>LSTM Neural Network</div>
            <div style='color:#aaa; font-size:0.85rem; margin-top:0.3rem'>2 stacked LSTM layers with Dropout regularization</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size:2.5rem'></div>
            <div style='color:#00d2ff; font-size:1.1rem; font-weight:700; margin-top:0.5rem'>Real-Time Data</div>
            <div style='color:#aaa; font-size:0.85rem; margin-top:0.3rem'>Live historical data from Yahoo Finance API</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size:2.5rem'></div>
            <div style='color:#00d2ff; font-size:1.1rem; font-weight:700; margin-top:0.5rem'>7-Day Forecast</div>
            <div style='color:#aaa; font-size:0.85rem; margin-top:0.3rem'>Next week price prediction with Buy/Sell signal</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    <b> How to Use:</b><br>
    1. Enter a stock ticker in the sidebar (e.g., <b>AAPL</b> for Apple, <b>TSLA</b> for Tesla, <b>RELIANCE.NS</b> for Reliance Industries)<br>
    2. Select how much historical data to train on<br>
    3. Adjust model hyperparameters if desired<br>
    4. Click <b> Run Prediction</b> to train the LSTM and generate forecasts
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ###  Model Architecture
    ```
    Input → LSTM(128, return_sequences=True) → Dropout(0.2)
          → LSTM(64, return_sequences=False)  → Dropout(0.2)
          → Dense(32, activation='relu')
          → Dense(1)  [Output: predicted price]
    
    Optimizer : Adam
    Loss      : Mean Squared Error (MSE)
    Sequences : 60-day sliding window
    Split     : 80% Train / 20% Test
    ```
    """)


# ─────────────────────────────────────────────
#  MAIN PREDICTION PIPELINE
# ─────────────────────────────────────────────
if run_button:
    try:
        # ── Step 1: Fetch Data ──────────────────────────
        with st.spinner(f"📡 Fetching data for **{ticker}** from Yahoo Finance..."):
            df, info = fetch_stock_data(ticker, period)
            df = compute_moving_averages(df)

        # ── Company Info Header ─────────────────────────
        company_name = info.get("longName", ticker)
        sector       = info.get("sector", "N/A")
        industry     = info.get("industry", "N/A")
        currency     = info.get("currency", "USD")

        market_cap = safe_number(info.get("marketCap"))

        st.markdown(f"""
        <div style='background:rgba(0,210,255,0.08); border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem;'>
            <h2 style='color:#00d2ff; margin:0'>{company_name} <span style='color:#aaa; font-size:1rem'>({ticker})</span></h2>
            <span style='color:#888; font-size:0.9rem'> {sector} · {industry} ·  Currency: {currency}</span>
            {'<span style="color:#888; font-size:0.9rem"> · Market Cap: ${:,.0f}</span>'.format(market_cap) if market_cap else ''}
        </div>
        """, unsafe_allow_html=True)
        

        # ── Key Stats Row ───────────────────────────────
        current_price = safe_number(df["Close"].iloc[-1])
        previous_price = safe_number(df["Close"].iloc[-2])

        price_change = None
        price_pct = None

        if current_price is not None and previous_price is not None:
           price_change = current_price - previous_price
           price_pct = (price_change / previous_price) * 100

        high_52w = safe_number(df["Close"].tail(252).max())
        low_52w  = safe_number(df["Close"].tail(252).min())
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("💵 Current Price", f"${current_price:.2f}",
                      f"{price_change:+.2f} ({price_pct:+.2f}%)")
        with c2:
            st.metric("📈 52W High", f"${high_52w:.2f}")
        with c3:
            st.metric("📉 52W Low",  f"${low_52w:.2f}")
        with c4:
            st.metric("📅 Data Points", f"{len(df):,}")
        with c5:
            st.metric("📆 Period", f"{df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')}")

        st.markdown("---")

        # ── TAB LAYOUT ──────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Price Analysis",
            "🤖 LSTM Prediction",
            "🔮 7-Day Forecast",
            "📋 Raw Data"
        ])
        

        # ══════════════════════════════════════════
        #  TAB 1: PRICE ANALYSIS
        # ══════════════════════════════════════════
        with tab1:
            st.markdown('<div class="section-header">📊 Stock Price History & Moving Averages</div>', unsafe_allow_html=True)
            st.pyplot(plot_price_history(df, ticker), use_container_width=True)

            st.markdown('<div class="section-header">📦 Trading Volume</div>', unsafe_allow_html=True)
            st.pyplot(plot_volume(df, ticker), use_container_width=True)

            # Moving Average Analysis
            st.markdown('<div class="section-header">🔄 Moving Average Summary</div>', unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            ma50_val  = df["MA50"].iloc[-1]
            ma200_val = df["MA200"].iloc[-1]

            with col_a:
                st.metric("📊 50-Day MA",  f"${ma50_val:.2f}",
                          f"{'Above' if current_price > ma50_val else 'Below'} MA50")
            with col_b:
                st.metric("📊 200-Day MA", f"${ma200_val:.2f}",
                          f"{'Above' if current_price > ma200_val else 'Below'} MA200")
            with col_c:
                golden = ma50_val > ma200_val
                st.metric("🌟 MA Signal",
                          "Golden Cross ✨" if golden else "Death Cross ⚠️",
                          "Bullish" if golden else "Bearish")

            # Rolling statistics
            st.markdown('<div class="section-header">📐 Statistical Summary (Last 252 Trading Days)</div>', unsafe_allow_html=True)
            
            # Force 1D array
            recent_vals = df["Close"].tail(252).values.flatten()

            mean_val = float(np.mean(recent_vals))
            std_val  = float(np.std(recent_vals))
            skew_val = float(pd.Series(recent_vals).skew())
            kurt_val = float(pd.Series(recent_vals).kurt())
            min_val  = float(np.min(recent_vals))
            max_val  = float(np.max(recent_vals))

            stats_df = pd.DataFrame({
               "Metric": [
                  "Mean Price",
                  "Std Deviation",
                  "Skewness",
                  "Kurtosis",
                  "Min Price",
                  "Max Price"
            ],
                "Value": [
                   f"${mean_val:.2f}",
                   f"${std_val:.2f}",
                   f"{skew_val:.4f}",
                   f"{kurt_val:.4f}",
                   f"${min_val:.2f}",
                   f"${max_val:.2f}"
            ]
       })

        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        # ══════════════════════════════════════════
        #  TAB 2: LSTM TRAINING & PREDICTION
        # ══════════════════════════════════════════
        with tab2:
            st.markdown('<div class="section-header">🤖 Training LSTM Model</div>', unsafe_allow_html=True)

            # Prepare data
            close_prices = df["Close"].values
            X_train, y_train, X_test, y_test, scaler, train_size = prepare_lstm_data(
                close_prices, seq_len=seq_len
            )

            st.markdown(f"""
            <div class='info-box'>
            📐 <b>Data Split:</b>
            Training samples: <b>{len(X_train):,}</b> &nbsp;|&nbsp;
            Testing samples: <b>{len(X_test):,}</b> &nbsp;|&nbsp;
            Sequence length: <b>{seq_len} days</b>
            </div>
            """, unsafe_allow_html=True)

            # Build and train
            model = build_lstm_model(seq_len)

            progress_bar = st.progress(0, text=" Initializing model...")
            status_text  = st.empty()

            # Custom callback to update Streamlit progress
            class StreamlitCallback:
                def __init__(self, total_epochs):
                    self.total_epochs = total_epochs

                def on_epoch_end(self, epoch, logs=None):
                    pct = int((epoch + 1) / self.total_epochs * 100)
                    loss     = logs.get('loss', 0)
                    val_loss = logs.get('val_loss', 0)
                    progress_bar.progress(pct, text=f"Epoch {epoch+1}/{self.total_epochs} | Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")

            cb = StreamlitCallback(epochs)

            # Keras callback wrapper
            import tensorflow as tf 
            Callback = tf.keras.callbacks.Callback
            class ProgressCallback(Callback):
                def __init__(self, sl_cb):
                    self.sl_cb = sl_cb
                def on_epoch_end(self, epoch, logs=None):
                    self.sl_cb.on_epoch_end(epoch, logs)

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )

            progress_bar.progress(100, text="✅ Training complete!")
            status_text.success(f"Model trained for {len(history.history['loss'])} epochs.")

            # Evaluate
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
            y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
            mae  = mean_absolute_error(y_actual, y_pred)
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

            # Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>${rmse:.2f}</div>
                    <div class='metric-label'>RMSE (Root Mean Square Error)</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>${mae:.2f}</div>
                    <div class='metric-label'>MAE (Mean Absolute Error)</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{mape:.2f}%</div>
                    <div class='metric-label'>MAPE (Mean Absolute Percentage Error)</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Plots
            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                st.markdown('<div class="section-header">📈 Actual vs Predicted</div>', unsafe_allow_html=True)
                st.pyplot(plot_actual_vs_predicted(df, train_size, seq_len, y_pred, ticker), use_container_width=True)
            with col_p2:
                st.markdown('<div class="section-header">📉 Training Loss</div>', unsafe_allow_html=True)
                st.pyplot(plot_loss_curve(history), use_container_width=True)

            # Scatter plot
            st.markdown('<div class="section-header"> Prediction Accuracy (Scatter)</div>', unsafe_allow_html=True)
            fig_sc, ax_sc = plt.subplots(figsize=(7, 5))
            ax_sc.scatter(y_actual, y_pred, alpha=0.5, color="#00d2ff", s=15)
            mn = min(y_actual.min(), y_pred.min())
            mx = max(y_actual.max(), y_pred.max())
            ax_sc.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label="Perfect Prediction")
            ax_sc.set_xlabel("Actual Price", fontsize=11)
            ax_sc.set_ylabel("Predicted Price", fontsize=11)
            ax_sc.set_title("Actual vs Predicted (Scatter)", fontsize=13, color="white")
            ax_sc.legend(facecolor="#1a1a2e", edgecolor="#00d2ff", labelcolor="white")
            fig_sc.tight_layout()
            st.pyplot(fig_sc, use_container_width=True)


        # ══════════════════════════════════════════
        #  TAB 3: 7-DAY FORECAST
        # ══════════════════════════════════════════
        with tab3:
            st.markdown('<div class="section-header">🔮 7-Day Price Forecast</div>', unsafe_allow_html=True)

            # Use last seq_len closing prices as seed
            scaled_all   = scaler.transform(close_prices.reshape(-1, 1))
            last_seq     = scaled_all[-seq_len:]
            future_prices = predict_next_7_days(model, last_seq, scaler, seq_len)

            # Signal
            signal_label, signal_class, change_pct = generate_signal(future_prices, current_price)
            direction = "📈 Upward" if change_pct >= 0 else "📉 Downward"

            col_sig1, col_sig2 = st.columns([1, 2])
            with col_sig1:
                st.markdown(f"<div class='{signal_class}'>{signal_label}</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='text-align:center; margin-top:0.5rem; color:#aaa'>
                    Predicted Trend: <b style='color:#00d2ff'>{direction}</b><br>
                    Avg Change: <b style='color:{"#38ef7d" if change_pct >= 0 else "#fc4a1a"}'>{change_pct:+.2f}%</b>
                </div>
                """, unsafe_allow_html=True)

            with col_sig2:
                # Future table
                future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
                future_df = pd.DataFrame({
                    "Date":           [d.strftime("%A, %b %d %Y") for d in future_dates],
                    "Predicted Price":["${:.2f}".format(p) for p in future_prices],
                    "Change vs Today":["${:+.2f} ({:+.2f}%)".format(p - current_price, (p - current_price)/current_price*100)
                                       for p in future_prices],
                    "Trend":          ["📈 Up" if p >= current_price else "📉 Down" for p in future_prices]
                })
                st.dataframe(future_df, hide_index=True, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.pyplot(plot_future_7_days(future_prices, df.index[-1], current_price, ticker), use_container_width=True)

            st.markdown("""
            <div class='info-box' style='margin-top:1rem'>
            ⚠️ <b>Important Note:</b> These predictions are generated by a machine learning model trained on historical price data only. 
            Stock prices are affected by countless real-world factors not captured here. 
            This is for <b>educational purposes</b> — not financial advice.
            </div>
            """, unsafe_allow_html=True)


        # ══════════════════════════════════════════
        #  TAB 4: RAW DATA
        # ══════════════════════════════════════════
        with tab4:
            st.markdown('<div class="section-header">📋 Historical Price Data</div>', unsafe_allow_html=True)

            display_df = df[["Open", "High", "Low", "Close", "Volume", "MA50", "MA200"]].copy()
            display_df = display_df.round(2)
            display_df.index = display_df.index.strftime("%Y-%m-%d")
            display_df.index.name = "Date"

            st.dataframe(display_df.sort_index(ascending=False), use_container_width=True, height=500)

            # Correlation heatmap
            st.markdown('<div class="section-header">🔗 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
            fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
            corr = df[["Open", "High", "Low", "Close", "Volume"]].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax_corr, cbar_kws={"shrink": 0.8},
                        linewidths=0.5, linecolor="#333")
            ax_corr.set_title("Correlation Between OHLCV Features", color="white", fontsize=13)
            fig_corr.tight_layout()
            st.pyplot(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.markdown("""
        **Common Issues:**
        - Invalid ticker symbol → Check spelling (use `.NS` suffix for Indian stocks)
        - No internet connection → Verify your network
        - Insufficient data → Try a shorter period or different ticker
        """)
