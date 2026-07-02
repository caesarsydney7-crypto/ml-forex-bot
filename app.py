import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set up clean web page configuration
st.set_page_config(page_title="AI Quant Trading Engine", page_icon="🤖", layout="wide")

st.title("🤖 Autonomous ML Forex Analytics Interface")
st.markdown("Interact with live regularized machine learning architectures tracking high-frequency currency movements.")
st.sidebar.header("🎛️ Control Panel")

# User inputs right on the website sidebar
symbol_choice = st.sidebar.selectbox("Select Currency Pair", ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"])
timeframe = st.sidebar.selectbox("Timeframe Matrix", ["15m", "30m", "1h"])
safety_threshold = st.sidebar.slider("AI Signal Consensus Threshold (%)", 50, 90, 70)

@st.cache_data(ttl=900) # Caches data for 15 minutes to keep page ultra-fast
def fetch_live_data(symbol, interval):
    df = yf.download(symbol, period="30d", interval=interval)
    if not df.empty:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

try:
    df = fetch_live_data(symbol_choice, timeframe)
    
    # Machine Learning Data Engineering Pipelines
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    df['RSI'] = 100 - (100 / (1 + ((df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean()) / 
                                   ( -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean() + 1e-8))))
    df['Target'] = np.where(df['Close'].shift(-4) > (df['Close'] + df['ATR'] * 1.2), 1, 
                   np.where(df['Close'].shift(-4) < (df['Close'] - df['ATR'] * 1.2), 0, np.nan))
    df.dropna(subset=['Target'], inplace=True)

    X = df[['Log_Return', 'ATR', 'RSI']]
    y = df['Target'].astype(int)

    # Train Model on the Cloud Instance
    scaler = StandardScaler()
    split = int(len(X) * 0.85)
    X_train_scaled = scaler.fit_transform(X.iloc[:split])
    
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y.iloc[:split])

    # Run inference on live market state
    latest_scaled = scaler.transform(X.tail(1))
    prediction = model.predict(latest_scaled)[0]
    confidence = model.predict_proba(latest_scaled)[0][prediction]

    # --- UI Layout Design Components ---
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("🎯 System State")
        conf_percent = confidence * 100
        
        if conf_percent >= safety_threshold:
            if prediction == 1:
                st.success("🤖 AI PREDICTION:\n\n**BUY MODE (UPWARD WAVE)**")
            else:
                st.error("🤖 AI PREDICTION:\n\n**SELL MODE (DOWNWARD WAVE)**")
        else:
            st.warning("🤖 AI PREDICTION:\n\n**HOLDING IN CASH**")
            
        st.metric(label="Pattern Consensus Confidence", value=f"{conf_percent:.2f}%")
        st.info(f"Current Market Close: **{df['Close'].iloc[-1]:.5f}**")

    with col2:
        st.subheader("📊 Dynamic Market Structural Canvas")
        sns.set_theme(style="darkgrid")
        plot_df = df.tail(80)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(plot_df.index, plot_df['Close'], color='#2471A3', label='Live Rate', linewidth=2)
        ax.fill_between(plot_df.index, plot_df['Close'] + plot_df['ATR'], plot_df['Close'] - plot_df['ATR'], 
                         color='#2471A3', alpha=0.12, label='Volatility Bounds')
        plt.xticks(rotation=25)
        ax.legend()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Waiting for live market feed activation array: {e}")
