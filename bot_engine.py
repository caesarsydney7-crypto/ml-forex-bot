import os
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class VisualQuantEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.scaler = StandardScaler()

    def run_pipeline(self):
        # 1. Ingest & Process Data
        print("Ingesting market matrix...")
        df = yf.download(self.symbol, period="30d", interval="15m")
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # 2. Advanced Math Calculations
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        df['Target'] = np.where(df['Close'].shift(-4) > (df['Close'] + df['ATR'] * 1.2), 1, 
                       np.where(df['Close'].shift(-4) < (df['Close'] - df['ATR'] * 1.2), 0, np.nan))
        df.dropna(subset=['Target'], inplace=True)

        feature_cols = ['Log_Return', 'ATR', 'RSI']
        X = df[feature_cols]
        y = df['Target'].astype(int)

        # 3. Machine Learning Training Engine
        split = int(len(X) * 0.85)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predict most recent state
        latest_scaled = self.scaler.transform(X.tail(1))
        prediction = model.predict(latest_scaled)[0]
        confidence = model.predict_proba(latest_scaled)[0][prediction]

        # 4. GRAPHICS ENGINE GENERATION LAYER
        self.generate_dashboard(df.tail(100), prediction, confidence)

    def generate_dashboard(self, plot_df, pred, conf):
        print("Generating professional dashboard graphic...")
        sns.set_theme(style="darkgrid")
        
        # Create a beautiful 2-panel visual canvas
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        fig.suptitle(f"AI QUANT METRICS INTERFACE: {self.symbol}", fontsize=16, fontweight='bold', color='#111111')

        # Panel 1: Price Action Matrix with Volatility Barriers
        ax1.plot(plot_df.index, plot_df['Close'], color='#2471A3', label='Market Price (EURUSD)', linewidth=2)
        ax1.fill_between(plot_df.index, plot_df['Close'] + plot_df['ATR'], plot_df['Close'] - plot_df['ATR'], 
                         color='#2471A3', alpha=0.15, label='Volatility Envelope (ATR)')
        ax1.set_title("Market Structural Framework & Price Action", fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')

        # Panel 2: Relative Strength Technical Engine
        ax2.plot(plot_df.index, plot_df['RSI'], color='#E67E22', label='RSI Data Array', linewidth=1.5)
        ax2.axhline(70, color='#C0392B', linestyle='--', alpha=0.6)
        ax2.axhline(30, color='#27AE60', linestyle='--', alpha=0.6)
        ax2.set_title("Neural Feature Inputs (Relative Strength Index)", fontsize=12, fontweight='bold')
        ax2.set_ylim(10, 90)

        # Inject Visual Metric Status Overlay HUD Boxes
        status_text = f"PREDICTION MODEL DIRECTION: {'UP TREND (BUY)' if pred == 1 else 'DOWN TREND (SELL)'}\nPATTERN CONSENSUS CONFIDENCE: {conf*100:.2f}%"
        box_color = '#27AE60' if pred == 1 else '#C0392B'
        
        fig.text(0.13, 0.02, status_text, fontsize=11, fontweight='bold', color='white',
                 bbox=dict(facecolor=box_color, alpha=0.85, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        
        # Save output graphic layout binary
        plt.savefig("trading_dashboard.png", dpi=150)
        plt.close()
        print("Dashboard output written to trading_dashboard.png")

    def _calculate_rsi(self, series, window):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        return 100 - (100 / (1 + (gain / (loss + 1e-8))))

if __name__ == "__main__":
    engine = VisualQuantEngine("EURUSD=X")
    engine.run_pipeline()
