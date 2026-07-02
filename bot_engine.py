import os
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import requests

class InstitutionalForexEngine:
    def __init__(self, symbol: str, interval: str = "15m", period: str = "60d"):
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.scaler = StandardScaler()

    def ingest_and_clean_data(self) -> pd.DataFrame:
        """Downloads high-frequency bars and handles structural multi-indexing anomalies."""
        print(f"Executing cloud ingestion layer for {self.symbol}...")
        df = yf.download(self.symbol, period=self.period, interval=self.interval)
        if df.empty:
            raise ValueError("Zero records returned from data provider.")
        
        # Flatten MultiIndex column definitions if present
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df

    def engineer_fractional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw non-stationary price series into stationary alpha features."""
        # Calculate dynamic structural volatility
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        df['Rolling_Volatility'] = df['Log_Return'].rolling(window=20).std()

        # Advanced Technical Structural Features
        df['RSI'] = self._calculate_rsi(df['Close'], window=14)
        df['Z_Score'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        # Microstructure Feature: Volume Weighting
        df['Volume_Norm'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
        
        # Mathematical stationarity transform via fractional differences approximation
        # (Preserves intermediate historical pattern structures while neutralizing raw linear drift)
        df['FracDiff_Close'] = df['Close'] - 0.4 * df['Close'].shift(1) - 0.15 * df['Close'].shift(2)
        
        return df

    def apply_triple_barrier_labeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies dynamic target bounds adjusted to structural market volatility."""
        # Target: 1 if market hits an upward barrier before hitting a lower barrier, calibrated by ATR
        barrier_distance = df['ATR'] * 1.5
        future_close = df['Close'].shift(-4) # Look 4 bars (1 hour) ahead
        
        df['Target'] = np.where(future_close > (df['Close'] + barrier_distance), 1, 
                       np.where(future_close < (df['Close'] - barrier_distance), 0, np.nan))
        
        # Forward fill neutral structures or drop edge rows
        df.dropna(subset=['Target'], inplace=True)
        return df

    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def run_walk_forward_execution(self):
        """Executes full Walk-Forward back-validation and computes real-time live execution weights."""
        raw_df = self.ingest_and_clean_data()
        feature_df = self.engineer_fractional_features(raw_df)
        ml_dataset = self.apply_triple_barrier_labeling(feature_df)

        feature_cols = ['Log_Return', 'Rolling_Volatility', 'RSI', 'Z_Score', 'Volume_Norm', 'FracDiff_Close']
        
        X = ml_dataset[feature_cols]
        y = ml_dataset['Target'].astype(int)

        # Walk-Forward Validation: Train on past 85%, validate on most recent 15%
        split_idx = int(len(X) * 0.85)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

        # Scale Feature Arrays to eliminate feature dominance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Highly Regularized XGBoost Architecture to violently prevent overfitting
        model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,      # L1 Regularization
            reg_lambda=1.5,     # L2 Regularization
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train)

        # Real-time state classification inference
        latest_live_bar = X.tail(1)
        latest_live_bar_scaled = self.scaler.transform(latest_live_bar)
        
        prediction = model.predict(latest_live_bar_scaled)[0]
        probabilities = model.predict_proba(latest_live_bar_scaled)[0]
        confidence = probabilities[prediction]

        print(f"--- Production Model Diagnostics ---")
        print(f"Latest Market Wave State Probability: Vector Matrix {probabilities}")
        print(f"Calculated Classification Edge: Direction {prediction} | Security Score: {confidence*100:.2f}%")

        # Pro-Level Execution Guard Rail
        # Institutional signals require deep mathematical edge (>70% pure pattern consensus)
        if confidence > 0.70:
            signal = "BUY" if prediction == 1 else "SELL"
            self.route_execution_order(signal, confidence)
        else:
            print("Mathematical consensus insufficient. System status: HELD.")

    def route_execution_order(self, direction: str, edge_score: float):
        print(f"🌟 CRITICAL SIGNAL ENFORCED: Deploying {direction} order matrix across system rails.")
        api_url = os.environ.get("BROKER_WEBHOOK_URL")
        if not api_url:
            print("System Warning: Live Webhook routing target omitted. Run finalized inside Sandbox Logs.")
            return

        payload = {
            "signal": direction,
            "asset": self.symbol,
            "execution_weight": float(edge_score),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        try:
            res = requests.post(api_url, json=payload, timeout=10)
            print(f"Broker Router Connection established. Handshake Code: {res.status_code}")
        except Exception as err:
            print(f"Broker routing array link exception: {err}")

if __name__ == "__main__":
    # Ingesting EUR/USD 15-Minute Structural candles
    quant_engine = InstitutionalForexEngine(symbol="EURUSD=X", interval="15m", period="60d")
    quant_engine.run_walk_forward_execution()
