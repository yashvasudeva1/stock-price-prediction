# models/ann.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# ==========================================================
# 1. BUILD ANN MODEL
# ==========================================================
def build_regression_ann(input_shape, units=[64, 32], lr=1e-3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    for u in units:
        model.add(tf.keras.layers.Dense(u, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ==========================================================
# 2. WINDOWED DATASET CREATOR
# ==========================================================
def create_windowed_dataset(df, feature_cols, target_col="Target_Close", window_size=20):
    data = df[feature_cols + [target_col]].values
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :len(feature_cols)])
        y.append(data[i + window_size, -1])

    return np.array(X), np.array(y)


# ==========================================================
# 3. SCALER WRAPPER
# ==========================================================
class ScalerWrapper:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit_transform(self, X, y):
        n, w, f = X.shape
        X_flat = X.reshape(n * w, f)
        X_scaled = self.feature_scaler.fit_transform(X_flat).reshape(n, w, f)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
        return X_scaled, y_scaled

    def transform(self, X, y=None):
        n, w, f = X.shape
        X_flat = X.reshape(n * w, f)
        X_scaled = self.feature_scaler.transform(X_flat).reshape(n, w, f)
        if y is None:
            return X_scaled
        y_scaled = self.target_scaler.fit_transform(
            y.reshape(-1, 1).astype("float64")
        ).reshape(-1)
        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled):
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).reshape(-1)


# ==========================================================
# 4. HELPER: Generate realistic features from Close price
# ==========================================================
def generate_features_from_close(close_price, historical_volatility=0.02):
    """
    Generate realistic OHLV features from a predicted Close price.
    Uses small random variations based on historical volatility.
    """
    # Add small random variations
    variation = np.random.uniform(-historical_volatility, historical_volatility)
    
    open_price = close_price * (1 + variation * 0.5)
    high_price = max(open_price, close_price) * (1 + abs(variation) * 0.3)
    low_price = min(open_price, close_price) * (1 - abs(variation) * 0.3)
    
    # Volume estimation (simplified - could be improved)
    volume = np.random.uniform(0.8, 1.2)  # Relative volume
    
    return {
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume
    }


# ==========================================================
# 5. N-DAY FORECAST (IMPROVED VERSION)
# ==========================================================
def predict_next_n_days(model, scaler, df, feature_cols, window_size=30, n_days=7):
    """
    Predict future stock prices for n_days.
    
    IMPORTANT: This function now intelligently handles different feature types:
    - For price-based features (Open, High, Low, Close): Generates realistic values
    - For technical indicators: Uses simple heuristics or carries forward
    - For Volume: Uses average historical volume
    """
    
    df = df.copy()

    # Get historical features
    features = df[feature_cols].astype("float64").values
    scaled_features = scaler.feature_scaler.transform(features)

    current_window = scaled_features[-window_size:].reshape(
        1, window_size, len(feature_cols)
    )

    predictions = []
    
    # Get feature indices
    feature_indices = {col: feature_cols.index(col) for col in feature_cols}
    
    # Check if Close exists
    if "Close" not in feature_indices:
        raise Exception("❌ 'Close' must be included in feature_cols for forecasting.")
    
    close_idx = feature_indices["Close"]

    # Calculate historical volatility for realistic feature generation
    historical_volatility = df["Close"].pct_change().std() if len(df) > 1 else 0.02

    # Scaler bounds
    min_val = scaler.target_scaler.data_min_[0]
    max_val = scaler.target_scaler.data_max_[0]

    for day in range(n_days):
        # Predict next Close price
        pred_scaled = model.predict(current_window, verbose=0)[0][0]

        # Numeric safety
        if pred_scaled is None or np.isnan(pred_scaled) or np.isinf(pred_scaled):
            pred_scaled = 0.0

        # Clip to scaler limits
        pred_scaled = float(np.clip(pred_scaled, min_val, max_val))
        pred_scaled_arr = np.array([[pred_scaled]], dtype="float64")

        # Get unscaled prediction
        pred_unscaled = scaler.target_scaler.inverse_transform(pred_scaled_arr)[0][0]
        predictions.append(pred_unscaled)

        # BUILD NEXT WINDOW WITH UPDATED FEATURES
        new_row_scaled = current_window[:, -1, :].copy().reshape(1, -1)
        
        # Update Close price
        new_row_scaled[0, close_idx] = pred_scaled
        
        # Update other price features if they exist
        # Generate realistic OHLV values based on predicted Close
        generated_features = generate_features_from_close(pred_unscaled, historical_volatility)
        
        # Create unscaled feature row for proper scaling
        new_row_unscaled = np.zeros((1, len(feature_cols)))
        
        for col in feature_cols:
            idx = feature_indices[col]
            
            if col == "Close":
                new_row_unscaled[0, idx] = pred_unscaled
            elif col == "Open":
                new_row_unscaled[0, idx] = generated_features['Open']
            elif col == "High":
                new_row_unscaled[0, idx] = generated_features['High']
            elif col == "Low":
                new_row_unscaled[0, idx] = generated_features['Low']
            elif col == "Volume":
                # Use average historical volume
                new_row_unscaled[0, idx] = df["Volume"].mean() if "Volume" in df.columns else 1.0
            else:
                # For technical indicators (RSI, MACD, etc.), carry forward the last value
                # This is a simplification - ideally you'd recalculate them
                last_unscaled_value = scaler.feature_scaler.inverse_transform(
                    current_window[:, -1, :].reshape(1, -1)
                )[0, idx]
                new_row_unscaled[0, idx] = last_unscaled_value
        
        # Scale the complete new row
        new_row_scaled = scaler.feature_scaler.transform(new_row_unscaled)

        # Update window
        current_window = np.concatenate(
            [
                current_window[:, 1:, :],
                new_row_scaled.reshape(1, 1, len(feature_cols)),
            ],
            axis=1,
        )

    # Generate future dates
    future_dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=n_days
    )

    return pd.DataFrame({
        "Date": future_dates,
        "y_true": [None] * n_days,
        "y_pred": predictions
    })


# ==========================================================
# 6. ALTERNATIVE: Simple version using only Close prices
# ==========================================================
def predict_next_n_days_simple(model, scaler, df, feature_cols, window_size=30, n_days=7):
    """
    Simplified version - ONLY works if feature_cols contains just ['Close']
    or price-related features that can be derived from Close.
    
    RECOMMENDED: Train your model with feature_cols = ['Close'] for best results.
    """
    
    if feature_cols != ['Close']:
        print("⚠️ WARNING: For best results, retrain model with feature_cols=['Close']")
        print(f"   Current features: {feature_cols}")
        print("   Falling back to full version...")
        return predict_next_n_days(model, scaler, df, feature_cols, window_size, n_days)
    
    df = df.copy()
    features = df[feature_cols].astype("float64").values
    scaled_features = scaler.feature_scaler.transform(features)

    current_window = scaled_features[-window_size:].reshape(1, window_size, 1)
    predictions = []

    min_val = scaler.target_scaler.data_min_[0]
    max_val = scaler.target_scaler.data_max_[0]

    for _ in range(n_days):
        pred_scaled = model.predict(current_window, verbose=0)[0][0]
        pred_scaled = float(np.clip(pred_scaled, min_val, max_val))
        
        pred_unscaled = scaler.target_scaler.inverse_transform(
            np.array([[pred_scaled]], dtype="float64")
        )[0][0]
        predictions.append(pred_unscaled)

        # Simple: just append the prediction
        current_window = np.concatenate(
            [current_window[:, 1:, :], 
             np.array([[[pred_scaled]]])],
            axis=1
        )

    future_dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=n_days
    )

    return pd.DataFrame({
        "Date": future_dates,
        "y_true": [None] * n_days,
        "y_pred": predictions
    })
