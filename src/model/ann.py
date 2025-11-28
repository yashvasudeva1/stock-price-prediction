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
def create_windowed_dataset(df, feature_cols, target_col="Close", window_size=20):
    """
    Creates:
        X → (num_samples, window_size, num_features)
        y → (num_samples,)
    """
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

        # Flatten for scaling
        X2 = X.reshape(n * w, f)
        X_scaled = self.feature_scaler.fit_transform(X2).reshape(n, w, f)

        y_scaled = self.target_scaler.fit_transform(
            y.reshape(-1, 1)
        ).reshape(-1)

        return X_scaled, y_scaled

    def transform(self, X, y=None):
        n, w, f = X.shape

        X2 = X.reshape(n * w, f)
        X_scaled = self.feature_scaler.transform(X2).reshape(n, w, f)

        if y is None:
            return X_scaled

        y_scaled = self.target_scaler.transform(
            y.reshape(-1, 1)
        ).reshape(-1)

        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled):
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).reshape(-1)


# ==========================================================
# 4. FUTURE PREDICTION — N-DAY FORECAST
# ==========================================================
def predict_next_n_days(model, scaler, df, feature_cols, window_size=30, n_days=7):

    df = df.copy()

    # Scale ALL features (same as training)
    features = df[feature_cols].values
    scaled_features = scaler.feature_scaler.transform(features)

    # Get last window
    last_window = scaled_features[-window_size:].reshape(
        1, window_size, len(feature_cols)
    )

    predictions = []
    current_window = last_window.copy()

    # Index of Close in the feature columns
    try:
        close_idx = feature_cols.index("Close")
    except ValueError:
        raise Exception("❌ 'Close' must be included in feature_cols for forecasting.")

    for _ in range(n_days):

        # Predict next day (scaled)
        pred_scaled = model.predict(current_window, verbose=0)[0][0]

        # Convert back to real scale
        pred_unscaled = scaler.target_scaler.inverse_transform(
            np.array(pred_scaled).reshape(-1, 1)
        )[0][0]

        predictions.append(pred_unscaled)

        # -----------------------------
        # Build the next window
        # -----------------------------
        new_row_scaled = current_window[:, -1, :].copy().reshape(1, -1)

        # Update ONLY the Close column
        new_row_scaled[0, close_idx] = pred_scaled

        # Slide window: remove first, append new row
        current_window = np.concatenate(
            [current_window[:, 1:, :], new_row_scaled.reshape(1, 1, len(feature_cols))],
            axis=1
        )

    # Generate future dates
    future_dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        periods=n_days
    )

    # Return prediction dataframe
    return pd.DataFrame({
        "Date": future_dates,
        "y_true": [None] * n_days,
        "y_pred": predictions
    })
