# models/ann.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ==========================================================
# 1. BUILD MULTI-STEP ANN
# ==========================================================
def build_multistep_ann(input_shape, output_steps, units=[64, 32], lr=1e-3):
    """
    input_shape = (window_size * num_features,)
    output_steps = number of future days to predict
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    for u in units:
        model.add(tf.keras.layers.Dense(u, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.2))

    # Predict N future days at ONCE
    model.add(tf.keras.layers.Dense(output_steps))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ==========================================================
# 2. CREATE WINDOWED DATASET
# ==========================================================
def create_multistep_dataset(df, feature_cols, target_col, window_size, forecast_days):
    """
    X: (samples, window_size * num_features)
    y: (samples, forecast_days)
    """
    data = df[feature_cols + [target_col]].values
    num_features = len(feature_cols)

    X, y = [], []

    for i in range(len(data) - window_size - forecast_days):
        window = data[i : i + window_size, :num_features].flatten()
        future = data[i + window_size : i + window_size + forecast_days, -1]
        X.append(window)
        y.append(future)

    return np.array(X), np.array(y)


# ==========================================================
# 3. SCALER
# ==========================================================
class MultiScaler:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit_transform(self, X, y):
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        return X_scaled, y_scaled

    def transform(self, X):
        return self.feature_scaler.transform(X)

    def inverse_transform_y(self, y_scaled):
        return self.target_scaler.inverse_transform(y_scaled)


# ==========================================================
# 4. FUTURE PREDICTION (Single forward pass)
# ==========================================================
def predict_future(model, scaler, df, feature_cols, window_size, n_days):
    df = df.copy()

    data = df[feature_cols].values
    last_window = data[-window_size:].flatten().reshape(1, -1)

    # scale using trained scaler
    last_window_scaled = scaler.feature_scaler.transform(last_window)

    # predict N future days in ONE pass
    pred_scaled = model.predict(last_window_scaled, verbose=0)[0]

    # invert scale
    pred = scaler.inverse_transform_y(pred_scaled.reshape(-1, 1)).flatten()

    # build dataframe
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)

    return pd.DataFrame({
        "Date": future_dates,
        "y_pred": pred
    })
