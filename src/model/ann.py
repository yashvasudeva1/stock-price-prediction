# models/ann.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd


# ----------------------------------------------------------
# 1. ANN MODEL
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# 2. CREATE WINDOWED DATASET
# ----------------------------------------------------------
def create_windowed_dataset(df, feature_cols, target_col="Close", window_size=20):
    """
    Creates X, y:
    - X shape: (samples, window_size, features)
    - y: next day's target value
    """
    data = df[feature_cols + [target_col]].values
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :len(feature_cols)])
        y.append(data[i+window_size, -1])  # target

    return np.array(X), np.array(y)


# ----------------------------------------------------------
# 3. SCALER WRAPPER
# ----------------------------------------------------------
class ScalerWrapper:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit_transform(self, X, y):
        n, w, f = X.shape

        X2 = X.reshape(n * w, f)
        X_scaled = self.feature_scaler.fit_transform(X2).reshape(n, w, f)

        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

        return X_scaled, y_scaled

    def transform(self, X, y=None):
        n, w, f = X.shape

        X2 = X.reshape(n * w, f)
        X_scaled = self.feature_scaler.transform(X2).reshape(n, w, f)

        if y is None:
            return X_scaled

        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).reshape(-1)
        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled):
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


# ----------------------------------------------------------
# 4. FUTURE PREDICTION (N-Day Forecast)
# ----------------------------------------------------------
def predict_next_n_days(model, scaler, df, feature_cols, window_size=30, n_days=7):

    df = df.copy()

    # scale ALL features the same way as training
    features = df[feature_cols].values
    scaled_features = scaler.feature_scaler.transform(features)

    # prepare last window
    last_window = scaled_features[-window_size:].reshape(1, window_size, len(feature_cols))

    predictions = []
    current_window = last_window.copy()

    for _ in range(n_days):
        # predict next point
        pred_scaled = model.predict(current_window, verbose=0)[0][0]

        # return original scale
        pred_unscaled = scaler.target_scaler.inverse_transform(
            np.array(pred_scaled).reshape(-1, 1)
        )[0][0]
        predictions.append(pred_unscaled)

        # build new window: drop first row, append prediction
        new_row_scaled = np.zeros((1, len(feature_cols)))
        new_row_scaled[0, feature_cols.index("Close")] = pred_scaled  

        current_window = np.concatenate(
            [current_window[:, 1:, :],
             new_row_scaled.reshape(1, 1, len(feature_cols))],
            axis=1
        )

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)

    return pd.DataFrame({
        "Date": future_dates,
        "y_true": [None] * n_days,
        "y_pred": predictions
    })

