# models/ann.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def build_regression_ann(input_shape, units=[64, 32], lr=1e-3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for u in units:
        model.add(tf.keras.layers.Dense(u, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

def create_windowed_dataset(df, feature_cols, target_col="Close", window_size=20):
    """
    Builds X,y where X are window_size timesteps of features and y is next-day target.
    Returns: X (n, window, features), y (n,)
    """
    data = df[feature_cols + [target_col]].values
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :len(feature_cols)])
        y.append(data[i+window_size, -1])  # target_col at future index
    X = np.array(X)
    y = np.array(y)
    return X, y

class ScalerWrapper:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit_transform(self, X, y):
        # X shape: (n, window, features) -> reshape to 2D
        n, w, f = X.shape
        X2 = X.reshape(n*w, f)
        Xs = self.feature_scaler.fit_transform(X2).reshape(n, w, f)
        ys = self.target_scaler.fit_transform(y.reshape(-1,1)).reshape(-1)
        return Xs, ys

    def transform(self, X, y=None):
        n, w, f = X.shape
        X2 = X.reshape(n*w, f)
        Xs = self.feature_scaler.transform(X2).reshape(n, w, f)
        if y is None:
            return Xs
        ys = self.target_scaler.transform(y.reshape(-1,1)).reshape(-1)
        return Xs, ys

    def inverse_transform_target(self, y_scaled):
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1,1)).reshape(-1)
