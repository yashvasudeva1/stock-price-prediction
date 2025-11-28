# ann_multistep.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer

# ==========================================================
# 1. SCALER CLASS
# ==========================================================
class MultiScaler:
    """
    Wrapper to handle separate scaling for Features (Input) and Target (Output).
    This ensures we can inverse transform the predictions correctly later.
    """
    def __init__(self):
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

    def fit_transform(self, X_raw, y_raw):
        """
        Fits scalers on raw data and returns scaled data.
        X_raw: 2D numpy array (samples, features)
        y_raw: 1D or 2D numpy array (samples, target)
        """
        # Fit/Transform Features
        X_scaled = self.feature_scaler.fit_transform(X_raw)
        
        # Fit/Transform Target (reshaping needed for single column)
        y_reshaped = y_raw.reshape(-1, 1)
        y_scaled = self.target_scaler.fit_transform(y_reshaped)
        
        return X_scaled, y_scaled.flatten()

    def transform_features(self, X_raw):
        """Scales features only (used during inference)."""
        return self.feature_scaler.transform(X_raw)

    def inverse_transform_y(self, y_scaled):
        """
        Converts scaled predictions back to original price domain.
        y_scaled shape: (1, forecast_days) or (n, forecast_days)
        """
        # If input is 1D (forecast_days,), reshape to (forecast_days, 1)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        # If input is (1, forecast_days), we transpose to (forecast_days, 1) 
        # for the scaler, then transpose back or flatten.
        # However, standard inverse_transform expects (n_samples, n_features).
        # Our target scaler was trained on shape (n, 1).
        
        # We must iterate or reshape carefully. 
        # Since we predicted 'forecast_days' separately, each point represents a target value.
        
        original_shape = y_scaled.shape
        y_flat = y_scaled.flatten().reshape(-1, 1)
        y_inv = self.target_scaler.inverse_transform(y_flat)
        
        return y_inv.reshape(original_shape)


# ==========================================================
# 2. PREPROCESS DATA (WINDOWING & FLATTENING)
# ==========================================================
def preprocess_data(df, feature_cols, target_col, window_size, forecast_days, scaler=None):
    """
    Creates the dataset for Direct Multi-Step forecasting.
    
    Logic:
    Input X: Flattened window of past 'window_size' days.
    Output y: Vector of future 'forecast_days' prices.
    """
    # 1. Extract raw numpy arrays
    data_X = df[feature_cols].values
    data_y = df[target_col].values

    # 2. Scale Data
    if scaler is None:
        raise ValueError("Scaler instance must be provided.")
    
    X_scaled_full, y_scaled_full = scaler.fit_transform(data_X, data_y)

    X, y = [], []

    # 3. Create Windows
    # We need enough data for the window AND the forecast horizon
    # Loop stops when we don't have enough future data for y
    total_len = len(X_scaled_full)
    stop_index = total_len - window_size - forecast_days + 1

    for i in range(stop_index):
        # Window of features
        window = X_scaled_full[i : i + window_size, :]
        
        # FLATTEN: Convert (window_size, num_features) -> (window_size * num_features, )
        # This makes it compatible with a standard Dense (Fully Connected) Input Layer
        X.append(window.flatten())
        
        # Forecast Horizon (Direct Strategy)
        # We take the next 'forecast_days' values of the target
        future_vals = y_scaled_full[i + window_size : i + window_size + forecast_days]
        y.append(future_vals)

    return np.array(X), np.array(y)


# ==========================================================
# 3. BUILD MULTI-STEP ANN MODEL
# ==========================================================
def build_multistep_ann(input_size, output_steps, learning_rate=0.001):
    """
    Builds a Feed-Forward Neural Network (MLP).
    
    input_size: window_size * num_features
    output_steps: forecast_days (The model predicts N days at once)
    """
    model = Sequential()
    
    # Input Layer
    model.add(InputLayer(input_shape=(input_size,)))
    
    # Hidden Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) # Regularization
    
    model.add(Dense(32, activation='relu'))
    
    # Output Layer
    # Direct Strategy: One neuron per future day to predict
    model.add(Dense(output_steps, activation='linear')) 

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model


# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    """
    Trains the ANN model.
    """
    history = model.fit(
        X_train, 
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True
    )
    return history


# ==========================================================
# 5. PREDICT FUTURE (INFERENCE)
# ==========================================================
def predict_future(model, scaler, df, feature_cols, window_size, forecast_days):
    """
    Takes the *last available* window from the dataframe, 
    predicts the next 'forecast_days' at once, and returns a DataFrame.
    """
    # 1. Get the last window of data
    if len(df) < window_size:
        raise ValueError(f"Not enough data. Need at least {window_size} rows.")
        
    last_window_df = df[feature_cols].tail(window_size)
    last_window_values = last_window_df.values
    
    # 2. Scale the features
    # Note: We use transform, NOT fit_transform, to keep scaling consistent with training
    last_window_scaled = scaler.transform_features(last_window_values)
    
    # 3. Flatten inputs for ANN (1, window_size * features)
    # Reshape logic: (1 sample, total_flattened_features)
    input_flattened = last_window_scaled.flatten().reshape(1, -1)
    
    # 4. Predict
    # Output shape will be (1, forecast_days)
    prediction_scaled = model.predict(input_flattened, verbose=0)
    
    # 5. Inverse Transform
    # Convert scaled 0-1 values back to prices
    prediction_real = scaler.inverse_transform_y(prediction_scaled[0])
    
    # 6. Create Result DataFrame
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'y_pred': prediction_real.flatten()
    })
    
    return pred_df
