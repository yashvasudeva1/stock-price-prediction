import numpy as np
import pandas as pd

def predict_next_n_days(model, scaler, df, window_size=30, n_days=7):
    """
    Predict next N days of Close prices using sliding window method.
    Requires:
    - model: trained ANN model
    - scaler: fitted MinMaxScaler
    - df: cleaned dataframe with essential columns
    - window_size: number of past days used as input
    - n_days: how many days to forecast
    """

    df = df.copy()

    # Get only Close column for prediction
    close = df["Close"].values.reshape(-1, 1)

    # Scale close prices using your scaler wrapper
    scaled_close = scaler.transform(close)

    # Take the last "window_size" values
    last_window = scaled_close[-window_size:].reshape(1, window_size)

    predictions_scaled = []
    predictions_unscaled = []

    current_window = last_window.copy()

    for _ in range(n_days):
        pred_scaled = model.predict(current_window, verbose=0)[0][0]
        predictions_scaled.append(pred_scaled)

        pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions_unscaled.append(pred_unscaled)

        # Slide window
        new_window = np.append(current_window[:, 1:], [[pred_scaled]], axis=1)
        current_window = new_window

    # Build output dataframe
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days)

    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": predictions_unscaled
    })

    return pred_df
