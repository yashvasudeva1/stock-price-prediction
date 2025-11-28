# analysis/eda.py

import pandas as pd
import numpy as np

# ----------------------------------------------------
# 1. Basic Dataset Overview
# ----------------------------------------------------
def dataset_summary(df):
    return {
        "total_rows": len(df),
        "start_date": df.index.min(),
        "end_date": df.index.max(),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
    }


# ----------------------------------------------------
# 2. Summary statistics (describe)
# ----------------------------------------------------
def summary_stats(df):
    return df.describe()

# ----------------------------------------------------
# 3. Trading day analysis
# ----------------------------------------------------
def trading_day_info(df):
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    df_sorted["Day_Diff"] = df_sorted["Date"].diff().dt.days

    return {
        "total_trading_days": len(df_sorted),
        "average_gap_days": df_sorted["Day_Diff"].mean(),
        "max_gap_days": df_sorted["Day_Diff"].max(),
        "missing_days_flag": df_sorted["Day_Diff"].max() > 3
    }

# ----------------------------------------------------
# 4. Daily returns analysis
# ----------------------------------------------------
def daily_return_stats(df):
    returns = df["Close"].pct_change().dropna()

    return {
        "mean_daily_return": returns.mean(),
        "volatility_daily": returns.std(),
        "max_gain": returns.max(),
        "max_loss": returns.min(),
        "positive_days": (returns > 0).sum(),
        "negative_days": (returns < 0).sum(),
    }

# ----------------------------------------------------
# 5. Outlier detection (Z-score)
# ----------------------------------------------------
def detect_outliers(df, col="Close", threshold=3):
    mean = df[col].mean()
    std = df[col].std()
    df["Z"] = (df[col] - mean) / std
    outliers = df[np.abs(df["Z"]) > threshold]
    df.drop(columns=["Z"], inplace=True)
    return outliers

# ----------------------------------------------------
# 6. Correlation matrix
# ----------------------------------------------------
def correlation_matrix(df):
    return df.corr()

# ----------------------------------------------------
# 7. Rolling statistics
# ----------------------------------------------------
def rolling_stats(df, window=20):
    df = df.copy()
    df["Rolling_Mean"] = df["Close"].rolling(window).mean()
    df["Rolling_Std"] = df["Close"].rolling(window).std()
    return df[["Date", "Rolling_Mean", "Rolling_Std"]]

# ----------------------------------------------------
# 8. Volume spike detection
# ----------------------------------------------------
def volume_spikes(df, threshold_factor=2):
    avg_volume = df["Volume"].rolling(20).mean()
    spikes = df[df["Volume"] > threshold_factor * avg_volume]
    return spikes

# ----------------------------------------------------
# 9. Uptrend / Downtrend streaks
# ----------------------------------------------------
def trend_streaks(df):
    df = df.copy()
    df["Change"] = df["Close"].diff()

    streaks = {"uptrend_max": 0, "downtrend_max": 0}
    current_up = 0
    current_down = 0

    for ch in df["Change"]:
        if ch > 0:
            current_up += 1
            current_down = 0
        elif ch < 0:
            current_down += 1
            current_up = 0

        streaks["uptrend_max"] = max(streaks["uptrend_max"], current_up)
        streaks["downtrend_max"] = max(streaks["downtrend_max"], current_down)

    return streaks
