# features/essential_columns.py

import pandas as pd
import numpy as np

def add_essential_columns(df):
    """
    Takes a cleaned DataFrame with at least:
        Date, Close, High, Low, Open, Volume
    Returns a DataFrame with useful engineered features for ANN models.
    """

    # --- BASIC RETURN/RANGE FEATURES ---
    df["Price_Change"] = df["Close"].pct_change().fillna(0)
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Low"]
    df["Close_Open_Change"] = (df["Close"] - df["Open"]) / df["Open"]

    # --- MOVING AVERAGES ---
    df["SMA_5"] = df["Close"].rolling(5).mean().fillna(method="bfill")
    df["SMA_10"] = df["Close"].rolling(10).mean().fillna(method="bfill")
    df["SMA_20"] = df["Close"].rolling(20).mean().fillna(method="bfill")

    # --- EXPONENTIAL MOVING AVERAGES ---
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # --- VOLATILITY ---
    df["Volatility_5"] = df["Close"].pct_change().rolling(5).std().fillna(0)
    df["Volatility_10"] = df["Close"].pct_change().rolling(10).std().fillna(0)

    # --- VOLUME FEATURES ---
    df["Volume_Change"] = df["Volume"].pct_change().fillna(0)
    df["Volume_MA_5"] = df["Volume"].rolling(5).mean().fillna(method="bfill")

    # --- PRICE MOMENTUM ---
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    df["Target_Close"] = df["Close"].shift(-1)

    df.fillna(0, inplace=True)
    return df
