import pandas as pd

def add_essential_columns(df):
    df = df.copy()

    # ---------------------------------------------------
    # 1. RETURNS
    # ---------------------------------------------------
    df["Return"] = df["Close"].pct_change()

    # ---------------------------------------------------
    # 2. MOVING AVERAGES
    # ---------------------------------------------------
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # ---------------------------------------------------
    # 3. RSI (14)
    # ---------------------------------------------------
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ---------------------------------------------------
    # 4. MACD (12, 26, 9)
    # ---------------------------------------------------
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ---------------------------------------------------
    # 5. TARGET COLUMN (used for ANN)
    # ---------------------------------------------------
    df["Target_Close"] = df["Close"].shift(-1)

    # Remove rows that contain NaN (because of rolling windows)
    df = df.dropna()

    return df
