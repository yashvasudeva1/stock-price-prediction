# visuals/plots.py

import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# 1. Close Price Chart
# -----------------------------
def plot_close_price(df):
    fig = px.line(
        df.reset_index(), 
        x=df.reset_index().columns[0],  # first column = Date index
        y="Close",
        title="Close Price Over Time"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Close Price")
    return fig

# -----------------------------
# 2. Candlestick Chart
# -----------------------------
def plot_candlestick(df):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        ]
    )
    fig.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    return fig

# -----------------------------
# 3. Volume Chart
# -----------------------------
def plot_volume(df):
    fig = px.bar(df, x="Date", y="Volume", title="Trading Volume")
    fig.update_layout(xaxis_title="Date", yaxis_title="Volume")
    return fig

# -----------------------------
# 4. Moving Averages Chart
# -----------------------------
def plot_moving_averages(df):
    fig = px.line(title="Moving Averages")

    fig.add_scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price")

    for ma in ["SMA_5", "SMA_10", "SMA_20"]:
        if ma in df.columns:
            fig.add_scatter(x=df["Date"], y=df[ma], mode="lines", name=ma)

    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    return fig

# -----------------------------
# 5. RSI Chart
# -----------------------------
def plot_rsi(df):
    if "RSI_14" not in df.columns:
        raise ValueError("RSI_14 column not found. Add indicators first.")

    fig = px.line(df, x="Date", y="RSI_14", title="RSI (14)")

    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")

    fig.update_layout(xaxis_title="Date", yaxis_title="RSI")
    return fig

# -----------------------------
# 6. MACD Chart
# -----------------------------
def plot_macd(df):
    if "MACD" not in df.columns or "MACD_SIGNAL" not in df.columns:
        raise ValueError("MACD columns not found. Add indicators first.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], mode="lines", name="Signal"))

    fig.update_layout(title="MACD", xaxis_title="Date", yaxis_title="MACD Value")
    return fig

# -----------------------------
# 7. Price vs Target Comparison (ANN output)
# -----------------------------
def plot_pred_vs_actual(df_pred):
    """
    df_pred must contain:
    Date, y_true, y_pred
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred["Date"], y=df_pred["y_true"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=df_pred["Date"], y=df_pred["y_pred"], mode="lines", name="Predicted"))

    fig.update_layout(
        title="Actual vs Predicted Close Prices",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )
    return fig
