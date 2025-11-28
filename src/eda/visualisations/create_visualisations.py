# visuals/plots.py

import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------
# Helper function to ensure Date exists as a column
# ------------------------------------------------------
def prepare_df(df):
    """
    Prepare dataframe for plotting by ensuring Date is a column.
    Handles cases where Date is either an index or already a column.
    """
    df_plot = df.copy()
    
    # If 'Date' is in the index, reset it
    if df_plot.index.name == 'Date':
        df_plot = df_plot.reset_index()
    # If index is not named but Date is not in columns, assume index is Date
    elif 'Date' not in df_plot.columns:
        df_plot = df_plot.reset_index()
        df_plot = df_plot.rename(columns={'index': 'Date'})
    
    return df_plot


# ------------------------------------------------------
# 1. Close Price Chart
# ------------------------------------------------------
def plot_close_price(df):
    df_plot = prepare_df(df)
    fig = px.line(
        df_plot,
        x="Date",
        y="Close",
        title="Close Price Over Time"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Close Price")
    return fig


# ------------------------------------------------------
# 2. Candlestick Chart
# ------------------------------------------------------
def plot_candlestick(df):
    df_plot = prepare_df(df)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_plot["Date"],
                open=df_plot["Open"],
                high=df_plot["High"],
                low=df_plot["Low"],
                close=df_plot["Close"],
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        ]
    )

    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    return fig


# ------------------------------------------------------
# 3. Volume Chart
# ------------------------------------------------------
def plot_volume(df):
    df_plot = prepare_df(df)
    fig = px.bar(df_plot, x="Date", y="Volume", title="Trading Volume")
    fig.update_layout(xaxis_title="Date", yaxis_title="Volume")
    return fig


# ------------------------------------------------------
# 4. Moving Averages Chart
# ------------------------------------------------------
def plot_moving_averages(df):
    df_plot = prepare_df(df)

    fig = px.line(title="Moving Averages")
    fig.add_scatter(x=df_plot["Date"], y=df_plot["Close"], mode="lines", name="Close Price")

    for ma in ["SMA_5", "SMA_10", "SMA_20"]:
        if ma in df_plot.columns:
            fig.add_scatter(x=df_plot["Date"], y=df_plot[ma], mode="lines", name=ma)

    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    return fig


# ------------------------------------------------------
# 5. RSI Chart
# ------------------------------------------------------
def plot_rsi(df):
    df_plot = prepare_df(df)

    if "RSI_14" not in df_plot.columns:
        raise ValueError("RSI_14 column not found. Add indicators first.")

    fig = px.line(df_plot, x="Date", y="RSI_14", title="RSI (14)")

    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")

    fig.update_layout(xaxis_title="Date", yaxis_title="RSI")
    return fig


# ------------------------------------------------------
# 6. MACD Chart
# ------------------------------------------------------
def plot_macd(df):
    df_plot = prepare_df(df)

    if "MACD" not in df_plot.columns or "MACD_SIGNAL" not in df_plot.columns:
        raise ValueError("MACD columns not found. Add indicators first.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["MACD_SIGNAL"], mode="lines", name="Signal"))

    fig.update_layout(title="MACD", xaxis_title="Date", yaxis_title="MACD Value")
    return fig


# ------------------------------------------------------
# 7. Price vs Target Comparison (ANN output)
# ------------------------------------------------------
def plot_pred_vs_actual(df_plot):
    import plotly.graph_objects as go

    fig = go.Figure()

    # ----------------------------
    # 1. Add Actual Line (if exists)
    # ----------------------------
    if "y_true" in df_plot.columns and df_plot["y_true"].notna().any():
        fig.add_trace(go.Scatter(
            x=df_plot["Date"],
            y=df_plot["y_true"],
            mode="lines",
            name="Actual",
            line=dict(color="blue")
        ))

    # ----------------------------
    # 2. Always Add Prediction Line
    # ----------------------------
    fig.add_trace(go.Scatter(
        x=df_plot["Date"],
        y=df_plot["y_pred"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="red")
    ))

    fig.update_layout(
        title="Future Stock Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    return fig

