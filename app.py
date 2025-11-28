import streamlit as st
import os
import pandas as pd

# ---------------------------
# IMPORTING YOUR MODULES
# ---------------------------
from src/preprocessing/clean_data.py import load_and_clean_stock
from src/eda/analytics/create_essential_columns.py import add_essential_columns
from src/eda/eda import (
    dataset_summary,
    summary_stats,
    trading_day_info,
    daily_return_stats,
    detect_outliers,
    correlation_matrix,
    rolling_stats,
    volume_spikes,
    trend_streaks
)
from src/visualisations/create_visualisations.py import (
    plot_close_price,
    plot_candlestick,
    plot_volume,
    plot_moving_averages,
    plot_rsi,
    plot_macd,
    plot_pred_vs_actual
)
from src/model/ann.py import (
    build_regression_ann,
    create_windowed_dataset,
    ScalerWrapper
)

# -----------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Stock Analyzer + ANN", layout="wide")

# -----------------------------------
# SESSION STATE INIT
# -----------------------------------
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "history" not in st.session_state:
    st.session_state["history"] = None
if "pred_df" not in st.session_state:
    st.session_state["pred_df"] = None

# -----------------------------------
# SIDEBAR SETTINGS
# -----------------------------------
st.sidebar.header("⚙ Settings")

data_folder = "data"

if not os.path.exists(data_folder):
    st.sidebar.error("❌ 'data/' folder not found.")
    st.stop()

csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
tickers = [os.path.splitext(f)[0] for f in csv_files]

selected_stock = st.sidebar.selectbox("Select Stock", tickers)

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Preview", "EDA", "Visualisations", "Train ANN", "Predictions"]
)

# =========================================================
# HOME PAGE
# =========================================================
if page == "Home":
    st.title("📊 Stock Analyzer & ANN Prediction")

    st.write("Select a stock from the sidebar and load the data.")

    if st.button("Load & Clean Data"):
        df = load_and_clean_stock(selected_stock, data_folder)
        df = add_essential_columns(df)

        st.session_state["cleaned_df"] = df
        st.success(f"Loaded and cleaned: {selected_stock}")

# =========================================================
# DATA PREVIEW
# =========================================================
elif page == "Data Preview":
    st.title(f"📄 Data Preview — {selected_stock}")

    df = st.session_state["cleaned_df"]

    if df is None:
        st.warning("⚠ Load data from the Home page first.")
    else:
        st.subheader("Head")
        st.dataframe(df.head(50))

        st.subheader("Tail")
        st.dataframe(df.tail(50))

# =========================================================
# EDA PAGE
# =========================================================
elif page == "EDA":
    st.title(f"🔍 EDA — {selected_stock}")

    df = st.session_state["cleaned_df"]

    if df is None:
        st.warning("⚠ Please load data first.")
        st.stop()

    st.subheader("1. Dataset Summary")
    st.json(dataset_summary(df))

    st.subheader("2. Summary Stats")
    st.dataframe(summary_stats(df))

    st.subheader("3. Trading Day Info")
    st.json(trading_day_info(df))

    st.subheader("4. Daily Return Stats")
    st.json(daily_return_stats(df))

    st.subheader("5. Outliers (Close)")
    st.dataframe(detect_outliers(df))

    st.subheader("6. Correlation Matrix")
    st.dataframe(correlation_matrix(df))

    st.subheader("7. Trend Streaks")
    st.json(trend_streaks(df))

# =========================================================
# VISUALISATIONS
# =========================================================
elif page == "Visualisations":
    st.title(f"📈 Visualisations — {selected_stock}")

    df = st.session_state["cleaned_df"]

    if df is None:
        st.warning("⚠ Load data from Home first.")
        st.stop()

    chart = st.selectbox(
        "Select Chart",
        ["Close Price", "Candlestick", "Volume", "Moving Averages", "RSI", "MACD"]
    )

    if chart == "Close Price":
        st.plotly_chart(plot_close_price(df), use_container_width=True)
    elif chart == "Candlestick":
        st.plotly_chart(plot_candlestick(df), use_container_width=True)
    elif chart == "Volume":
        st.plotly_chart(plot_volume(df), use_container_width=True)
    elif chart == "Moving Averages":
        st.plotly_chart(plot_moving_averages(df), use_container_width=True)
    elif chart == "RSI":
        st.plotly_chart(plot_rsi(df), use_container_width=True)
    elif chart == "MACD":
        st.plotly_chart(plot_macd(df), use_container_width=True)

# =========================================================
# TRAIN ANN PAGE
# =========================================================
elif page == "Train ANN":
    st.title("🤖 Train ANN Model")

    df = st.session_state["cleaned_df"]
    if df is None:
        st.warning("⚠ Load data first.")
        st.stop()

    st.subheader("Training Configuration")

    window_size = st.number_input("Window Size", min_value=5, max_value=200, value=20)
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=30)
    batch_size = st.number_input("Batch Size", min_value=8, max_value=512, value=32)
    test_split = st.slider("Test Split (%)", 5, 50, 20)

    feature_cols = [c for c in df.columns if c not in ["Date", "Target_Close"]]

    if st.button("Train Model"):
        X, y = create_windowed_dataset(df, feature_cols, "Target_Close", window_size)

        split = int(len(X) * (1 - test_split / 100))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = ScalerWrapper()
        X_train_s, y_train_s = scaler.fit_transform(X_train, y_train)
        X_test_s, y_test_s = scaler.transform(X_test, y_test)

        model = build_regression_ann(X_train_s.shape[1:])

        with st.spinner("Training ANN..."):
            history = model.fit(
                X_train_s, y_train_s,
                validation_data=(X_test_s, y_test_s),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

        st.success("Training Complete!")

        st.session_state["model"] = model
        s
