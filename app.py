import streamlit as st
import os
import pandas as pd

# ---------------------------
# IMPORTING YOUR MODULES
# ---------------------------
from src.preprocessing.clean_data import load_and_clean_stock
from src.eda.analytics.create_essential_columns import add_essential_columns
from src.eda.analytics.eda import (
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
from src.eda.visualisations.create_visualisations import (
    plot_close_price,
    plot_candlestick,
    plot_volume,
    plot_moving_averages,
    plot_rsi,
    plot_macd,
    plot_pred_vs_actual
)
from src.model.ann import (
    MultiScaler,
    preprocess_data,
    build_multistep_ann,
    train_model,
    predict_future
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

data_folder = "data/stock-data"

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
    st.title("Stock Analyzer & ANN Prediction")

    if st.button("Load & Clean Data"):
        if selected_stock:
            df = load_and_clean_stock(selected_stock)
            df = add_essential_columns(df)

            st.session_state["cleaned_df"] = df
            st.success("Data Loaded Successfully!")
            st.dataframe(df)

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

    st.subheader("📊 1. Dataset Summary")
    summary = dataset_summary(df)

    with st.expander("🧱 Columns in Dataset"):
        st.write(", ".join(summary["columns"]))

    st.subheader("📌 Missing Values")
    missing_df = pd.DataFrame.from_dict(
        summary["missing_values"], orient="index", columns=["Missing"]
    )
    st.table(missing_df)

    st.subheader("📈 2. Summary Statistics")
    st.dataframe(summary_stats(df))

    st.subheader("📅 3. Trading Day Info")
    td_df = pd.DataFrame([trading_day_info(df)])
    st.dataframe(td_df)

    st.subheader("📉 4. Daily Return Stats")
    st.json(daily_return_stats(df), expanded=True)

    st.subheader("⚠️ 5. Outliers in Close Price")
    outliers = detect_outliers(df)
    st.dataframe(outliers if len(outliers) else pd.DataFrame({"Message": ["No outliers detected"]}))

    st.subheader("🔗 6. Correlation Matrix")
    st.dataframe(correlation_matrix(df))

    st.subheader("📊 7. Trend Streaks")
    st.json(trend_streaks(df), expanded=True)

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
# TRAIN ANN PAGE (MULTI-STEP)
# =========================================================
elif page == "Train ANN":
    st.title("🤖 Train Multi-Step ANN Model")

    df = st.session_state["cleaned_df"]
    if df is None:
        st.warning("⚠ Load and clean the data first.")
        st.stop()

    st.subheader("Select Feature Columns")

    default_features = ["Close", "SMA_5", "SMA_10", "SMA_20", "RSI_14", "MACD", "MACD_SIGNAL"]

    feature_cols = st.multiselect(
        "Choose features to train on:",
        options=df.columns.tolist(),
        default=[col for col in default_features if col in df.columns]
    )

    if len(feature_cols) == 0:
        st.warning("⚠ Select at least one feature.")
        st.stop()

    target_col = "Close"

    window_size = st.slider("Window Size (days)", 20, 100, 60)
    forecast_days = st.slider("Forecast horizon (future days)", 1, 30, 7)

    if st.button("Train Multi-Step ANN"):
        with st.spinner("Training ANN..."):

            from src.model.ann import (
                MultiScaler,
                preprocess_data,
                build_multistep_ann,
                train_model
            )

            # Create Scaler
            scaler = MultiScaler()

            # Create dataset
            X, y = preprocess_data(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                window_size=window_size,
                forecast_days=forecast_days,
                scaler=scaler
            )

            # Build ANN
            input_size = X.shape[1]
            model = build_multistep_ann(
                input_size=input_size,
                output_steps=forecast_days
            )

            # Train
            history = train_model(model, X, y)

            # Save for Prediction page
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["feature_cols"] = feature_cols
            st.session_state["window_size"] = window_size
            st.session_state["forecast_days"] = forecast_days

            st.success("🎉 Model trained successfully!")

        # Training Curves
        st.subheader("📉 Training Performance")
        st.line_chart({
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"]
        })



# =========================================================
# PREDICTION PAGE (MULTI-STEP)
# =========================================================
elif page == "Predictions":
    st.title("📈 Predict Future Stock Prices")

    df = st.session_state["cleaned_df"]
    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    feature_cols = st.session_state.get("feature_cols")
    window_size = st.session_state.get("window_size")
    forecast_days = st.session_state.get("forecast_days")

    required_items = [df, model, scaler, feature_cols, window_size, forecast_days]

    if any(item is None for item in required_items):
        st.warning("⚠ Train the ANN model first.")
        st.stop()


    st.write(f"Model will predict **{forecast_days} future days** at once.")

    from src.model.ann import predict_future

    if st.button("Predict Future Prices"):
        pred_df = predict_future(
            model=model,
            scaler=scaler,
            df=df,
            feature_cols=feature_cols,
            window_size=window_size,
            forecast_days=forecast_days
        )

        st.session_state["pred_df"] = pred_df

        st.success("Prediction complete!")
        st.dataframe(pred_df)

        # Visualise
        st.subheader("Prediction Plot")
        fig_pred = plot_pred_vs_actual(pred_df)        

        st.plotly_chart(fig_pred, use_container_width=True)
