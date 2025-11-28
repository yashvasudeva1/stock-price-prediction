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
    predict_next_n_days
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

    st.write("Select a stock from the sidebar and load the data.")

    if st.button("Load & Clean Data"):
        if selected_stock:
            df = load_and_clean_stock(selected_stock)
            df = add_essential_columns(df)
            st.session_state["cleaned_df"] = df
            st.success("Data Loaded Successfully")
            st.dataframe(st.session_state["cleaned_df"])


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

    # ===========================================
    # 1. DATASET SUMMARY
    # ===========================================
    st.subheader("📊 1. Dataset Summary")

    summary = dataset_summary(df)

    # # Top metrics
    # col1, col2, col3 = st.columns(3)
    # col1.metric("Total Rows", summary["total_rows"])
    
    # col2.write(f"**Start Date:** {summary['start_date']}")
    # col3.write(f"**End Date:** {summary['end_date']}")




    # Columns list
    with st.expander("🧱 Columns in Dataset"):
        st.write(", ".join(summary["columns"]))

    # Missing values
    st.subheader("📌 Missing Values")
    missing_df = pd.DataFrame.from_dict(
        summary["missing_values"], 
        orient="index", 
        columns=["Missing"]
    )
    st.table(missing_df)

    # ===========================================
    # 2. SUMMARY STATS
    # ===========================================
    st.subheader("📈 2. Summary Statistics")
    st.dataframe(summary_stats(df))

    # ===========================================
    # 3. TRADING DAY INFO
    # ===========================================
    st.subheader("📅 3. Trading Day Info")
    td_df = pd.DataFrame([trading_day_info(df)])
    st.dataframe(td_df)
    # ===========================================
    # 4. DAILY RETURN STATS
    # ===========================================
    st.subheader("📉 4. Daily Return Stats")
    st.json(daily_return_stats(df), expanded=True)

    # ===========================================
    # 5. OUTLIERS
    # ===========================================
    st.subheader("⚠️ 5. Outliers in Close Price")
    outliers = detect_outliers(df)
    if len(outliers) == 0:
        st.info("No outliers detected.")
    else:
        st.dataframe(outliers)

    # ===========================================
    # 6. CORRELATION MATRIX
    # ===========================================
    st.subheader("🔗 6. Correlation Matrix")
    st.dataframe(correlation_matrix(df))

    # ===========================================
    # 7. TREND STREAKS
    # ===========================================
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
# TRAIN ANN PAGE
# =========================================================
elif page == "Train ANN":
    st.title("🤖 Train ANN Model")

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

    window_size = st.slider("Window Size (days)", 5, 60, 30)

    if st.button("Train Model"):
        with st.spinner("Training ANN..."):

            # Create dataset
            X, y = create_windowed_dataset(df, feature_cols, target_col="Target_Close", window_size=window_size)

            # Scaling
            scaler = ScalerWrapper()
            Xs, ys = scaler.fit_transform(X, y)

            # Build model
            model = build_regression_ann(input_shape=(window_size, len(feature_cols)))

            history = model.fit(
                Xs, ys,
                validation_split=0.2,
                epochs=20,
                batch_size=32,
                verbose=0
            )

            st.success("Model trained successfully!")

            # Save to session state
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["history"] = history
            st.session_state["feature_cols"] = feature_cols
            st.session_state["window_size"] = window_size

        # Show training curves
        st.subheader("Training Performance")
        st.line_chart({"loss": history.history["loss"], "val_loss": history.history["val_loss"]})



# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "Predictions":
    st.title("📈 Predict Future Stock Prices")

    df = st.session_state["cleaned_df"]
    model = st.session_state.get("model", None)
    scaler = st.session_state.get("scaler", None)

    if df is None or model is None or scaler is None:
        st.warning("⚠ Train the ANN model first.")
        st.stop()

    n_days = st.slider("How many future days to predict?", 1, 30, 7)

    from src.model.ann import predict_next_n_days

    if st.button("Predict"):

        pred_df = predict_next_n_days(
            model=model,
            scaler=scaler,
            df=df,
            window_size=st.session_state["window_size"],
            n_days=n_days
        )

        st.session_state["pred_df"] = pred_df

        st.success("Prediction complete!")
        st.dataframe(pred_df)

        # Plot Prediction Chart
        st.subheader("Prediction Plot")
        fig_pred = plot_pred_vs_actual(
            pred_df.rename(columns={"Predicted_Close": "y_pred"})
        )
        st.plotly_chart(fig_pred, use_container_width=True)
