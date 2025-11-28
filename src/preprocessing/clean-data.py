import pandas as pd
import os
import streamlit as st

def load_and_clean_stock(stock_name, data_folder="data"):
    file_path = os.path.join(data_folder, f"{stock_name}.csv")

    # Read the multi-level CSV (your format)
    df = pd.read_csv(file_path, header=[0, 1])

    # Flatten multi-level columns → keep only first level (Price, Close, High, ...)
    df.columns = [col[0] for col in df.columns]

    # Fix the Date index row
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

    # Convert Date column to proper datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # Clean the data
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Store in Streamlit session state
    st.session_state["cleaned_data"] = df

    return df
