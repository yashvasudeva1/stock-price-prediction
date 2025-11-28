import pandas as pd
import os
import streamlit as st
def load_and_clean_stock(stock_name, data_folder="data/stock-data"):

    file_path = os.path.join(data_folder, f"{stock_name}.csv")

    df = pd.read_csv(file_path, header=[0, 1])

    # Flatten column levels
    df.columns = [col[0] for col in df.columns]

    # Move index to a column (Fix for your bug)
    df = df.reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)

    # Convert Date safely
    df["Date"] = pd.to_datetime(
        df["Date"], 
        format="%d-%m-%Y", 
        errors="coerce"
    )

    # Drop invalid dates
    df = df.dropna(subset=["Date"])

    # Sort by date
    df = df.sort_values("Date")

    df = df.set_index("Date")

    return df

