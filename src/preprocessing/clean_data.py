import pandas as pd
import os
import streamlit as st

def load_and_clean_stock(stock_name, data_folder="data/stock-data"):
    import os
    import pandas as pd

    file_path = os.path.join(data_folder, f"{stock_name}.csv")

    # Read multi-level header CSV
    df = pd.read_csv(file_path, header=[0, 1])

    # Flatten multi-level columns → Keep only the first level
    df.columns = [col[0] for col in df.columns]

    # Reset the index to make "Date" a column
    df.rename(columns={"Price": "Date"}, inplace=True)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

    # Remove rows where date failed to parse
    df = df.dropna(subset=["Date"])

    # Remove duplicates and empty rows
    df = df.drop_duplicates()
    df = df.dropna(how="all")

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    return df


    # Store in Streamlit session state
    st.session_state["cleaned_data"] = df

    return df
