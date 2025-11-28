import pandas as pd
import os

def load_and_clean_stock(stock_name, data_folder="data/stock-data"):

    file_path = os.path.join(data_folder, f"{stock_name}.csv")

    df = pd.read_csv(file_path, header=[0, 1])

    # Flatten first header level
    df.columns = [col[0] for col in df.columns]

    # The first row contains the second header ("Date", 0, 0...) → REMOVE IT
    # It is ALWAYS the first row because of your CSV structure.
    df = df.iloc[1:].reset_index(drop=True)

    # Fix the Date column
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Convert date safely (auto-detect format)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop invalid dates
    df = df.dropna(subset=["Date"])

    # Set date index
    df = df.set_index("Date")

    # Convert numeric columns properly
    df = df.apply(pd.to_numeric, errors="coerce")

    return df

