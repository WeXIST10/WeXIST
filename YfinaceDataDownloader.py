import numpy as np
import pandas as pd
import yfinance as yf
import ta  # Make sure 'ta' is installed

pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)


def preprocessData(ticker, start_date, end_date):
    """Main function to process data, add indicators, and return the required columns."""

    # Step 1: Fetch and preprocess data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker symbol and date range.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]  # Flatten the MultiIndex

    # Remove ticker suffix (e.g., '_aapl') and standardize column names
    df.columns = [col.split('_')[0] for col in df.columns]
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Preprocess: Drop NaN, fill missing values, and convert index to datetime
    df.dropna(how="all", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Step 2: Add technical indicators
    if 'close' not in df.columns:
        raise KeyError(f"Expected 'close' column not found. Available columns: {df.columns}")

    # Simple Moving Averages (SMA)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()

    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Step 3: Remove the first 200 rows
    df_without_first_200 = df.iloc[200:]

    # Step 4: Return column names from 6th column to last column as an array
    column_names_array = df_without_first_200.columns[5:].to_numpy()

    # Output all data, column names, and the resulting DataFrame
    print("Column names after preprocessing:", df.columns)
    print("Final DataFrame after removing first 200 rows:")
    print(df_without_first_200.head())  # Display first few rows after removal


    return df_without_first_200


def technicalIndicators(df):
    column_names_array = df.columns[5:].to_numpy()
    return column_names_array

# Example usage:
ticker = "AAPL"
start_date = "2018-01-01"
end_date = "2023-01-01"
final_df = preprocessData(ticker, start_date, end_date)
print(technicalIndicators(final_df))

# Now `final_df` contains the processed DataFrame, and `columns_array` contains the list of columns
