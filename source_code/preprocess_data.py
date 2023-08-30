# Import libraries
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def compute_indicators(df):
    df = df.copy()  # Create a copy of the input DataFrame
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Sort the DataFrame's index in ascending order
    df.sort_index(ascending=True, inplace=True)

    # Calculate SMA
    df['SMA'] = df['Close'].rolling(window=14).mean()

    # Calculate EMA
    df['EMA'] = df['Close'].ewm(span=14, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate VWAP
    df['VWAP'] = (df['Volume'] * df['Close']).cumsum() / df['Volume'].cumsum()

    # Calculate Average True Range (ATR)
    df['TR'] = pd.concat([
      abs(df['High'] - df['Low']),
      abs(df['High'] - df['Close'].shift(1)),
      abs(df['Low'] - df['Close'].shift(1))
      ], axis=1).max(axis=1, skipna=False)

    df['ATR'] = df['TR'].rolling(14).mean()

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    df.reset_index()
    return df


#############################################################################################################################################################################################################
# We transform our BTC price and technical indicator data from a daily timeframe to a weekly timeframe and filter it to cover a 10-year period starting from July 1, 2013 to July 1, 2023.
#############################################################################################################################################################################################################

def convert_to_weekly(df, end_date):
    df = df.copy()  # Create a copy of the input DataFrame

    # Define your start and end dates
    end = pd.to_datetime(end_date)
    start = datetime(year=end.year-10, month=end.month, day=end.day)

    # Sort the DataFrame's index in ascending order
    df.sort_index(ascending=True, inplace=True)

    # Filter by date range
    df = df.loc[start:end]

    return df

#############################################################################################################################################################################################################
# We merge multiple datasets into a single dataframe to combine the relevant information for further analysis.
#############################################################################################################################################################################################################

def merge_dataframes_on_column(dataframes, column_name):
    df_merged = dataframes[0]  # Start with the first DataFrame in the list

    # Merge all other DataFrames in the list
    for df in dataframes[1:]:
        df_merged = pd.merge(df_merged, df, on=column_name)

    df_merged = df_merged.set_index(column_name)

    return df_merged

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--input_folder', type=str, default='../data/historical_data/', help='Path to the input folder')
    parser.add_argument('--output_file', type=str, default='../data/preprocessed_data/', help='Path to the output file')
    args = parser.parse_args()
    

    #####################
    # Start the progress
    #####################
    print("Preprocessing the data...")

    # Reading csv data into dataframes
    path = args.input_folder
    btc_onchain = pd.read_csv(path + 'btc_onchain_data.csv')
    financial_data = pd.read_csv(path + 'financial_data.csv')
    google_trends = pd.read_csv(path + 'google_trends.csv')
    btc_ohlcv_daily = pd.read_csv(path + 'btc_ohlcv.csv')

    ## Remove btc price and marketcap from btc_onchain data as we use the data from btc_ohlcv df
    btc_onchain = btc_onchain.drop(['btc_price', 'market_cap'], axis=1)
    btc_onchain = btc_onchain.rename(columns={'has_ate': 'hash_rate'})

    btc_ohlcv_technical_daily = compute_indicators(btc_ohlcv_daily)

    end_date = '2023-07-01'
    btc_ohlcv_technical = convert_to_weekly(btc_ohlcv_technical_daily, end_date)
    btc_ohlcv_technical = btc_ohlcv_technical.reset_index()

    # Define the columns to be removed
    columns_to_remove = ['TR']

    # Check if these columns exist in the dataframe and if so, remove them
    btc_ohlcv_technical = btc_ohlcv_technical.drop(columns=[col for col in columns_to_remove if col in btc_ohlcv_technical.columns])

    # List of dataframes
    dataframes = [btc_onchain, financial_data, btc_ohlcv_technical, google_trends]

    # Convert the 'Date' column to datetime for all dataframes
    for i, df in enumerate(dataframes):
        df['Date'] = pd.to_datetime(df['Date'])
        dataframes[i] = df

    # Merge the dataframes
    df_data = merge_dataframes_on_column(dataframes, 'Date')

    # Print the number of rows and columns in df_data
    print(f"The DataFrame has {df_data.shape[0]} weekly data points (rows) and {df_data.shape[1]} variables (columns).")
    print(df_data.head())
    
    # Create folder if it not exists
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    
    df_data.to_csv(args.output_file + 'preprocessed_data.csv', index=False)

    #####################
    # End the progress
    #####################
    print("Preprocessing the data...Done")
        