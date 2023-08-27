
# Import libraries
import pandas as pd
import yfinance as yf
from datetime import date
from cryptocmd import CmcScraper
from pytrends.request import TrendReq
from coinmetrics.api_client import CoinMetricsClient

import warnings
warnings.filterwarnings("ignore")


def data_extraction():
    end = date(2023, 7, 1)
    start = date(year=end.year-10, month=end.month, day=end.day)

    client = CoinMetricsClient()
    asset_metrics = client.get_asset_metrics(
        assets='btc',
        metrics=['PriceUSD', 'HashRate', 'DiffMean', 'TxCnt', 'CapMrktCurUSD'],
        start_time='2013-06-25T00:00:00Z',
        end_time=end
    )

    #################################
    # Crawl Bitcoin onchain dataset #
    #################################
    btc_onchain = asset_metrics.to_dataframe()
    btc_onchain = btc_onchain.rename(columns={
        'PriceUSD': 'btc_price',
        'HashRate': 'hash_ate',
        'DiffMean': 'block_difficulty',
        'TxCnt': 'transaction_count',
        'CapMrktCurUSD': 'market_cap'
    })

    # Convert the 'timestamp' column to datetime
    btc_onchain['Date'] = pd.to_datetime(btc_onchain['time'])
    btc_onchain.set_index('Date', inplace=True)

    # Drop the 'time' column
    btc_onchain.drop('time', axis=1, inplace=True)
    btc_onchain = btc_onchain.resample('W-MON').mean(numeric_only=True)

    # Convert datetime index to date (removes time component)
    btc_onchain.index = btc_onchain.index.date
    btc_onchain = btc_onchain[(btc_onchain.index >= start) & (btc_onchain.index <= end)]
    btc_onchain.index.name = 'Date'


    #############################################################
    # Crawl Financial data - S&P, Gold, 13W treasury, USD Index #
    #############################################################
    symbols = ['^GSPC','GC=F', '^IRX', 'DX-Y.NYB']
    financial_prices_df = pd.DataFrame()
    for symbol in symbols:
      try:
        data = yf.download(symbol, start=start, end=end, interval='1wk')['Adj Close']
        financial_prices_df[symbol] = data
      except:
        print(f"{symbol} not found on Yahoo Finance")


    # Rename columns for easier interpretation
    financial_prices_df = financial_prices_df.rename(columns={
        '^GSPC': 's&p500',
        'GC=F': 'gold',
        '^IRX': '13w_treasury',
        'DX-Y.NYB': 'usd_index'
    })

    #######################
    # Crawl Google Trends #
    #######################
    pytrends = TrendReq(hl='en-US', tz=360)
    keywords = ["Bitcoin", "BTC"]

    # Initialize a dataframe to store the results
    google_trends = pd.DataFrame()

    # Iterate over each year and fetch the weekly data
    for year in range(2013, 2024):
        pytrends.build_payload(
            kw_list=keywords,
            cat=0,
            timeframe=f'{year}-01-01 {year}-12-31'
        )
        data = pytrends.interest_over_time()
        google_trends = pd.concat([google_trends, data])

    google_trends = google_trends.resample('W-MON').sum()

    # Combine the 'Bitcoin' and 'BTC' columns
    google_trends['google_trends'] = google_trends['Bitcoin'] + google_trends['BTC']

    # Normalize the 'google_trends_BTC' column to a 0-100 scale as it provides normalized data
    google_trends['google_trends'] = ((google_trends['google_trends'] - google_trends['google_trends'].min()) /
                                          (google_trends['google_trends'].max() - google_trends['google_trends'].min())) * 100

    # Drop the original 'Bitcoin', 'BTC', and 'isPartial' columns
    google_trends.drop(columns=['Bitcoin', 'BTC', 'isPartial'], inplace=True)

    # Convert the index to datetime, filter, and then convert to date
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    google_trends.index = pd.to_datetime(google_trends.index)
    google_trends = google_trends[(google_trends.index >= start_date) & (google_trends.index <= end_date)]

    google_trends.index = google_trends.index.date
    google_trends.index.name = 'Date'


    #############################################################
    # Crawl BTC Open, high close, volume, marketCap data #
    #############################################################
    scraper = CmcScraper('BTC', '01-01-2013', '01-07-2023')
    # Transform collected data into a dataframe
    btc_ohlcv = scraper.get_dataframe()

    return btc_onchain, financial_prices_df, google_trends, btc_ohlcv

if __name__ == '__main__':
    btc_onchain, financial_prices_df, google_trends, btc_ohlcv = data_extraction()
    print(btc_onchain.head())
    print(financial_prices_df.head())
    print(google_trends.head())
    print(btc_ohlcv.head())
    btc_onchain.to_csv('./historical_data/btc_onchain_data.csv')
    financial_prices_df.to_csv('./historical_data/financial_data.csv')
    google_trends.to_csv('./historical_data/google_trends.csv')
    btc_ohlcv.to_csv('./historical_data/btc_ohlcv.csv')
