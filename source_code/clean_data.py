# Import libraries
import ta
import os
import copy
import hmms
import quandl

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
import pandas_datareader.data as web

from scipy import stats
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, ListedColormap, BoundaryNorm
from coinmetrics.api_client import CoinMetricsClient
from cryptocmd import CmcScraper


import pylab as plb
import networkx as nx
from tqdm import tqdm
from pytrends.request import TrendReq
from pgmpy.models import BayesianNetwork
from pgmpy.models import MarkovNetwork, BayesianModel
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import BDeuScore, BDsScore, BicScore, HillClimbSearch, K2Score, BayesianEstimator

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#############################################################################################################################################################################################################
# Bad data: we detect and replace incorrect data, such as negative or infinite asset values, with NaN values to maintain the accuracy of the dataset.
#############################################################################################################################################################################################################

def handle_bad_data(df):
    """
    Handle nan/null value from the dataset

    Parameters:
    df (pandas dataframe): The input dataframe.

    Returns:
    A cleaned dataframe with nan replaced by empty.
    """
    df[df <= 0] = np.nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace('', np.nan, inplace=True)

    return df


#############################################################################################################################################################################################################
# Missing data: we address any gaps in the data by using the ffill (forward fill) and bfill (backward fill) methods to fill in missing values.
#############################################################################################################################################################################################################

def handle_missing_data(prices: pd.DataFrame):
    """
    Handle missing value

    Parameters:
    df (pandas dataframe): The input dataframe.

    Returns:
    A cleaned dataframe with missing value filled using forward fill
    """
    # Forward fill the holes, by filling them with the data behind.
    prices_ffill = prices.fillna(method='ffill', inplace=False)
    prices_filled = prices_ffill.fillna(method='bfill', inplace=False)

    return prices_filled

#############################################################################################################################################################################################################
# Outliers: we use the z-score technique to identify and appropriately manage any extreme or unusual data points.
#############################################################################################################################################################################################################

def detect_outliers_zscore(df, threshold=3):
    """
    Detects and removes outliers from a pandas dataframe using z-score approach.

    Parameters:
    df (pandas dataframe): The input dataframe.
    threshold (float): The z-score threshold above which a data point is considered an outlier. Default is 3.

    Returns:
    A cleaned dataframe with outliers removed.
    """
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    cleaned_data = df[filtered_entries]

    return cleaned_data

if __name__ == '__main__':
    df_after_bad_data = handle_bad_data(df_data)
    df_after_missing_data = handle_missing_data(df_after_bad_data)
    data = detect_outliers_zscore(df_after_missing_data)
    data.index = pd.to_datetime(data.index)
    data['forecast'] = data['Close'].shift(-1)

    # Remove these variables and initial rows with 0 Volume
    data.drop(['block_size', 'block_difficulty'], axis=1,inplace=True)
    data = data[data['Volume'] != 0]

    # Train, Validation and Test split
    train_data = data[: int(data.shape[0] * 0.80)]
    vald_data = data[int(0.80 * data.shape[0]) : int(0.90 * data.shape[0])]
    test_data = data[int(0.90* data.shape[0]) : int(data.shape[0])]

    # Save the data
    train_data.to_csv('train_data.csv')
    vald_data.to_csv('vald_data.csv')
    test_data.to_csv('test_data.csv')
