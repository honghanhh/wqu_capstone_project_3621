# Import libraries
import os
import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import logging
logging.basicConfig(filename="training.log", level=logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
warnings.filterwarnings("ignore")

def trading_strategy(test_sheet, model_type, forecast_col):
    # Check the model type
    if model_type not in ['bayesian', 'markov']:
        raise ValueError("model_type must be either 'bayesian' or 'markov'")

    test_sheet = test_sheet.copy()

    # Initialize columns for holding data and a counter for trades
    test_sheet[f'{model_type}_holdings'] = 0.0
    test_sheet[f'{model_type}_cash'] = 0.0
    test_sheet[f'{model_type}_total'] = 0.0
    num_trades = 0

    # Initialize cash and holdings
    initial_cash = 0.0  # Start with no cash
    cash = initial_cash
    holdings = 1.0  # Start with 1 unit of the asset

    # Update the first row
    test_sheet[f'{model_type}_holdings'].iloc[0] = holdings * test_sheet['Close'].iloc[0]
    test_sheet[f'{model_type}_cash'].iloc[0] = cash
    test_sheet[f'{model_type}_total'].iloc[0] = test_sheet[f'{model_type}_holdings'].iloc[0] + test_sheet[f'{model_type}_cash'].iloc[0]

    # Loop through each row starting from the second row
    for i in range(1, len(test_sheet)):
        # Sell
        if test_sheet[forecast_col].iloc[i-1] == 0 and holdings > 0:  # Signal to sell
            cash += holdings * test_sheet['Close'].iloc[i]
            holdings = 0.0
            num_trades += 1  # Increase the trade counter

        # Buy
        elif test_sheet[forecast_col].iloc[i-1] == 2 and cash > 0:  # Signal to buy
            holdings += cash / test_sheet['Close'].iloc[i]
            cash = 0.0
            num_trades += 1  # Increase the trade counter

        # Update dataframe
        test_sheet[f'{model_type}_holdings'].iloc[i] = holdings * test_sheet['Close'].iloc[i]
        test_sheet[f'{model_type}_cash'].iloc[i] = cash
        test_sheet[f'{model_type}_total'].iloc[i] = test_sheet[f'{model_type}_holdings'].iloc[i] + test_sheet[f'{model_type}_cash'].iloc[i]

    return test_sheet, num_trades


def get_stats(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculate relevant financial statistics for a given return series.

    Args:
        returns (pd.Series): The return series to analyze.
        risk_free_rate (float): The risk-free rate for Sharpe and Sortino ratios. Default is 0.0.

    Returns:
        dict: A dictionary containing computed financial statistics.
    """
    stats = {}

    # Total Returns
    stats['total returns'] = (np.exp(returns.sum()) - 1) * 100

    # Mean Annual Returns
    stats['annual returns'] = (np.exp(returns.mean() * 252) - 1) * 100

    # Annual Volatility
    stats['annual volatility'] = returns.std() * np.sqrt(252) * 100

    # Sharpe Ratio
    stats['sharpe ratio'] = ((stats['annual returns'] / 100) - risk_free_rate) / (stats['annual volatility'] / 100)

    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns)/running_max - 1
    stats['max drawdown'] = drawdown.min() * 100

    # Max Drawdown Duration
    underwater = drawdown < 0
    underwater_periods = underwater.cumsum() - np.maximum.accumulate(underwater * (1 - underwater).cumsum())
    stats['max drawdown duration'] = underwater_periods.max()

    return stats

if __name__ == '__main__':

    #####################
    # Start the progress
    #####################
    print("Evaluating...")
    
    ### Bayesian Model ###
    results_df_bayesian = pd.read_csv("../data/bayesian_results.csv", index_col=0)
    prediction_test_bayesian = results_df_bayesian['forecast'].to_numpy()
    real_values_bayesian = results_df_bayesian['close'].to_list()
    
    ### Hidden Markov Model ###
    results_df_markov = pd.read_csv("../data/markov_results.csv", index_col=0)
    prediction_test_markov = results_df_markov['forecast'].to_numpy()
    real_values_markov = results_df_markov['close'].to_list()

    # Calculate error
    ### Bayesian Model ###
    print(f'**********Bayesian Network Model**********')
    print(f'Bayesian real values: \n {real_values_bayesian}')
    print(f'Bayesian predicted values: \n {prediction_test_bayesian}')
    error_test_bayesian = calculate_error(prediction_test_bayesian, real_values_bayesian)
    print(f'Bayesian MAE: \n {error_test_bayesian}')

    ### Hidden Markov Model ###
    print(f'**********Markov Network Model**********')
    print(f'Markov real values: \n {real_values_bayesian}')
    print(f'Markov predicted values: \n {prediction_test_markov}')
    error_test_markov = calculate_error(prediction_test_markov, real_values_markov)
    print(f'Markov MAE: \n {error_test_markov}')

    test_data = pd.read_csv('../data/cleaned_data/test_data.csv', index_col='Date')
    test_sheet = pd.DataFrame(index=test_data.index[1:])  # exclude the first row
    test_sheet['Close'] = test_data['Close'].iloc[1:].to_numpy()

    # Add separate 'forecast' columns for Bayesian and Markov predictions
    test_sheet['bayesian_forecast'] = prediction_test_bayesian
    test_sheet['markov_forecast'] = prediction_test_markov

    # Perform trading strategy with Bayesian model predictions
    test_sheet, num_trades_bayesian = trading_strategy(test_sheet, 'bayesian', 'bayesian_forecast')
    print(f"Number of trades with Bayesian Model: {num_trades_bayesian}")

    # Perform trading strategy with Markov model predictions
    test_sheet, num_trades_markov = trading_strategy(test_sheet, 'markov', 'markov_forecast')
    print(f"Number of trades with Markov Model: {num_trades_markov}")

    plt.plot(test_sheet.index, test_sheet['Close'], 'r')
    plt.plot(test_sheet.index, test_sheet['bayesian_total'], 'g')
    plt.plot(test_sheet.index, test_sheet['markov_total'], 'b')

    r_patch = mpatches.Patch(color='red', label='BTC Price')
    g_patch = mpatches.Patch(color='green', label='Bayesian Model')
    b_patch = mpatches.Patch(color='blue', label='Markov Model')

    plt.legend(handles=[r_patch, g_patch, b_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
    # rotate the x-axis label to 45 degree
    plt.xticks(rotation=45)
    # plt.show()

    # Create folder if not exists
    if not os.path.exists('./plots/comparison'):
        os.makedirs('./plots/comparison')

    # Save the plot
    plt.savefig('./plots/comparison/comparison.png')

    # Calculate returns
    test_sheet['actual_returns'] = np.log(test_sheet['Close'] / test_sheet['Close'].shift(1))
    test_sheet['returns_bayesian'] = np.log(test_sheet['bayesian_total'] / test_sheet['bayesian_total'].shift(1))
    test_sheet['returns_markov'] = np.log(test_sheet['markov_total'] / test_sheet['markov_total'].shift(1))

    # Calculate metrics
    actual_stats = get_stats(test_sheet['actual_returns'].dropna())
    predicted_bayesian_stats = get_stats(test_sheet['returns_bayesian'].dropna())
    predicted_markov_stats = get_stats(test_sheet['returns_markov'].dropna())

    # Convert the dictionaries to dataframes
    actual_stats_df = pd.DataFrame.from_dict(actual_stats, orient='index', columns=['Actual'])
    predicted_bayesian_stats_df = pd.DataFrame.from_dict(predicted_bayesian_stats, orient='index', columns=['Bayesian'])
    predicted_markov_stats_df = pd.DataFrame.from_dict(predicted_markov_stats, orient='index', columns=['Markov'])

    # Concatenate the dataframes side by side
    stats_table = pd.concat([actual_stats_df, predicted_bayesian_stats_df, predicted_markov_stats_df], axis=1)
    stats_table_rounded = stats_table.round(2)
    print(stats_table_rounded)

    #####################
    # End the progress
    #####################
    print("Evaluating...Done")
