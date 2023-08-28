# Import libraries
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import  ListedColormap, BoundaryNorm

import warnings
warnings.filterwarnings("ignore")

def plot_regime_switch(data, states, data_type):
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    for series_id in data.columns:
        df = pd.DataFrame(index=data[1:].index)
        df[series_id] = data[series_id][1:]
        df['Diff'] = data[series_id].diff()[1:]
        # index_diff = df.index.difference(states.index)
        df['Regime'] = states[series_id]

        # Get means of all assigned states
        means = df.groupby(['Regime'])['Diff'].mean()
        lst_1 = means.index.tolist()
        lst_2 = means.sort_values().index.tolist()
        map_regimes = dict(zip(lst_2, lst_1))
        df['Regime'] = df['Regime'].map(map_regimes)

        cmap = ListedColormap(['r','b','g'],'indexed')
        norm = BoundaryNorm(range(3 + 1), cmap.N)
        inxval = mdates.date2num(df[series_id].index)
        points = np.array([inxval, df[series_id]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(df['Regime'])
        plt.gca().add_collection(lc)
        plt.xlim(df[series_id].index.min(), df[series_id].index.max())
        plt.ylim(df[series_id].min(), df[series_id].max())
        r_patch = mpatches.Patch(color='red', label='Bear')
        g_patch = mpatches.Patch(color='green', label='Bull')
        b_patch = mpatches.Patch(color='blue', label='Stagnant')
        plt.legend(handles=[r_patch, g_patch, b_patch])
        # Create the folder if not exists
        if not os.path.exists(f"./plots/{data_type}"):
            os.makedirs(f"./plots/{data_type}")
        name = f"./plots/{data_type}/{series_id.replace('.', '_')}_{data_type}.png"
        plt.savefig(name)
        plt.close()

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Clean the data')
    parser.add_argument('--data_version', type=str, default='train_data.csv', help='Path to the data file')
    args = parser.parse_args()

    # Load the data and states
    if args.data_version == 'train_data.csv':
        train = pd.read_csv("../data/cleaned_data/train_data.csv", index_col=0)
        state = pd.read_csv("../data/hmm_data/train_data.csv", index_col=0)
        train.index = pd.to_datetime(train.index)
        state.index = pd.to_datetime(state.index)
        plot_regime_switch(train, state, 'train')
    elif args.data_version == 'validation_data.csv':
        val = pd.read_csv("../data/cleaned_data/validation_data.csv", index_col=0)
        state = pd.read_csv("../data/hmm_data/validation_data.csv", index_col=0)
        val.index = pd.to_datetime(val.index)
        state.index = pd.to_datetime(state.index)
        plot_regime_switch(val, state, 'validation')
    elif args.data_version == 'test_data.csv':
        test = pd.read_csv("../data/cleaned_data/test_data.csv", index_col=0)
        state = pd.read_csv("../data/hmm_data/test_data.csv", index_col=0)
        test.index = pd.to_datetime(test.index)
        state.index = pd.to_datetime(state.index)
        plot_regime_switch(test, state, 'test')
    else:
        raise ValueError('Wrong data version. Choose from train_data.csv, validation_data.csv, and test_data.csv')
