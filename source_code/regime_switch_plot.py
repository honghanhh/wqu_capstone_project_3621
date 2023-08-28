
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
    name = f"./plots/{series_id.replace('.', '_')}_{'TRAIN'}.png"
    plt.savefig(name)
    plt.close()


states = pd.read_csv("./data/train_data.csv", index_col=0)
states.index = pd.to_datetime(states.index)
plot_regime_switch(train_data, states, 'TRAIN')
