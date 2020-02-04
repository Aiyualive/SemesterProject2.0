import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_defects(d_df, peak_height=50, prominence=None, threshold=None, savepath='./AiyuDocs/imgs/'):
    """
    Plots each defect and saves them to a figure
    params:
        d_df:
        peak_height:

    """
    # filtering?
    """
    if defect_filter != None:
        d_df = d_df.filter(regex=defect_filter)
    if ax_filt != None:
        d_df = d_df.filter(regex=ax_filt)
    """

    defect_types = np.unique(d_df['defect_type'])

    # get ybot, ytop
    """
    def_type_indexes = {name: np.where(de.df.columns.str.contains(name))[0] for name in defect_types}
    ylim = {}
    for k, v in def_type_indexes.items():
        accs = np.asarray([i[1] for i in d_df.iloc[0, v].values]).flatten()
        ylim[k] = (min(accs), max(accs))"""

    feature_names = d_df.columns
    display_params = feature_names[3:-1] # skip first 3 columns
    rows, cols = d_df.shape
    for i, row in d_df.iterrows():

        #ybot, ytop = ylim[d_df.loc["def_type", i]]

        caption = ',    '.join([ f"|{feature}|: {d_df.loc[i, feature]}"  for feature in display_params])

        x, y = (row['timestamps'], row['accelerations'])

        g_max = np.argmax(y)

        pos_peaks, _ = find_peaks( y, height=peak_height, prominence=prominence, threshold=threshold)
        neg_peaks, _ = find_peaks(-y, height=peak_height, prominence=prominence, threshold=threshold)

        fig = plt.figure(figsize=(20,5), constrained_layout=True)
        ax1 = fig.add_axes((0.1, 0.2, 1, 1))
        ax1.set_title(i)
        ax1.scatter(x[pos_peaks], y[pos_peaks], marker="x", color="red", zorder=2)
        ax1.scatter(x[neg_peaks], y[neg_peaks], marker="^", color="green", zorder=2)
        ax1.plot(x, np.zeros(len(x)) + peak_height, "--", color="black")
        ax1.plot(x, np.zeros(len(x)) - peak_height, "--", color="black")

        #ax1.scatter(x[g_max], y[g_max], marker='x', s=1000, color="black")
        ax1.plot(x, y, zorder = 1)
        ax1.set_xlabel("Nanoseconds")
        ax1.set_ylabel("Amplitude")
        #ax1.set_ylim(ybot,ytop)
        ax1.margins(x=0, y=0.1)

        fig.text(0.1, .0005, caption, ha='left', fontsize=12)
        fig.set_size_inches(20, 5, forward=True)
        fig.savefig(savepath + i,  bbox_inches='tight')
        plt.show()
