import numpy as np
import matplotlib.pyplot as plt

def plot_entity(df, threshold=None, entity= "", savepath='./AiyuDocs/new_signalplots/'):
    """
    Plots each entity and saves them to a figure
    params:
        df: dataframe for an entity
    """
    accs = np.asarray(np.vstack([val for val in df.accelerations]))
    ybot, ytop = np.amax(accs), np.amin(accs)

    feature_names  = df.columns
    display_params = feature_names[3:-1] # Skip acc, timestamp, window length
    for entityname, row in df[:2].iterrows():
        fig = plt.figure(figsize=(20,5), constrained_layout=True)

        x = (row['timestamps'] - row['timestamps'][0])/10**9
        y = row['accelerations']


        ax1 = fig.add_axes((0.1, 0.2, 1, 1))
        ax1.plot(x, y, zorder = 1)
        if entity == 'defect':
            ax1.set_title("Defect: " + entityname)
        else:
            ax1.set_title(i)

        ax1.set_xlabel("Seconds")
        ax1.set_ylabel("Amplitude")
        ax1.set_ylim(ybot,ytop)
        ax1.margins(x=0, y=0.1)

        caption = ',    '.join([ f"|{feature}|: {df.loc[i, feature]}"  for feature in display_params])
        fig.text(0.1, .0005, caption, ha='left', fontsize=12)
        fig.set_size_inches(20, 5, forward=True)
        fig.savefig(savepath + entityname,  bbox_inches='tight')
        plt.show()
