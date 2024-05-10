import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import table


def plot_distributions(real: pandas.DataFrame, synthetic: pandas.DataFrame, store_destination: str) -> None:
    """
    Plots violin plots (numeric features) or bar charts (categorical features) together with their summary statistics.

    :param real: The real data
    :param synthetic: The synthetic data
    :param store_destination: Path to the folder where the results should be stored.
    """
    for column_name in real.columns:
        matplotlib.use('Agg')
        real_col = real[column_name].to_numpy()
        synthetic_col = synthetic[column_name].to_numpy()
        plt.figure()
        plt.title(column_name)
        patient_types = np.concatenate([np.zeros(real_col.size), np.ones(synthetic_col.size)])
        df = pd.DataFrame(data={"type": np.where(patient_types == 0, "real", "synthetic"),
                                "value": np.concatenate([real_col, synthetic_col])})
        if real_col.dtype == str or real_col.dtype == object:
            ax = sns.countplot(data=df, x="value", hue="type", order=df['value'].value_counts().index)
        elif np.sum(real_col) % 1 == 0 and np.max(real_col) < 10:
            ax = sns.countplot(data=df, x="value", hue="type")
        else:
            if len(real_col) != len(synthetic_col):
                # Determine the length of the longest column
                max_length = max(len(real_col), len(synthetic_col))
                # Fill the shorter column with NaN values to match the length of the longest column
                real_col = np.pad(real_col, pad_width=(0, max_length - len(real_col)), mode='constant',
                                  constant_values=np.nan)
                synthetic_col = np.pad(synthetic_col, pad_width=(0, max_length - len(synthetic_col)), mode='constant',
                                       constant_values=np.nan)
            # create df with now equal values
            df = pd.DataFrame(data={"real": real_col, "synthetic": synthetic_col})
            ax = sns.violinplot(data=df)
            # remove y-labels as they are redundant with the table headers
            ax.set_xticks([])
            table(ax, df.describe().round(2), loc='bottom', colLoc='center', bbox=[0, -0.55, 1, 0.5],
                  colWidths=[.5, .5])
        fig = ax.get_figure()
        matplotlib.pyplot.close()
        fig.savefig(store_destination + "/" + column_name + '.png', bbox_inches="tight")


def plot_correlations(real: pandas.DataFrame, synthetic: pandas.DataFrame, store_destination: str) -> None:
    """
    Plots correlation matrices for real and synthetic features in form of heatmaps.

    :param real: The real data
    :param synthetic: The synthetic data
    :param store_destination: Path to the folder where the results should be stored.
    """
    names = ["real_corr", "syntehtic_corr"]
    for idx, patient_type in enumerate([real, synthetic]):
        plt.figure()
        plt.title("Correlation")
        ax = sns.heatmap(patient_type.corr())
        fig = ax.get_figure()
        fig.savefig(store_destination + "/" + names[idx] + '.png', bbox_inches="tight")




