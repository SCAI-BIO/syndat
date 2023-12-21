import pandas
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import table

from sklearn.manifold import TSNE
from syndat.quality import get_outliers


def get_tsne_plot_data(real: pandas.DataFrame, synthetic: pandas.DataFrame):
    x = pandas.concat([real, synthetic])
    perplexity = 30
    if real.shape[1] < 30:
        perplexity = real.shape[1] - 1
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(x)
    border = real.shape[0]
    x_real = tsne_result[:border, 0]
    y_real = tsne_result[:border, 1]
    x_virtual = tsne_result[border:, 0]
    y_virtual = tsne_result[border:, 1]
    return x_real, y_real, x_virtual, y_virtual


def show_outlier_plot(real: pandas.DataFrame, synthetic: pandas.DataFrame):
    trace_real, trace_virtual = get_tsne_plot_data(real, synthetic)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace_real["x"], y=trace_real["y"], mode="markers", name='real'))
    fig.add_trace(go.Scatter(x=trace_virtual['x'], y=trace_virtual['y'], mode="markers", name='synthetic'))
    # Add outlier markings for virtual patients
    outliers = get_outliers(synthetic)
    for outlier in outliers:
        x0 = trace_virtual['x'][outlier] - 0.5
        y0 = trace_virtual['y'][outlier] - 0.5
        x1 = trace_virtual['x'][outlier] + 0.5
        y1 = trace_virtual['y'][outlier] + 0.5
        fig.add_shape(type="circle", xref="x", yref="y", x0=x0, y0=y0, x1=x1, y1=y1, line_color="LightSeaGreen")
    # display
    fig.show()


def plot_distributions(real: pandas.DataFrame, synthetic: pandas.DataFrame, store_destination: str):
    for column_name in real.columns:
        matplotlib.use('Agg')
        real_col = real[column_name].to_numpy()
        virtual_col = synthetic[column_name].to_numpy()
        plt.figure()
        plt.title(column_name)
        patient_types = np.concatenate([np.zeros(real_col.size), np.ones(virtual_col.size)])
        df = pd.DataFrame(data={"type": np.where(patient_types == 0, "real", "synthetic"),
                                "value": np.concatenate([real_col, virtual_col])})
        if real_col.dtype == str or real_col.dtype == object:
            ax = sns.countplot(data=df, x="value", hue="type", order=df['value'].value_counts().index)
        elif np.sum(real_col) % 1 == 0 and np.max(real_col) < 10:
            ax = sns.countplot(data=df, x="value", hue="type")
        else:
            df = pd.DataFrame(data={"real": real_col, "synthetic": virtual_col})
            ax = sns.violinplot(data=df)
            # remove y-labels as they are redundant with the table headers
            ax.set_xticks([])
            table(ax, df.describe().round(2), loc='bottom', colLoc='center', bbox=[0, -0.55, 1, 0.5],
                  colWidths=[.5, .5])
        fig = ax.get_figure()
        matplotlib.pyplot.close()
        fig.savefig(store_destination + "/" + column_name + '.png', bbox_inches="tight")


def create_correlation_plots(real_patients, virtual_patients, store_destination):
    names = ["dec_rp", "dec_vp"]
    for idx, patient_type in enumerate([real_patients, virtual_patients]):
        plt.figure()
        plt.title("Correlation")
        ax = sns.heatmap(patient_type.corr())
        fig = ax.get_figure()
        fig.savefig(store_destination + "/" + names[idx] + '.png', bbox_inches="tight")




