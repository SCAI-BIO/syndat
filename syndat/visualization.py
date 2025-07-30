import logging

import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

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
        ax = sns.heatmap(patient_type.corr(), vmin=-1, vmax=1)
        fig = ax.get_figure()
        fig.savefig(store_destination + "/" + names[idx] + '.png', bbox_inches="tight")


def plot_shap_discrimination(real: pd.DataFrame, synthetic: pd.DataFrame, save_path: str = None) -> None:
    """
    Generates a SHAP summary plot to illustrate the discrimination between real and synthetic datasets
    using a Random Forest classifier.

    :param real: The real data
    :param synthetic: The synthetic data
    :param save_path: Path to the file where the resulting plot should be saved. If None, the plot will not be saved.

    :return: None
    """
    # Assuming 'real' and 'synthetic' are your datasets and are pandas DataFrames
    # Add a label column to each dataset
    real['label'] = 1
    synthetic['label'] = 0

    # Combine datasets
    combined_data = pd.concat([real, synthetic])

    # Separate features and labels
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)

    # Predict probabilities and calculate AUC score
    y_pred_proba = rfc.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    logger.info(f'AUC Score: {auc_score}')

    # Compute SHAP values
    explainer = shap.TreeExplainer(rfc)
    shap_values = explainer.shap_values(X_test)

    # Plot SHAP summary
    plt.figure()
    shap.summary_plot(shap_values[:, :, 1], X_test, show=False)

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    else:
        # Show the plot
        plt.show()


def plot_categorical_feature(feature: str, real_data: pandas.DataFrame, synthetic_data: pandas.DataFrame) -> None:
    """
    Plots count plots for a categorical feature from both real and synthetic datasets.

    :param feature: The feature to be plotted
    :param real_data: The real data
    :param synthetic_data: The synthetic data
    """
    plt.figure(figsize=(14, 6))

    # Plot for the real dataset
    plt.subplot(1, 2, 1)
    sns.countplot(x=feature, data=real_data, color='blue')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Real Data - {feature}')
    plt.xticks(rotation=90)

    # Get the maximum y value for the real dataset
    real_max_y = plt.gca().get_ylim()[1]  # Get the current y limits of the real plot

    # Plot for the synthetic dataset
    plt.subplot(1, 2, 2)
    sns.countplot(x=feature, data=synthetic_data, color='orange')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Synthetic Data - {feature}')
    plt.xticks(rotation=90)

    # Set the y-axis limits to be the same as the real dataset
    plt.ylim(0, real_max_y)  # Align y-axis limits

    plt.tight_layout()
    plt.show()


def plot_numerical_feature(feature: str, real_data: pandas.DataFrame, synthetic_data: pandas.DataFrame) -> None:
    """
    Plots violin plots for a numerical feature from both real and synthetic datasets and displays their
    summary statistics.

    :param feature: The feature to be plotted
    :param real_data: The real data
    :param synthetic_data: The synthetic data
    """
    # Calculate summary statistics
    real_stats = __get_summary_stats(real_data, feature)
    synthetic_stats = __get_summary_stats(synthetic_data, feature)

    # Create summary statistics DataFrame
    stats_df = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Real Data': [real_stats['Mean'], real_stats['Median'], real_stats['Std Dev'], real_stats['Min'],
                      real_stats['Max']],
        'Synthetic Data': [synthetic_stats['Mean'], synthetic_stats['Median'], synthetic_stats['Std Dev'],
                           synthetic_stats['Min'], synthetic_stats['Max']]
    })

    plt.figure(figsize=(14, 8))

    # Compute the combined range for x-axis limits across both datasets
    min_value = min(real_data[feature].min(), synthetic_data[feature].min())
    max_value = max(real_data[feature].max(), synthetic_data[feature].max())

    # Plot for the real dataset
    plt.subplot(2, 2, 1)
    sns.violinplot(x=real_data[feature], color='blue')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title(f'Real Data - {feature}')
    plt.xlim(min_value, max_value)

    # Plot for the synthetic dataset
    plt.subplot(2, 2, 2)
    sns.violinplot(x=synthetic_data[feature], color='orange')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title(f'Synthetic Data - {feature}')
    plt.xlim(min_value, max_value)

    # Display summary statistics table
    plt.subplot(2, 1, 2)
    plt.axis('off')
    summary_stat_table = plt.table(cellText=stats_df.values,
                                   colLabels=stats_df.columns,
                                   rowLabels=stats_df['Statistic'],
                                   cellLoc='center',
                                   loc='center',
                                   bbox=[0.0, -0.5, 1.0, 0.4])
    summary_stat_table.auto_set_font_size(False)
    summary_stat_table.set_fontsize(10)
    summary_stat_table.scale(1.2, 1.2)  # Adjust the size of the table

    plt.tight_layout()
    plt.show()


def __get_summary_stats(data: pd.DataFrame, col_name: str):
    return {
        'Mean': data[col_name].mean(),
        'Median': data[col_name].median(),
        'Std Dev': data[col_name].std(),
        'Min': data[col_name].min(),
        'Max': data[col_name].max()
    }
