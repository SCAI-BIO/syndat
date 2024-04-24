import pandas
import pandas as pd
import numpy as np
import scipy.spatial.distance

from sklearn import ensemble, neighbors
from sklearn.model_selection import cross_val_score

from syndat.domain import OutlierPredictionMode, AggregationMethod


def get_auc(real: pandas.DataFrame, synthetic: pandas.DataFrame, n_folds=10):
    x = pd.concat([real, synthetic])
    y = np.concatenate((np.zeros(real.shape[0]), np.ones(synthetic.shape[0])), axis=None)
    rfc = ensemble.RandomForestClassifier()
    return np.average(cross_val_score(rfc, x, y, cv=n_folds, scoring='roc_auc'))


def get_jsd(real: pandas.DataFrame, synthetic: pandas.DataFrame, aggregate_results: bool = True,
            aggregation_method: AggregationMethod = AggregationMethod.AVERAGE):
    # load datasets & remove id column
    jsd_dict = {}
    for col in real:
        # delete empty cells
        real_wo_missing = real[col].dropna()
        # binning
        if np.sum(real[col].values) % 1 == 0 and np.sum(synthetic[col].values) % 1 == 0:
            # categorical column
            real_binned = np.bincount(real[col])
            virtual_binned = np.bincount(synthetic[col])
        else:
            # get optimal amount of bins
            n_bins = np.histogram_bin_edges(real_wo_missing, bins='auto')
            real_binned = np.bincount(np.digitize(real_wo_missing, n_bins))
            virtual_binned = np.bincount(np.digitize(synthetic[col], n_bins))
        # one array might be shorter here then the other, e.g. if real patients contain the categorical
        # encoding 0-3, but virtual patients only contain 0-2
        # in this case -> fill missing bin with zero
        if len(real_binned) != len(virtual_binned):
            padding_size = np.abs(len(real_binned) - len(virtual_binned))
            if len(real_binned) > len(virtual_binned):
                virtual_binned = np.pad(virtual_binned, (0, padding_size))
            else:
                real_binned = np.pad(real_binned, (0, padding_size))
        # compute jsd
        jsd = scipy.spatial.distance.jensenshannon(real_binned, virtual_binned)
        jsd_dict[col] = jsd
    if aggregate_results and aggregation_method == AggregationMethod.AVERAGE:
        return np.mean(np.array(list(jsd_dict.values())))
    elif aggregate_results and aggregation_method == AggregationMethod.MEDIAN:
        return np.median(np.array(list(jsd_dict.values())))
    else:
        return jsd_dict


def get_correlation_quotient(real: pandas.DataFrame, synthetic: pandas.DataFrame):
    corr_real = real.corr()
    corr_synthetic = synthetic.corr()
    norm_diff = np.linalg.norm(corr_real - corr_synthetic)
    norm_real = np.linalg.norm(corr_real)
    norm_quotient = norm_diff / norm_real
    return norm_quotient


def get_outliers(synthetic: pd.DataFrame, mode: OutlierPredictionMode = OutlierPredictionMode.isolationForest,
                 anomaly_score: bool = False):
    if mode == OutlierPredictionMode.isolationForest:
        model = ensemble.IsolationForest(random_state=42)
        return outlier_predictions(model, anomaly_score, x=synthetic)
    elif mode == OutlierPredictionMode.local_outlier_factor:
        model = neighbors.LocalOutlierFactor(n_neighbors=2)
        return outlier_predictions(model, anomaly_score, x=synthetic)


def outlier_predictions(model, anomaly_score, x):
    if anomaly_score:
        model.fit(x)
        return model.score_samples(X=x) * -1
    else:
        predictions = model.fit_predict(X=x)
        outliers_idx = np.array(np.where(predictions == -1))[0]
        return outliers_idx
