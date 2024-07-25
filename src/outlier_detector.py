import pprint

import numpy as np
import pandas as pd


def outlier_detector(df, grouping_col, param_df, algorithms, params):
    """
    Detects outliers in amplification curves using specified algorithms.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing amplification curve data.
    grouping_col:
        str with column name (i.e. Panels, or Targets, or assays, and so on)
    param_df : pandas.DataFrame
        DataFrame containing parameter data with columns specified in `params`.
    algorithms : dict
        Dictionary of algorithm names and their corresponding algorithm objects.
    params : list
        List of parameter names to be used, should be a subset of `param_df.columns`.

    Returns
    -------
    dict
        Dictionary where keys are algorithm names and values are tuples containing
        lists of inlier and outlier indices.
    """
    results = {}

    for algo_name, algo in algorithms.items():
        print(algo_name)

        # initialising dataframes for inliers and outliers
        inliers_df = pd.DataFrame(columns=df.columns)
        outliers_df = pd.DataFrame(columns=df.columns)

        for _, group_df in df.sort_values(grouping_col).groupby(grouping_col):
            if group_df.shape[0] < 10:
                outliers_df = outliers_df.append(group_df)
                continue  # Ignore samples with fewer than 10 positive curves

            param_subset_df = param_df.loc[group_df.index, :].copy()
            curve_subset_df = group_df.copy()

            # For certain algorithms, label system switching is required.
            if algo_name in ["Feature Bagging", "ABOD"]:
                algo.fit(param_subset_df[params])
                predictions = algo.labels_  # binary labels (0: inliers, 1: outliers)
                predictions[predictions == 1] = -1
                predictions[predictions == 0] = 1
            elif algo_name == "DBSCAN":
                algo.fit(param_subset_df[params])
                predictions = algo.labels_  # binary labels (>=0: inliers, -1: outliers)
                predictions[predictions >= 0] = 1
            else:
                predictions = algo.fit_predict(param_subset_df[params].values)

            predictions[predictions >= 0] = 1
            predictions[predictions < 0] = (
                -1
            )  # universal label system: 1 -> inliers, -1 -> outliers

            inliers = curve_subset_df.iloc[np.where(predictions == 1)[0], :].copy()
            outliers = curve_subset_df.iloc[np.where(predictions == -1)[0], :].copy()

            inliers_df = inliers_df.append(inliers)
            outliers_df = outliers_df.append(outliers)

        results[algo_name] = (list(inliers_df.index), list(outliers_df.index))

    return results


def outlier_mapping(curve_numbers):
    """
    Maps the number of curves to the corresponding outlier percentage.

    Parameters
    ----------
    curve_numbers : pandas.Series or list
        Series or list of curve numbers (index: sample name).

    Returns
    -------
    pandas.Series
        Series of outlier percentages (index: sample name).
    """
    curve_series = pd.Series(curve_numbers).astype(float)
    outlier_percentage_series = curve_series.copy()

    for i, num in enumerate(curve_series):
        if num < 100:
            outlier_percentage = 11
        elif num < 307:
            outlier_percentage = 11 - 3 / 206 * (num - 100)
        elif num < 715:
            outlier_percentage = 8 - 4 / 407 * (num - 307)
        elif num < 771:
            outlier_percentage = 4 - 2 / 55 * (num - 715)
        else:
            outlier_percentage = 2
        outlier_percentage_series.iloc[i] = outlier_percentage

    return outlier_percentage_series
