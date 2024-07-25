import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import src.extractor as extractor
import src.features as features
import src.fitter as fitter
import src.outlier_detector as outlier_detector
import src.preprocess as preprocess


def load_data(folder_path, metadata_path):
    """
    Load the raw data and metadata, merge them into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing raw data files.
    metadata_path : str
        Path to the metadata CSV file.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing raw data and metadata.
    """
    df_ac = extractor.process_files_in_folder(folder_path)
    df_meta = pd.read_csv(metadata_path)
    df_raw = df_meta.merge(df_ac, on="Panel").reset_index(drop=True)
    return df_raw


def preprocess_data(
    df_raw,
    nmeta,
    N_CYCLE_INIT,
    CT_THRESH,
    FLUO_THRESH,
    FLUO_UPPER_THRESH,
    FLUO_LOWER_THRESH,
):
    """
    Preprocess the raw data by removing baseline, negative and late curves.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw data DataFrame.
    NMETA : int
        Number of metadata columns.
    N_CYCLE_INIT : int
        Number of initial cycles to calculate the average baseline.
    CT_THRESH : int
        Cycle threshold for filtering.
    FLUO_THRESH : float
        Fluorescence threshold for filtering.
    FLUO_UPPER_THRESH : float
        Upper fluorescence threshold for late curve removal.
    FLUO_LOWER_THRESH : float
        Lower fluorescence threshold for late curve removal.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    df_bs = preprocess.remove_baseline(
        df_raw, nmeta, N_CYCLE_INIT, CT_THRESH, FLUO_THRESH
    )
    df_bs_lc = preprocess.remove_late_curves(
        df_bs, nmeta, CT_THRESH, FLUO_UPPER_THRESH, FLUO_LOWER_THRESH
    )
    return df_bs_lc


def fit_curves_parallel(df_bs_lc, nmeta, initial_params, param_bounds, n_jobs):
    """
    Fit curves to the data in parallel.

    Parameters
    ----------
    df_bs_lc : pd.DataFrame
        Preprocessed DataFrame.
    NMETA : int
        Number of metadata columns.
    initial_params : tuple
        Initial parameters for curve fitting.
    param_bounds : tuple
        Parameter bounds for curve fitting.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing fitted parameters.
    """
    split_df = np.array_split(df_bs_lc, n_jobs)
    results_obj = []

    print(f"Start Parallel Fitting with {n_jobs} CPU CORES...")
    with multiprocessing.Pool() as pool:
        for df_ in split_df:
            results_obj.append(
                pool.apply_async(
                    fitter.fit_curves, (df_, nmeta, initial_params, param_bounds)
                )
            )
        pool.close()
        pool.join()
    print(f"End Parallel Fitting!")
    sample_list = [obj.get() for obj in results_obj]
    return pd.concat(sample_list)


def normalize_parameters(param_df, param_set):
    """
    Normalize the parameters for each panel.

    Parameters
    ----------
    param_df : pd.DataFrame
        DataFrame containing parameters.
    param_set : list
        List of parameter names to be normalized.

    Returns
    -------
    pd.DataFrame
        DataFrame containing normalized parameters.
    """
    param_df_norm = param_df.copy()
    param_norm_set = [param + "_norm" for param in param_set]

    for _, df_ in param_df.groupby("Panel"):
        scaler = StandardScaler()
        scaled_params = scaler.fit_transform(df_.loc[:, param_set])
        param_df_norm.loc[df_.index, param_norm_set] = scaled_params

    return param_df_norm


def detect_outliers(df_bs_lc, param_df_norm, param_set, outlierpc_series):
    """
    Detect outliers using specified algorithms.

    Parameters
    ----------
    df_bs_lc : pd.DataFrame
        Preprocessed DataFrame.
    param_df_norm : pd.DataFrame
        DataFrame containing normalized parameters.
    param_set : list
        List of parameters for outlier detection.
    outlierpc_series : pd.Series
        Series containing outlier percentages.

    Returns
    -------
    tuple
        Tuple containing lists of inlier and outlier indices.
    """
    inlier_index = []
    outlier_index = []

    with multiprocessing.Pool() as pool:
        results_obj = []

        for panel_name, current_df_param in param_df_norm.groupby("Panel"):
            algo_set = {
                "Isolation_Forest": IsolationForest(
                    contamination=outlierpc_series[panel_name] / 100,
                    n_jobs=1,
                    random_state=42,
                    verbose=0,
                )
            }

            results_obj.append(
                pool.apply_async(
                    outlier_detector.outlier_detector,
                    (
                        df_bs_lc.loc[current_df_param.index, :],
                        "Panel",  # column for the grouping to filter within a panel
                        current_df_param,
                        algo_set,
                        param_set,
                    ),
                )
            )

        pool.close()
        pool.join()

    for obj in results_obj:
        result = obj.get()
        inlier_index.extend(result["Isolation_Forest"][0])
        outlier_index.extend(result["Isolation_Forest"][1])

    return inlier_index, outlier_index


def main(folder_path, metadata_path, output_folder, nmeta=5):
    """
    Main pipeline for processing dPCR data.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing AC.txt files.
    metadata_path : str
        Path to the CSV file containing metadata.

    Returns
    -------
    tuple
        DataFrames containing inliers and outliers data and parameters.
    """
    ### EXTRACT DATA from TXT FILES and META
    df_raw = load_data(folder_path, metadata_path)

    N_CYCLE_INIT = 5
    CT_THRESH = 35
    FLUO_THRESH = 0.1
    FLUO_UPPER_THRESH = 1000
    FLUO_LOWER_THRESH = 0.25

    ### PREPROCESS DATA
    df_bs_lc = preprocess_data(
        df_raw,
        nmeta,
        N_CYCLE_INIT,
        CT_THRESH,
        FLUO_THRESH,
        FLUO_UPPER_THRESH,
        FLUO_LOWER_THRESH,
    )

    ### FITTING
    initial_params = (1, 0, 0.5, 20, 100)
    param_bounds = ((0, -0.3, 0, -30, 0), (10, 0.3, 2, 50, 100))
    n_jobs = multiprocessing.cpu_count()

    print("\nINITIAL FIT")
    sample_df_param = fit_curves_parallel(
        df_bs_lc.sample(frac=0.01, random_state=9),
        nmeta,
        initial_params,
        param_bounds,
        n_jobs,
    )
    print("\nWHOLE DATASET FIT")
    param_set = ["Fm", "Fb", "Sc", "Cs", "As"]
    param_df = fit_curves_parallel(
        df_bs_lc, nmeta, sample_df_param[param_set].median(), param_bounds, n_jobs
    )

    ### ADD ENDSLOP PARAM & NORMALISE PARAMS
    param_df["endSlope"] = features.end_slope_extract(df_bs_lc, nmeta, 5)
    param_df_norm = normalize_parameters(param_df, param_set + ["endSlope"])

    ### OUTLIER DETECTION
    print("\nOUTLIER DETECTION")
    outlierpc_series = outlier_detector.outlier_mapping(
        param_df_norm.value_counts("Panel").sort_index()
    )

    FEATURE_SET = ["Fm_norm", "Fb_norm", "Sc_norm", "Cs_norm", "endSlope_norm"]

    inlier_index, outlier_index = detect_outliers(
        df_bs_lc,
        param_df_norm,
        FEATURE_SET,
        outlierpc_series,
    )

    inlier_df_ac = df_bs_lc.loc[inlier_index, :]
    outlier_df_ac = df_bs_lc.loc[outlier_index, :]

    inlier_df_param_ex = param_df.loc[inlier_index, :]
    outlier_df_param_ex = param_df.loc[outlier_index, :]

    mse_threshold = 0.0005
    removed_df_param = inlier_df_param_ex[inlier_df_param_ex["mse"] > mse_threshold]

    inlier_df_ac = inlier_df_ac.drop(removed_df_param.index)
    inlier_df_param_ex = inlier_df_param_ex.drop(removed_df_param.index)

    removed_df_ac = df_bs_lc.loc[removed_df_param.index]
    outlier_df_ac = outlier_df_ac.append(removed_df_ac)
    outlier_df_param_ex = outlier_df_param_ex.append(removed_df_param)

    df_param_inliers = param_df_norm.loc[inlier_df_ac.index]
    df_param_outliers = param_df_norm.loc[outlier_df_ac.index]

    # Save the results
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    inlier_df_ac.to_csv(os.path.join(output_folder, "inliers_ac.csv"), index=False)
    outlier_df_ac.to_csv(os.path.join(output_folder, "outliers_ac.csv"), index=False)
    df_param_inliers.to_csv(
        os.path.join(output_folder, "inliers_params.csv"), index=False
    )
    df_param_outliers.to_csv(
        os.path.join(output_folder, "outliers_params.csv"), index=False
    )

    return (
        inlier_df_ac,
        outlier_df_ac,
        df_param_inliers,
        df_param_outliers,
    )


###### TEST USAGE ######

if __name__ == "__main__":

    folder_path = r"data/test_data/raw_data"  # specify the path of your data
    metadata_path = r"data/test_data/metadata_test.csv"  # specify the path of your metadata and adjust NMETA if needed.
    output_folder = r"data/test_data/processed"
    NMETA = 5

    df_ac_inliers, df_ac_outliers, df_param_inliers, df_param_outliers = main(
        folder_path, metadata_path, output_folder, nmeta=NMETA
    )

    # # PLOT IF YOU WANT
    # from src.plotter import plot_amplification_curves
    # plot_amplification_curves(df_ac_inliers, df_ac_outliers, NMETA)
