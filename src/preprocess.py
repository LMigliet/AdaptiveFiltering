import logging

from src.logging_config import setup_logger

# Set up logger for the preprocessing module
logger = setup_logger(__name__)


def remove_baseline(df, meta_cols, initial_cycles, threshold_cycle, fluo_threshold):
    """
    Adjusts the baseline of each curve by subtracting the average of the first N cycles.
    Removes baseline curves below a specified fluorescence threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing curve data with metadata.
    meta_cols : int
        Number of metadata columns at the beginning of the DataFrame.
    initial_cycles : int
        Number of initial cycles to calculate the average baseline.
    threshold_cycle : int
        Cycle number used as a checkpoint for curve filtering.
    fluo_threshold : float
        Fluorescence threshold for filtering curves.

    Returns
    -------
    pandas.DataFrame
        DataFrame with adjusted baseline and filtered curves.
    """
    logger.info("Starting baseline removal")

    # Copy the original DataFrame to avoid modifying it directly
    df = df.copy()

    # Calculate the average of the first N cycles for each curve and store it in a new column
    df["initial_cycle_avg"] = (
        df.iloc[:, meta_cols : initial_cycles + meta_cols].astype(float).mean(axis=1)
    )
    logger.debug("Calculated initial cycle averages")

    # Subtract the average of the first N cycles from each curve (excluding metadata and the new average column)
    df.update(df.iloc[:, meta_cols:-1].sub(df["initial_cycle_avg"], axis=0))
    logger.debug("Subtracted initial cycle averages from each curve")

    # Filter out curves where the fluorescence value at the threshold cycle is below the specified threshold
    initial_count = len(df)
    df = df[df.iloc[:, meta_cols + threshold_cycle] > fluo_threshold]
    filtered_count = initial_count - len(df)
    logger.info(
        f"Filtered out {filtered_count} out of {initial_count} curves below the fluorescence threshold\n"
    )

    # Return the DataFrame without the initial_cycle_avg column
    return df.iloc[:, :-1]


def remove_late_curves(
    df, meta_cols, threshold_cycle, upper_threshold, lower_threshold
):
    """
    Filters out curves that have fluorescence values outside the specified range at a given cycle.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing curve data with metadata.
    meta_cols : int
        Number of metadata columns at the beginning of the DataFrame.
    threshold_cycle : int
        Cycle number used as a checkpoint for curve filtering.
    upper_threshold : float
        Upper fluorescence threshold for filtering curves.
    lower_threshold : float
        Lower fluorescence threshold for filtering curves.

    Returns
    -------
    pandas.DataFrame
        DataFrame with late curves removed based on the specified thresholds.
    """
    logger.info("Starting late curve removal")

    # Copy the original DataFrame to avoid modifying it directly
    df = df.copy()

    # Filter out curves where the fluorescence value at the threshold cycle is below the lower threshold
    initial_count = len(df)
    df = df[df.iloc[:, meta_cols + threshold_cycle] > lower_threshold]
    lower_filtered_count = initial_count - len(df)
    logger.info(
        f"Filtered out {lower_filtered_count} curves below the lower fluorescence threshold"
    )

    # Filter out curves where the fluorescence value at the threshold cycle is above the upper threshold
    initial_count = len(df)
    df = df[df.iloc[:, meta_cols + threshold_cycle] < upper_threshold]
    upper_filtered_count = initial_count - len(df)
    logger.info(
        f"Filtered out {upper_filtered_count} curves above the upper fluorescence threshold\n"
    )

    return df


def preprocess_data(
    df,
    nmeta,
    ncycle_int,
    ct_thresh,
    fluo_thresh,
    fluo_lower_thresh,
    fluo_upper_thresh,
    log_level=logging.INFO,
):
    """
    Preprocess the raw data by removing baseline, negative and late curves.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw data DataFrame.
    nmeta : int
        Number of metadata columns.
    ncycle_int : int
        Number of initial cycles to calculate the average baseline.
    ct_thresh : int
        Cycle threshold for filtering.
    fluo_thresh : float
        Fluorescence threshold for filtering.
    fluo_lower_thresh : float
        Upper fluorescence threshold for late curve removal.
    fluo_upper_thresh : float
        Lower fluorescence threshold for late curve removal.
    log_level : int
        The logging level.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    logger.setLevel(log_level)

    df_bs = remove_baseline(
        df,
        nmeta,
        ncycle_int,
        ct_thresh,
        fluo_thresh,
    )
    df_bs_lc = remove_late_curves(
        df_bs,
        nmeta,
        ct_thresh,
        fluo_upper_thresh,
        fluo_lower_thresh,
    )

    return df_bs_lc
