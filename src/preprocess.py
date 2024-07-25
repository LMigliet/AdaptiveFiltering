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
    # Copy the original DataFrame to avoid modifying it directly
    df = df.copy()

    # Calculate the average of the first N cycles for each curve and store it in a new column
    df["initial_cycle_avg"] = (
        df.iloc[:, meta_cols : initial_cycles + meta_cols].astype(float).mean(axis=1)
    )

    # Subtract the average of the first N cycles from each curve (excluding metadata and the new average column)
    df.update(df.iloc[:, meta_cols:-1].sub(df["initial_cycle_avg"], axis=0))

    # Filter out curves where the fluorescence value at the threshold cycle is below the specified threshold
    df = df[df.iloc[:, meta_cols + threshold_cycle] > fluo_threshold]

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
    # Copy the original DataFrame to avoid modifying it directly
    df = df.copy()

    # Filter out curves where the fluorescence value at the threshold cycle is below the lower threshold
    df = df[df.iloc[:, meta_cols + threshold_cycle] > lower_threshold]

    # Filter out curves where the fluorescence value at the threshold cycle is above the upper threshold
    df = df[df.iloc[:, meta_cols + threshold_cycle] < upper_threshold]

    return df
