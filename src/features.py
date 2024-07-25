def end_slope_extract(dataframe, n_meta, n_last_cycles):
    """
    Calculate the mean slope of the last N cycles for each amplification curve.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing amplification curve data with metadata.
    n_meta : int
        Number of metadata columns at the beginning of the DataFrame.
    n_last_cycles : int
        Number of last cycles used to calculate the mean slope.

    Returns
    -------
    pandas.Series
        Series containing the mean slope of the last N cycles for each curve.
    """
    # Calculate the difference between successive cycle values for each curve
    slope_df = dataframe.iloc[:, n_meta:].diff(axis=1)

    # Calculate the mean slope over the last N cycles
    mean_slope_series = slope_df.iloc[:, -n_last_cycles:].mean(axis=1)

    return mean_slope_series
