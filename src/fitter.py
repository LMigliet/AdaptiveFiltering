import numpy as np
import pandas as pd
import scipy.optimize as opt


def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
    """
    Five parameter sigmoid model (universal notations).

    Parameters
    ----------
    x : array_like
        Iterative x locations.
    Fm : float
        Maximum fluorescence.
    Fb : float
        Background fluorescence.
    Sc : float
        Slope of the curve.
    Cs : float
        Fractional cycle of the inflection point (1/c).
    As : float
        Asymmetric shape (Richard's coefficient).

    Returns
    -------
    y : array_like
        Calculated y outputs.
    """
    return Fm / (1.0 + np.exp(-(x - Cs) * Sc)) ** As + Fb


def residual_function(params, x, y_obs):
    """
    Calculate the residuals between observed and modeled data.

    Parameters
    ----------
    params : array_like
        Model parameters.
    x : array_like
        X data points.
    y_obs : array_like
        Observed y data points.

    Returns
    -------
    residuals : array_like
        Residuals between observed and modeled data.
    """
    return sigmoid5_un(x, *params) - y_obs


def loss_function(params, x, y_obs):
    """
    Calculate the least squares loss.

    Parameters
    ----------
    params : array_like
        Model parameters.
    x : array_like
        X data points.
    y_obs : array_like
        Observed y data points.

    Returns
    -------
    loss : float
        Calculated loss value.
    """
    residuals = residual_function(params, x, y_obs)
    return 0.5 * np.sum(residuals**2)


def fit_curves(dataframe, n_meta, initial_params, param_bounds):
    """
    Fit amplification curves to the five parameter sigmoid model.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing amplification curves with metadata.
    n_meta : int
        Number of metadata columns.
    initial_params : array_like
        Initial parameter estimates for curve fitting.
    param_bounds : tuple
        Bounds for the parameters.

    Returns
    -------
    param_df_with_meta : pandas.DataFrame
        DataFrame containing metadata and fitted parameters.
    """
    df_copy = dataframe.copy()
    x_data = df_copy.iloc[:, n_meta:].T.index.astype(float)
    n_samples = df_copy.shape[0]

    # DataFrame to store fitted parameters
    param_df = pd.DataFrame(columns=["Fm", "Fb", "Sc", "Cs", "As", "mse"])

    # Loop through each amplification curve
    for i in range(n_samples):
        y_data = df_copy.iloc[i, n_meta:].T.copy()

        try:
            popt, _ = opt.curve_fit(
                sigmoid5_un,
                x_data,
                y_data,
                p0=initial_params,
                bounds=param_bounds,
                method="trf",
            )
        except RuntimeError:
            print(f"Cannot find optimal solutions for sample #{df_copy.index[i]}")
            popt = param_bounds[1]

        mse = np.mean((y_data - sigmoid5_un(x_data, *popt)) ** 2)
        param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], mse]

    param_df.index = dataframe.index
    param_df_with_meta = pd.concat([dataframe.iloc[:, :n_meta], param_df], axis=1)

    return param_df_with_meta


def fit_curves_global_optimization(dataframe, n_meta, param_bounds):
    """
    Fit amplification curves using global optimization.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing amplification curves with metadata.
    n_meta : int
        Number of metadata columns.
    param_bounds : tuple
        Bounds for the parameters.

    Returns
    -------
    param_df_with_meta : pandas.DataFrame
        DataFrame containing metadata and fitted parameters.
    """
    df_copy = dataframe.copy()
    x_data = df_copy.iloc[:, n_meta:].T.index.astype(float)
    n_samples = df_copy.shape[0]

    # DataFrame to store fitted parameters
    param_df = pd.DataFrame(columns=["Fm", "Fb", "Sc", "Cs", "As", "mse"])

    # Loop through each amplification curve
    for i in range(n_samples):
        y_data = df_copy.iloc[i, n_meta:].T.copy()

        result = opt.dual_annealing(
            loss_function, bounds=np.array(param_bounds).T, args=(x_data, y_data)
        )
        popt = result.x

        mse = np.mean((y_data - sigmoid5_un(x_data, *popt)) ** 2)
        param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], mse]

    param_df.index = dataframe.index
    param_df_with_meta = pd.concat([dataframe.iloc[:, :n_meta], param_df], axis=1)

    return param_df_with_meta
