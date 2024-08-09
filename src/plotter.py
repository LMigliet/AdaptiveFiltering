import matplotlib.pyplot as plt


def plot_amplification_curves_by_panel(
    inlier_df_ac,
    outlier_df_ac,
    NMETA,
    path_figures,
    meta=None,
):
    """
    Plot amplification curves for each panel.

    Parameters
    ----------
    inlier_df_ac : pd.DataFrame
        DataFrame containing inlier amplification curves.
    outlier_df_ac : pd.DataFrame
        DataFrame containing outlier amplification curves.
    NMETA : int
        Number of metadata columns.
    """
    for panel_name, sample_inlier_df_ac in inlier_df_ac.groupby("Panel"):
        sample_outlier_df_ac = outlier_df_ac[outlier_df_ac["Panel"] == panel_name]
        x_ac = sample_inlier_df_ac.columns[NMETA:].astype(float).to_numpy()

        plt.figure(figsize=(10, 6))
        plt.grid(alpha=0.3)

        if meta:
            plt.title(
                f"Panel {panel_name} | {meta[0]}: {sample_inlier_df_ac[meta[0]].unique()[0]} | {meta[1]}: {sample_inlier_df_ac[meta[1]].unique()[0]} | {meta[2]}: {sample_inlier_df_ac[meta[2]].unique()[0]}"
            )
        else:
            plt.title(f"Panel {panel_name}")

        plt.plot(x_ac, sample_inlier_df_ac.iloc[:, NMETA:].T, color="blue")
        plt.plot(
            x_ac, sample_outlier_df_ac.iloc[:, NMETA:].T, color="lightgray", alpha=0.5
        )

        # Add legend for a single representative line for inliers and outliers
        plt.plot([], [], color="blue", label="Inliers")
        plt.plot([], [], color="lightgray", label="Outliers")

        plt.legend()
        plt.ylabel("Fluorescence")
        plt.xlabel("Cycles")

        # Save the figure
        plt.savefig(f"{path_figures}/Panel_{panel_name}.png")
        plt.close()
