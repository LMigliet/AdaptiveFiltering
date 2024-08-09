import os

from pipeline import run
from src.plotter import plot_amplification_curves_by_panel

if __name__ == "__main__":

    # Define directories
    folder_path = r"data/test_data/raw_data"
    metadata_path = r"data/test_data/metadata_test.csv"
    output_folder = r"data/test_data/processed"
    path_figures = r"data/test_data/plots"
    NMETA = 5

    # Create directories if they do not exist
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(path_figures, exist_ok=True)

    # Execute main function
    df_ac_inliers, df_ac_outliers, df_param_inliers, df_param_outliers = run(
        folder_path,
        metadata_path,
        output_folder,
        nmeta=NMETA,
    )

    # Plot amplification curves (memeory consuming)
    plot_amplification_curves_by_panel(
        df_ac_inliers,
        df_ac_outliers,
        NMETA,
        path_figures,
        meta=["Target", "Assay", "Conc"],
    )
