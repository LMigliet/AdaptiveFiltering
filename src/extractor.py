import os

import pandas as pd
from tqdm import tqdm


def extract_curves(filename):
    """
    Extracts the raw amplification data (AC) from a dPCR machine.
    This data is typically saved as panelXX_AC.txt, where XX is the panel number.

    Parameters
    ----------
    filename : str
        Path to the curve file from dPCR.

    Returns
    -------
    pandas.DataFrame
        DataFrame with fluorescence values and cycles as columns.
    """
    df = pd.read_csv(filename, sep="\t", header=None)
    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1::2]
    df.index.name = None
    return df


def process_files_in_folder(folder_path):
    """
    Processes all files in the specified folder, extracting and combining
    amplification data into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing curve files.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with extracted data from all files.
    """
    combined_df_list = []

    # Get list of files in the folder
    ac_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]

    for ac_file in tqdm(ac_files, desc="Reading AC Files", leave=False):
        panel_id = int(
            os.path.basename(ac_file)[5:7]
        )  # Extract panel id from the filename and convert to integer

        # Extract and process the data from the file
        single_df = extract_curves(ac_file).T
        single_df = single_df.astype(float)
        single_df["Panel"] = panel_id
        single_df = single_df.reset_index()

        combined_df_list.append(single_df)

    combined_df = pd.concat(combined_df_list)

    # Reordering columns
    cols = list(combined_df.columns)
    cols.insert(1, cols.pop(cols.index("Panel")))
    combined_df = combined_df[cols]

    # Drop the 'index' column if it exists
    if "index" in combined_df.columns:
        combined_df = combined_df.drop(columns=["index"])

    combined_df = combined_df.reset_index(drop=True)
    print("Raw Data Extracted from TXT files")
    return combined_df
