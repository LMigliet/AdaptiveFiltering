"""
extractor.py

This module contains functions to extract and process 
raw amplification data from dPCR machines.
It includes functions to extract curves from individual 
files and to process all files in a specified folder.
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from src.logging_config import setup_logger

logger = setup_logger(__name__)


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
    try:
        df = pd.read_csv(filename, sep="\t", header=None)
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1::2]
        df.index.name = None
        logger.info(f"Successfully extracted data from {filename}")
        return df
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return pd.DataFrame()


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

    for ac_file in ac_files:
        panel_id = int(
            os.path.basename(ac_file)[5:7]
        )  # Extract panel id from the filename and convert to integer

        # Extract and process the data from the file
        single_df = extract_curves(ac_file).T
        if single_df.empty:
            logger.warning(f"No data extracted from {ac_file}, skipping.")
            continue
        single_df = single_df.astype(float)
        single_df["Panel"] = panel_id
        single_df = single_df.reset_index()

        combined_df_list.append(single_df)

    if not combined_df_list:
        logger.warning("No valid data files found.")
        return pd.DataFrame()

    combined_df = pd.concat(combined_df_list)

    # Reordering columns
    cols = list(combined_df.columns)
    cols.insert(1, cols.pop(cols.index("Panel")))
    combined_df = combined_df[cols]

    # Drop the 'index' column if it exists
    if "index" in combined_df.columns:
        combined_df = combined_df.drop(columns=["index"])

    combined_df = combined_df.reset_index(drop=True)
    logger.info("Raw data extracted from TXT files")
    return combined_df


def load_data(folder_path, metadata_path, log_level=logging.INFO):
    """
    Load the raw data and metadata, merge them into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing raw data files.
    metadata_path : str
        Path to the metadata CSV file.
    log_level : int
        The logging level.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing raw data and metadata.
    """
    logger.setLevel(log_level)

    df_ac = process_files_in_folder(folder_path)
    if df_ac.empty:
        raise ValueError("No data extracted from the provided folder path.")
    df_meta = pd.read_csv(metadata_path)

    df_raw = df_meta.merge(df_ac, on="Panel").reset_index(drop=True)
    logger.info(
        f"Meta shape: {df_meta.shape} | Raw data shape: {df_ac.shape} | Final DF shape: {df_raw.shape}\n"
    )

    return df_raw
