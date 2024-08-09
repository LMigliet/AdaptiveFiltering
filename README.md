# Adaptive Filtering

Reference: [ACS article](https://pubs.acs.org/doi/full/10.1021/acs.analchem.2c01883)

This repository contains a comprehensive pipeline for outlier detection of sigmoidal signals. 

It can analyse data from BioMark HD (Fluidigm, now standard biotools) real-time digital PCR (qdPCR) amplification curve data. 

## Features

- **Data Extraction**: Reads raw dPCR amplification curve data and metadata.
- **Preprocessing**: Removes baseline, negative, and late curves from the dataset.
- **Curve Fitting**: Fits the amplification curves using a sigmoid model in parallel for efficiency.
- **Parameter Normalization**: Normalizes curve fitting parameters across different panels.
- **Outlier Detection**: Identifies outlier curves using the Isolation Forest algorithm.
- **Visualization**: Plots amplification curves, highlighting inliers and outliers.

## CODE STRUCTURE
```
AdaptiveFiltering/
├── data/                        # Folder for storing raw and processed data
│   ├── processed/               # Folder for processed dPCR data files (csv)
│   ├── raw_data/                # Folder for raw dPCR data files (AC.txt)
│   └── metadata_test.csv        # Metadata file
├── src/                         # Source code directory
│   ├── __init__.py              # Init file for src package
│   ├── extractor.py             # Module for data extraction
│   ├── features.py              # Module for feature extraction
│   ├── fitter.py                # Module for curve fitting
│   ├── outlier_detector.py      # Module for outlier detection
│   └── preprocess.py            # Module for data preprocessing
├── tests/                       # Folder for tests
│   ├── test_extractor.py        # Tests for extractor module
│   ├── test_features.py         # Tests for features module
│   ├── test_fitter.py           # Tests for fitter module
│   ├── test_outlier_detector.py # Tests for outlier detector module
│   └── test_preprocess.py       # Tests for preprocess module
├── requirements.txt             # List of dependencies
├── _version.py                  # Version file
├── README.md                    # Readme file
├── LICENSE                      # License file
└── pipeline.py                  # Main pipeline script
```

Python version required: 3.11

Install the required packages using:
```
pip install -r requirements.txt
```
The current version of this project is stored in the VERSION file.

## Usage

1. **Data Preparation**: Place your raw data files (`AC.txt` format) in a folder and prepare a CSV file containing metadata.
2. **Configuration**: Update the folder and metadata paths in the `main.py` file.
3. **Run the Pipeline**: Execute in your virtual enviroment (you can also visualize the amplificiation curves).

### Example

```python
if __name__ == "__main__":
    # Define directories
    folder_path = r"data/test_data/raw_data"
    metadata_path = r"data/test_data/metadata_test.csv"
    output_folder = r"data/test_data/processed"
    path_figures = r"data/test_data/plots"
    NMETA = 5

    # Execute pipeline
    df_ac_inliers, df_ac_outliers, df_param_inliers, df_param_outliers = run(
        folder_path,
        metadata_path,
        output_folder,
        nmeta=NMETA,
    )
```

This project is licensed under the MIT License - see the LICENSE file for details.
