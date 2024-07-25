# Adaptive Filtering

Reference: [https://pubs.acs.org/doi/full/10.1021/acs.analchem.2c01883]

This repository contains a comprehensive pipeline for outlier detection of sigmoidal signals. 

It can analyse data from BioMark HD (Fluidigm, now standard biotools) real-time digital PCR (qdPCR) amplification curve data. 

## Features

- **Data Extraction**: Reads raw dPCR amplification curve data and metadata.
- **Preprocessing**: Removes baseline, negative, and late curves from the dataset.
- **Curve Fitting**: Fits the amplification curves using a sigmoid model in parallel for efficiency.
- **Parameter Normalization**: Normalizes curve fitting parameters across different panels.
- **Outlier Detection**: Identifies outlier curves using the Isolation Forest algorithm.
- **Visualization**: Plots amplification curves, highlighting inliers and outliers.


## Usage

1. **Data Preparation**: Place your raw data files (`AC.txt` format) in a folder and prepare a CSV file containing metadata.
2. **Configuration**: Update the folder paths and metadata file path in the `main` function.
3. **Run the Pipeline**: Execute the pipeline to process the data and visualize the results.

### Example

```python
if __name__ == "__main__":
    folder_path = "path/to/your/raw_data"
    metadata_path = "path/to/your/metadata.csv"
    inlier_df_ac, outlier_df_ac, NMETA, inlier_params, outlier_params = main(folder_path, metadata_path)
```

This project is licensed under the MIT License - see the LICENSE file for details.
