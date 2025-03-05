# Rainfall and Flood Prediction Model

This project processes NetCDF rainfall dataset and builds machine learning models to predict rainfall occurrence and flood risk using geospatial and temporal features. The models are designed to handle large meteorological datasets efficiently by processing data in chunks.

## Overview

Climate data analysis often requires processing large datasets. This project demonstrates an approach to:
- Load and process large NetCDF meteorological datasets efficiently
- Extract meaningful features from geospatial rainfall data
- Implement machine learning models to predict rainfall occurrence and flood risk
- Visualize rainfall patterns and analyze correlations between weather data and flooding

## Requirements

- Python 3.8+
- Required packages:
  - numpy
  - pandas
  - scikit-learn
  - seaborn
  - matplotlib
  - netCDF4

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib netCDF4
```

## Dataset

This project uses rainfall ensemble mean data stored in NetCDF format. The dataset includes:

- Time series data spanning 2011-2021
- Geospatial coverage at 0.1-degree resolution
- Rainfall measurements in mm

To run this project, place the NetCDF file in the project root directory:
```
rr_ens_mean_0.1deg_reg_2011-2021_v25.0e.nc
```

## Project Structure

```
rain-prediction/
│
├── main.py                 # Main script for rainfall prediction
├── flood_prediction.py     # Module for flood risk assessment
├── rain-prediction.ipynb   # Jupyter notebook with exploratory data analysis
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
│
└── outputs/                # Generated outputs during execution
    ├── correlation_matrix.png
    ├── rainfall_distribution.png
    ├── flood_risk_distribution.png
    ├── rainfall_vs_flood_risk.png
    ├── geographical_flood_risk.png
    └── flood_prediction_confusion_matrix.png
```

## Usage

### Running the Rainfall Prediction Model

To run the rainfall prediction model:

```bash
python main.py
```

This will:
1. Load the NetCDF dataset
2. Process the data in memory-efficient chunks
3. Extract geospatial and temporal features
4. Build and evaluate a logistic regression model
5. Generate visualizations of the data

### Running the Flood Prediction Model

To run the flood risk assessment and prediction:

```bash
python flood_prediction.py
```

This will:
1. Load processed rainfall data or generate it from the NetCDF file
2. Calculate accumulated rainfall over continuous periods
3. Assign flood risk categories based on rainfall intensity and accumulation
4. Build a Random Forest model to predict flood risk
5. Generate visualizations of flood risk patterns
6. Demonstrate sample flood risk predictions

### Using the Notebook

The Jupyter notebook provides an interactive exploration of the dataset:

```bash
jupyter notebook rain-prediction.ipynb
```

## Features

### Data Processing
- Memory-efficient data loading and processing
- Geospatial and temporal feature extraction
- Handling of masked values and large datasets

### Rainfall Prediction Model
- Logistic regression for binary classification of rainfall events
- Feature engineering from geospatial coordinates and time
- Model evaluation and performance metrics

### Flood Prediction Model
- Calculation of accumulated rainfall over continuous periods
- Categorization of flood risk based on rainfall thresholds
- Random Forest classifier for multi-class flood risk prediction
- Feature importance analysis
- Geographical visualization of flood risk

### Visualizations
- Rainfall distribution histograms
- Correlation matrix heatmaps
- Flood risk distribution analysis
- Geographical mapping of high-risk areas

## Model Performance

- The logistic regression model achieves reasonable accuracy in predicting rainfall occurrence based on geospatial and temporal features.
- The Random Forest classifier provides insights into flood risk prediction with detailed probabilities for different risk levels.
- Both models include comprehensive performance metrics including precision, recall, and F1-score.


## License

This project is available under the MIT License.
