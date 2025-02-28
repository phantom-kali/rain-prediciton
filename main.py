"""
Rainfall Prediction Model

This script processes a NetCDF rainfall dataset, extracts features,
and builds a logistic regression model to predict rainfall occurrence.
It handles large datasets efficiently by processing data in chunks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import gc

def load_netcdf_metadata(file_path):
    """Load and display metadata from NetCDF file"""
    nc_data = Dataset(file_path, mode='r')
    print("NetCDF file information:")
    
    # List all variables
    print("\nAvailable variables:")
    for var in nc_data.variables:
        var_name = nc_data.variables[var].long_name if hasattr(nc_data.variables[var], 'long_name') else 'No long name'
        print(f"- {var}: {var_name}")
    
    return nc_data

def extract_data_in_chunks(nc_data, sample_size=50, chunk_size=5):
    """Extract data from NetCDF file in memory-efficient chunks"""
    # Extract coordinates and time data
    longitude = nc_data.variables['longitude'][:]
    latitude = nc_data.variables['latitude'][:]
    time = nc_data.variables['time'][:]
    
    # Display dimensions
    print(f"Dataset dimensions: Time={len(time)}, Latitude={len(latitude)}, Longitude={len(longitude)}")
    print(f"Rainfall data shape: {nc_data.variables['rr'].shape}")
    
    # Calculate sampling parameters
    sample_times = min(len(time), sample_size)
    sample_lats = min(len(latitude), 20)
    sample_lons = min(len(longitude), 20)
    
    lat_stride = max(1, len(latitude)//sample_lats)
    lon_stride = max(1, len(longitude)//sample_lons)
    
    print(f"Processing {sample_times} time steps in chunks of {chunk_size}")
    
    data_list = []
    
    # Process data in chunks to reduce memory usage
    for t_start in range(0, sample_times, chunk_size):
        t_end = min(t_start + chunk_size, sample_times)
        print(f"Processing time chunk {t_start+1}-{t_end} of {sample_times}")
        
        # Load only a small chunk of data with spatial strides
        rainfall_chunk = nc_data.variables['rr'][t_start:t_end, ::lat_stride, ::lon_stride]
        
        # Extract features from this chunk
        for t_idx in range(rainfall_chunk.shape[0]):
            actual_t_idx = t_start + t_idx
            
            for lat_idx in range(rainfall_chunk.shape[1]):
                actual_lat_idx = lat_idx * lat_stride
                
                for lon_idx in range(rainfall_chunk.shape[2]):
                    actual_lon_idx = lon_idx * lon_stride
                    
                    rain_value = rainfall_chunk[t_idx, lat_idx, lon_idx]
                    
                    # Skip masked values
                    if np.ma.is_masked(rain_value):
                        continue
                        
                    data_list.append({
                        'rainfall': float(rain_value),
                        'latitude': float(latitude[actual_lat_idx]),
                        'longitude': float(longitude[actual_lon_idx]),
                        'time_index': float(time[actual_t_idx]),
                        'rain_occurred': 1 if rain_value > 0 else 0
                    })
        
        # Free memory
        del rainfall_chunk
        gc.collect()
    
    # Close NetCDF file
    nc_data.close()
    
    return pd.DataFrame(data_list)

def analyze_rainfall_data(data):
    """Analyze rainfall data and generate visualizations"""
    print("\nDataFrame overview:")
    print(data.head())
    print(f"DataFrame shape: {data.shape}")
    
    # Create correlation matrix
    correlation_matrix = data.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Visualize rainfall distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data['rainfall'], kde=True)
    plt.title('Rainfall Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='rain_occurred', data=data)
    plt.title('Rain Occurrence')
    
    plt.tight_layout()
    plt.savefig('rainfall_distribution.png')
    
    return correlation_matrix

def build_rainfall_model(data):
    """Build and evaluate a rainfall prediction model"""
    # Prepare data for rainfall prediction
    X = data[['latitude', 'longitude', 'time_index']]
    y = data['rain_occurred']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'\nModel Evaluation:')
    print(f'Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    return model

def main():
    """Main function to execute the rainfall prediction workflow"""
    file_path = 'rr_ens_mean_0.1deg_reg_2011-2021_v25.0e.nc'
    
    # Load data
    nc_data = load_netcdf_metadata(file_path)
    
    # Extract features
    data = extract_data_in_chunks(nc_data, sample_size=50, chunk_size=5)
    
    # Free memory after extraction
    gc.collect()
    
    # Analyze data
    analyze_rainfall_data(data)
    
    # Build and evaluate model
    model = build_rainfall_model(data)
    
    print("\nRainfall prediction model complete.")
    return model, data

if __name__ == "__main__":
    model, data = main()
    plt.show()  # Show all generated figures