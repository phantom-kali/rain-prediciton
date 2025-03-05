"""
Flood Prediction Module

This module extends the rainfall prediction capabilities to assess flood risk.
It processes rainfall data to identify potential flooding conditions based on
rainfall intensity, accumulated precipitation, and geographical factors.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from main import load_netcdf_metadata, extract_data_in_chunks

# Constants for flood prediction
HEAVY_RAINFALL_THRESHOLD = 30.0  # mm per day
MODERATE_RAINFALL_THRESHOLD = 10.0  # mm per day
CONTINUOUS_RAIN_DAYS = 3  # Number of consecutive days to consider for accumulation

def calculate_accumulated_rainfall(data):
    """
    Calculate accumulated rainfall over time periods to identify flood risk.
    
    Args:
        data: DataFrame with rainfall data
        
    Returns:
        DataFrame with additional accumulated rainfall features
    """
    print("Calculating accumulated rainfall metrics...")
    
    # to avoid modifying the original dataframe
    flood_data = data.copy()
    
    # Sort by location and time then group by location
    flood_data = flood_data.sort_values(['latitude', 'longitude', 'time_index'])
    grouped = flood_data.groupby(['latitude', 'longitude'])
    
    def calculate_rolling_sum(group):
        if len(group) >= CONTINUOUS_RAIN_DAYS:
            # Calculate n-day accumulated rainfall
            group['accumulated_rainfall'] = group['rainfall'].rolling(window=CONTINUOUS_RAIN_DAYS, min_periods=1).sum()
            return group
        else:
            group['accumulated_rainfall'] = group['rainfall']
            return group
    
    # Apply the function to each group
    flood_data = grouped.apply(calculate_rolling_sum)
    flood_data = flood_data.reset_index(drop=True)
    
    # Fill NaN values with the original rainfall value
    flood_data['accumulated_rainfall'].fillna(flood_data['rainfall'], inplace=True)
    
    return flood_data

def calculate_flood_risk(data):
    """
    Calculate flood risk level based on rainfall intensity and accumulation.
    
    Args:
        data: DataFrame with rainfall data including accumulated values
        
    Returns:
        DataFrame with flood risk labels
    """
    print("Calculating flood risk categories...")
    
    # Add flood risk column (0: Low, 1: Moderate, 2: High)
    risk_data = data.copy()
    
    # Apply risk rules
    conditions = [
        (risk_data['accumulated_rainfall'] > HEAVY_RAINFALL_THRESHOLD),
        (risk_data['accumulated_rainfall'] > MODERATE_RAINFALL_THRESHOLD),
        (risk_data['accumulated_rainfall'] <= MODERATE_RAINFALL_THRESHOLD)
    ]
    choices = [2, 1, 0]  # High, Moderate, Low
    
    risk_data['flood_risk'] = np.select(conditions, choices, default=0)
    
    # Statistical analysis of risk levels
    risk_counts = risk_data['flood_risk'].value_counts()
    print("\nFlood risk distribution:")
    risk_labels = {0: "Low", 1: "Moderate", 2: "High"}
    for risk_level, count in risk_counts.items():
        percentage = (count / len(risk_data)) * 100
        print(f"{risk_labels[risk_level]} risk: {count} samples ({percentage:.2f}%)")
    
    return risk_data

def visualize_flood_risk(risk_data):
    """
    Visualize flood risk distribution and correlation with other features.
    
    Args:
        risk_data: DataFrame with flood risk data
    """
    print("Generating flood risk visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Plot flood risk distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='flood_risk', data=risk_data, 
                 order=[0, 1, 2],
                 palette=['green', 'yellow', 'red'])
    plt.title('Flood Risk Distribution')
    plt.xlabel('Flood Risk (0: Low, 1: Moderate, 2: High)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/flood_risk_distribution.png')
    
    # Plot relationship between accumulated rainfall and flood risk
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='flood_risk', y='accumulated_rainfall', data=risk_data, 
               order=[0, 1, 2],
               palette=['green', 'yellow', 'red'])
    plt.title('Relationship Between Accumulated Rainfall and Flood Risk')
    plt.xlabel('Flood Risk (0: Low, 1: Moderate, 2: High)')
    plt.ylabel('Accumulated Rainfall (mm)')
    plt.tight_layout()
    plt.savefig('outputs/rainfall_vs_flood_risk.png')
    
    # Plot geographical distribution of high risk areas (if sufficient data)
    if len(risk_data[risk_data['flood_risk'] == 2]) > 10:
        plt.figure(figsize=(12, 8))
        plt.scatter(risk_data['longitude'], risk_data['latitude'], 
                   c=risk_data['flood_risk'], cmap='RdYlGn_r', 
                   alpha=0.6, edgecolors='none')
        plt.colorbar(label='Flood Risk (0: Low, 1: Moderate, 2: High)')
        plt.title('Geographical Distribution of Flood Risk')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig('outputs/geographical_flood_risk.png')

def build_flood_prediction_model(risk_data):
    """
    Build a machine learning model to predict flood risk based on various features.
    
    Args:
        risk_data: DataFrame with flood risk data
        
    Returns:
        Trained flood prediction model
    """
    print("Building flood prediction model...")
    
    # Features for predicting flood risk
    X = risk_data[['rainfall', 'accumulated_rainfall', 'latitude', 'longitude', 'time_index']]
    y = risk_data['flood_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nFlood prediction model evaluation:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature importance:")
    print(feature_importance)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Low', 'Moderate', 'High'],
               yticklabels=['Low', 'Moderate', 'High'])
    plt.title('Flood Risk Prediction Confusion Matrix')
    plt.xlabel('Predicted Risk')
    plt.ylabel('True Risk')
    plt.tight_layout()
    plt.savefig('outputs/flood_prediction_confusion_matrix.png')
    
    return model

def predict_flood_probability(model, new_data):
    """
    Predict flood probability for new data points.
    
    Args:
        model: Trained flood prediction model
        new_data: New data points for prediction
        
    Returns:
        DataFrame with original data and flood probability predictions
    """
    # Features needed for prediction
    required_features = ['rainfall', 'accumulated_rainfall', 'latitude', 'longitude', 'time_index']
    
    # Check if all required features are present
    for feature in required_features:
        if feature not in new_data.columns:
            raise ValueError(f"Missing required feature for prediction: {feature}")
    
    # Make prediction
    X = new_data[required_features]
    probabilities = model.predict_proba(X)
    
    # Add probabilities to the dataset
    results = new_data.copy()
    results['low_risk_prob'] = probabilities[:, 0]
    results['moderate_risk_prob'] = probabilities[:, 1] if probabilities.shape[1] > 1 else 0
    results['high_risk_prob'] = probabilities[:, 2] if probabilities.shape[1] > 2 else 0
    
    # Classify based on highest probability
    results['predicted_risk'] = model.predict(X)
    
    return results

def main():
    """Main function to execute the flood prediction workflow"""
    file_path = 'rr_ens_mean_0.1deg_reg_2011-2021_v25.0e.nc'
    
    print("Starting flood prediction analysis...")
    print("======================================")
    
    # Load data if not already available
    try:
        # Try to load existing processed data first
        if os.path.exists('processed_rainfall_data.csv'):
            print("Loading previously processed rainfall data...")
            data = pd.read_csv('processed_rainfall_data.csv')
        else:
            # Load and process the NetCDF data
            print("Processing NetCDF rainfall data...")
            nc_data = load_netcdf_metadata(file_path)
            data = extract_data_in_chunks(nc_data, sample_size=100, chunk_size=10)
            data.to_csv('processed_rainfall_data.csv', index=False)
            
        print(f"Dataset loaded: {len(data)} samples")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 1: Calculate accumulated rainfall
    flood_data = calculate_accumulated_rainfall(data)
    
    # Step 2: Calculate flood risk
    risk_data = calculate_flood_risk(flood_data)
    
    # Step 3: Visualize flood risk
    visualize_flood_risk(risk_data)
    
    # Step 4: Build flood prediction model
    model = build_flood_prediction_model(risk_data)
    
    # Step 5: Sample prediction demonstration
    print("\nDemonstrating predictions for sample data points...")
    sample_data = risk_data.sample(10)
    predictions = predict_flood_probability(model, sample_data)
    
    print("\nSample predictions:")
    for i, (_, row) in enumerate(predictions.iterrows()):
        risk_level = {0: "Low", 1: "Moderate", 2: "High"}[row['predicted_risk']]
        high_prob = row['high_risk_prob'] * 100 if 'high_risk_prob' in row else 0
        print(f"Sample {i+1}: Rainfall={row['rainfall']:.2f}mm, " 
              f"Accumulated={row['accumulated_rainfall']:.2f}mm, "
              f"Predicted Risk={risk_level} (High Risk Probability: {high_prob:.2f}%)")
    
    print("\nFlood prediction analysis complete.")
    return model, risk_data

if __name__ == "__main__":
    model, risk_data = main()
    plt.show() # all visualizations
