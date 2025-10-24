#!/usr/bin/env python3
"""
Clean Dataset Script
Creates permanently cleaned versions of the extracted features
"""

import pandas as pd
import numpy as np
import os

def clean_feature_data(input_path, output_path):
    """
    Clean the feature data by converting string arrays to proper numerical values
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Original data types: {df.dtypes.value_counts()}")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Get feature columns (exclude metadata columns)
    feature_columns = [col for col in df.columns if col not in 
                      ['emotion', 'intensity', 'actor_id', 'file_name', 'file_path', 'filename', 'sentence']]
    
    print(f"Cleaning {len(feature_columns)} feature columns...")
    
    # Clean each feature column
    for col in feature_columns:
        if df_clean[col].dtype == 'object':
            # Handle string representations of arrays like '[112.34714674]'
            # First, strip brackets and convert to float
            df_clean[col] = df_clean[col].astype(str).str.strip('[]')
            # Convert to numeric, coercing errors to NaN, then fill with 0
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Ensure all feature columns are numeric
    for col in feature_columns:
        df_clean[col] = df_clean[col].astype(float)
    
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Cleaned data types: {df_clean.dtypes.value_counts()}")
    
    # Check for any remaining string arrays
    string_arrays_found = 0
    for col in feature_columns:
        if df_clean[col].dtype == 'object':
            string_arrays_found += 1
            print(f"Warning: Column {col} still contains object data")
    
    if string_arrays_found == 0:
        print("✅ All feature columns successfully converted to numeric!")
    else:
        print(f"⚠️  {string_arrays_found} columns still contain non-numeric data")
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")
    
    return df_clean

def main():
    """
    Clean all the feature files
    """
    print("=== Cleaning Dataset Files ===")
    
    # Define input and output paths
    files_to_clean = [
        ('features/crema_d_features.csv', 'features/crema_d_features_cleaned.csv'),
        ('features/raw_features.csv', 'features/raw_features_cleaned.csv'),
        ('features/metadata.csv', 'features/metadata_cleaned.csv')
    ]
    
    for input_path, output_path in files_to_clean:
        if os.path.exists(input_path):
            print(f"\n--- Cleaning {input_path} ---")
            clean_feature_data(input_path, output_path)
        else:
            print(f"⚠️  File not found: {input_path}")
    
    print("\n=== Dataset Cleaning Complete ===")
    print("Cleaned files created:")
    for _, output_path in files_to_clean:
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"  ✅ {output_path} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()
