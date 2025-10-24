#!/usr/bin/env python3
"""
Feature and Label Preparation for Speech Emotion Recognition
Prepares extracted features and labels for machine learning model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_explore_data():
    """
    Load the extracted features and explore the dataset
    This function loads the crema_d_features.csv file and provides basic data exploration
    """
    # Load the extracted features (use cleaned version if available)
    data_path = 'features/crema_d_features_cleaned.csv'
    if not os.path.exists(data_path):
        data_path = 'features/crema_d_features.csv'
        print("Using original dataset (cleaned version not found)")
    else:
        print("Using cleaned dataset")
    
    df = pd.read_csv(data_path)
    
    print("=== Dataset Overview ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 4}")  # Subtract non-feature columns
    print(f"Samples: {df.shape[0]}")
    
    # Display basic info about the dataset
    print("\n=== Column Information ===")
    print("Feature columns:", df.shape[1] - 4)  # emotion, intensity, actor_id, file_name
    print("Non-feature columns:", ['emotion', 'intensity', 'actor_id', 'file_name'])
    
    # Display emotion distribution
    print("\n=== Emotion Distribution ===")
    print(df['emotion'].value_counts())
    
    # Display intensity distribution  
    print("\n=== Intensity Distribution ===")
    print(df['intensity'].value_counts())
    
    # Display actor distribution
    print(f"\n=== Actor Information ===")
    print(f"Number of unique actors: {df['actor_id'].nunique()}")
    print(f"Actors per emotion:")
    print(df.groupby('emotion')['actor_id'].nunique())
    
    return df

def prepare_features_and_labels(df):
    """
    Prepare features (X) and labels (y) for machine learning
    This function separates the numerical features from the metadata columns
    """
    # Separate features from metadata
    feature_columns = [col for col in df.columns if col not in 
                      ['emotion', 'intensity', 'actor_id', 'file_name']]
    
    X = df[feature_columns]  # All numerical features
    y = df['emotion']       # Emotion labels
    
    print(f"\n=== Feature Preparation ===")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target labels shape: {y.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Check for missing values
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in labels: {y.isnull().sum()}")
    
    # Data should already be cleaned, but verify
    print("\n=== Data Verification ===")
    print("Verifying data types...")
    
    # Check if any feature columns are still object type (string arrays)
    object_columns = [col for col in feature_columns if X[col].dtype == 'object']
    
    if object_columns:
        print(f"⚠️  Found {len(object_columns)} columns with object data type - cleaning...")
        # Create a copy to avoid SettingWithCopyWarning
        X_clean = X.copy()
        
        # Convert all feature columns to numeric, handling string arrays
        for col in feature_columns:
            if X_clean[col].dtype == 'object':
                # Handle string representations of arrays like '[112.34714674]'
                X_clean[col] = X_clean[col].astype(str).str.strip('[]')
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce').fillna(0)
        
        # Ensure all features are numeric
        X_clean = X_clean.astype(float)
        print(f"Features cleaned. Final shape: {X_clean.shape}")
    else:
        print("✅ All feature columns are already numeric!")
        X_clean = X.astype(float)
    
    print(f"Final data types: {X_clean.dtypes.value_counts()}")
    
    return X_clean, y, feature_columns

def encode_labels(y):
    """
    Encode emotion labels to numerical values
    This function converts string emotion labels to integers for ML algorithms
    """
    # Create label encoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Display label mapping
    print("\n=== Label Encoding ===")
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Emotion to number mapping:")
    for emotion, number in label_mapping.items():
        print(f"  {emotion}: {number}")
    
    return y_encoded, le, label_mapping

def split_and_scale_data(X, y_encoded):
    """
    Split data into train/test sets and scale features
    This function creates training and testing splits and normalizes the features
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n=== Data Splitting ===")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train/Test ratio: {X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2f}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n=== Feature Scaling ===")
    print("Features scaled using StandardScaler")
    print(f"Training features mean: {X_train_scaled.mean():.6f}")
    print(f"Training features std: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def visualize_data_distribution(df, y_encoded, label_mapping):
    """
    Create visualizations to understand the data distribution
    This function creates plots to visualize emotion distribution and feature relationships
    """
    print("\n=== Creating Data Visualizations ===")
    
    # Create emotion distribution plot
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Emotion distribution
    plt.subplot(1, 2, 1)
    emotion_counts = df['emotion'].value_counts()
    plt.bar(emotion_counts.index, emotion_counts.values)
    plt.title('Emotion Distribution in Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Plot 2: Intensity distribution
    plt.subplot(1, 2, 2)
    intensity_counts = df['intensity'].value_counts()
    plt.bar(intensity_counts.index, intensity_counts.values)
    plt.title('Intensity Distribution in Dataset')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Data distribution plots saved as 'data_distribution.png'")

def define_model_architecture():
    """
    Define the machine learning models to be used
    This function outlines the models that should be implemented for emotion classification
    """
    print("\n=== Model Architecture Planning ===")
    print("Recommended models for emotion classification:")
    print("1. Random Forest - Good baseline, handles feature importance")
    print("2. Support Vector Machine (SVM) - Good for high-dimensional data")
    print("3. Logistic Regression - Simple, interpretable baseline")
    print("4. Neural Network - Can capture complex patterns")
    print("5. Gradient Boosting - Often performs well on tabular data")
    
    print("\nModel selection considerations:")
    print("- Start with Random Forest for baseline performance")
    print("- Use SVM with RBF kernel for non-linear relationships")
    print("- Try ensemble methods (Voting, Bagging) for improved performance")
    print("- Consider hyperparameter tuning for best models")

def model_training_workflow():
    """
    Outline the complete model training workflow
    This function provides a roadmap for implementing the ML pipeline
    """
    print("\n=== Model Training Workflow ===")
    print("1. Load and explore the dataset")
    print("2. Prepare features and encode labels")
    print("3. Split data into train/test sets")
    print("4. Scale features appropriately")
    print("5. Train multiple baseline models")
    print("6. Evaluate model performance using cross-validation")
    print("7. Select best performing model(s)")
    print("8. Perform hyperparameter tuning")
    print("9. Create ensemble models if beneficial")
    print("10. Final evaluation on test set")
    print("11. Save trained models for deployment")

def evaluation_metrics_planning():
    """
    Plan the evaluation metrics and visualization strategy
    This function outlines how to evaluate model performance
    """
    print("\n=== Evaluation Strategy ===")
    print("Key metrics to track:")
    print("- Accuracy: Overall correctness")
    print("- Precision: Per-class precision scores")
    print("- Recall: Per-class recall scores")
    print("- F1-Score: Harmonic mean of precision and recall")
    print("- Confusion Matrix: Detailed error analysis")
    
    print("\nVisualizations to create:")
    print("- Confusion matrix heatmap")
    print("- ROC curves for each emotion class")
    print("- Feature importance plots")
    print("- Model comparison bar charts")
    print("- Learning curves for model selection")

def main():
    """
    Main function - orchestrates the feature and label preparation
    This is the entry point that loads data and prepares features and labels for ML training
    """
    print("=== Speech Emotion Recognition - Feature and Label Preparation ===")
    
    # Step 1: Load and explore the dataset
    df = load_and_explore_data()
    
    # Step 2: Prepare features and labels
    X, y, feature_columns = prepare_features_and_labels(df)
    
    # Step 3: Encode labels
    y_encoded, label_encoder, label_mapping = encode_labels(y)
    
    # Step 4: Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y_encoded)
    
    # Step 5: Create visualizations
    visualize_data_distribution(df, y_encoded, label_mapping)
    
    # Step 6: Plan model architecture
    define_model_architecture()
    
    # Step 7: Outline training workflow
    model_training_workflow()
    
    # Step 8: Plan evaluation strategy
    evaluation_metrics_planning()
    
    print("\n=== Data Preparation Complete ===")
    print("The dataset is now ready for machine learning model training.")
    print("Next steps:")
    print("1. Implement the model training functions")
    print("2. Add cross-validation and hyperparameter tuning")
    print("3. Create evaluation and visualization functions")
    print("4. Save trained models for future use")
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler

if __name__ == "__main__":
    # Run the data preparation pipeline
    X_train, X_test, y_train, y_test, label_encoder, scaler = main()
