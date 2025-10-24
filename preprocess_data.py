#!/usr/bin/env python3
"""
Data Preprocessing Utilities for Speech Emotion Recognition
Handles feature scaling, encoding, and dataset preparation for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handle data preprocessing for speech emotion recognition"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        
    def load_data(self, features_path):
        """Load the extracted features"""
        self.df = pd.read_csv(features_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        return self.df
    
    def clean_data(self):
        """Clean the dataset by removing invalid entries"""
        print("Cleaning data...")
        
        # Remove rows with missing target labels
        if 'emotion' in self.df.columns:
            self.df = self.df.dropna(subset=['emotion'])
        
        # Remove rows with all NaN features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df = self.df.dropna(subset=numeric_cols)
        
        # Remove duplicate files
        if 'filename' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['filename'])
        
        print(f"After cleaning: {self.df.shape}")
        return self.df
    
    def encode_labels(self, target_column='emotion'):
        """Encode categorical labels"""
        if target_column in self.df.columns:
            self.df[f'{target_column}_encoded'] = self.label_encoder.fit_transform(self.df[target_column])
            self.label_mapping = dict(zip(
                self.label_encoder.classes_, 
                self.label_encoder.transform(self.label_encoder.classes_)
            ))
            print(f"Label mapping: {self.label_mapping}")
        return self.df
    
    def scale_features(self, feature_columns=None):
        """Scale numerical features"""
        if feature_columns is None:
            # Get all numeric columns except target and metadata
            exclude_cols = ['emotion', 'intensity', 'actor_id', 'sentence', 'filename', 'file_path']
            feature_columns = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_cols and not col.endswith('_encoded')]
        
        # Scale features
        self.df[feature_columns] = self.scaler.fit_transform(self.df[feature_columns])
        self.feature_columns = feature_columns
        print(f"Scaled {len(feature_columns)} features")
        return self.df
    
    def select_features(self, target_column='emotion_encoded', method='f_classif', k=50):
        """Select most relevant features"""
        if target_column not in self.df.columns:
            print(f"Target column {target_column} not found!")
            return self.df
        
        # Get feature columns
        feature_cols = [col for col in self.feature_columns if col != target_column]
        X = self.df[feature_cols]
        y = self.df[target_column]
        
        # Select features
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_cols)))
        else:
            raise ValueError("Method must be 'f_classif' or 'mutual_info'")
        
        X_selected = selector.fit_transform(X, y)
        self.selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        self.feature_selector = selector
        
        print(f"Selected {len(self.selected_features)} features using {method}")
        return self.selected_features
    
    def create_train_test_split(self, target_column='emotion_encoded', test_size=0.2, random_state=42):
        """Create train-test split"""
        if self.selected_features is None:
            feature_cols = self.feature_columns
        else:
            feature_cols = self.selected_features
        
        X = self.df[feature_cols]
        y = self.df[target_column]
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def create_cv_splits(self, n_splits=5, random_state=42):
        """Create cross-validation splits"""
        if self.selected_features is None:
            feature_cols = self.feature_columns
        else:
            feature_cols = self.selected_features
        
        X = self.df[feature_cols]
        y = self.df['emotion_encoded']
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.cv_splits = list(cv.split(X, y))
        
        print(f"Created {n_splits} CV splits")
        return self.cv_splits
    
    def analyze_feature_importance(self, target_column='emotion_encoded'):
        """Analyze feature importance and correlations"""
        if target_column not in self.df.columns:
            print(f"Target column {target_column} not found!")
            return
        
        # Get feature columns
        feature_cols = [col for col in self.feature_columns if col != target_column]
        
        # Calculate correlations with target
        correlations = []
        for col in feature_cols:
            corr, _ = pearsonr(self.df[col], self.df[target_column])
            correlations.append((col, abs(corr)))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 20 features by correlation with target:")
        for i, (feature, corr) in enumerate(correlations[:20]):
            print(f"{i+1:2d}. {feature:30s} {corr:.4f}")
        
        return correlations
    
    def visualize_data_distribution(self, save_path=None):
        """Create visualizations of the dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Emotion distribution
        if 'emotion' in self.df.columns:
            emotion_counts = self.df['emotion'].value_counts()
            axes[0, 0].bar(emotion_counts.index, emotion_counts.values)
            axes[0, 0].set_title('Emotion Distribution')
            axes[0, 0].set_xlabel('Emotion')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Intensity distribution
        if 'intensity' in self.df.columns:
            intensity_counts = self.df['intensity'].value_counts()
            axes[0, 1].bar(intensity_counts.index, intensity_counts.values)
            axes[0, 1].set_title('Intensity Distribution')
            axes[0, 1].set_xlabel('Intensity')
            axes[0, 1].set_ylabel('Count')
        
        # Actor distribution
        if 'actor_id' in self.df.columns:
            actor_counts = self.df['actor_id'].value_counts()
            axes[1, 0].hist(actor_counts.values, bins=20)
            axes[1, 0].set_title('Recordings per Actor')
            axes[1, 0].set_xlabel('Number of Recordings')
            axes[1, 0].set_ylabel('Number of Actors')
        
        # Duration distribution
        if 'duration' in self.df.columns:
            axes[1, 1].hist(self.df['duration'], bins=30)
            axes[1, 1].set_title('Audio Duration Distribution')
            axes[1, 1].set_xlabel('Duration (seconds)')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_correlation_heatmap(self, top_n=20, save_path=None):
        """Create correlation heatmap of top features"""
        if self.selected_features is None:
            # Get top features by correlation
            correlations = self.analyze_feature_importance()
            top_features = [feat for feat, _ in correlations[:top_n]]
        else:
            top_features = self.selected_features[:top_n]
        
        # Calculate correlation matrix
        corr_matrix = self.df[top_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'Feature Correlation Matrix (Top {len(top_features)} Features)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_processed_data(self, output_path):
        """Save processed dataset"""
        # Save the processed dataframe
        self.df.to_csv(output_path, index=False)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'label_mapping': self.label_mapping,
            'dataset_shape': self.df.shape
        }
        
        import json
        with open(output_path.replace('.csv', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Processed data saved to: {output_path}")

def main():
    """Main preprocessing pipeline"""
    
    # Paths
    features_path = "/Users/will/Projects/SpeechEmotionCS3540/features/crema_d_features.csv"
    output_path = "/Users/will/Projects/SpeechEmotionCS3540/features/processed_data.csv"
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and process data
    print("Loading data...")
    df = preprocessor.load_data(features_path)
    
    print("Cleaning data...")
    df = preprocessor.clean_data()
    
    print("Encoding labels...")
    df = preprocessor.encode_labels()
    
    print("Scaling features...")
    df = preprocessor.scale_features()
    
    print("Selecting features...")
    selected_features = preprocessor.select_features(method='f_classif', k=50)
    
    print("Creating train-test split...")
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split()
    
    print("Creating CV splits...")
    cv_splits = preprocessor.create_cv_splits()
    
    print("Analyzing feature importance...")
    correlations = preprocessor.analyze_feature_importance()
    
    print("Creating visualizations...")
    preprocessor.visualize_data_distribution(save_path="/Users/will/Projects/SpeechEmotionCS3540/features/data_distribution.png")
    preprocessor.create_feature_correlation_heatmap(save_path="/Users/will/Projects/SpeechEmotionCS3540/features/feature_correlation.png")
    
    print("Saving processed data...")
    preprocessor.save_processed_data(output_path)
    
    print("\nPreprocessing completed!")
    print(f"Final dataset shape: {preprocessor.df.shape}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

if __name__ == "__main__":
    main()
