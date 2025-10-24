#!/usr/bin/env python3
"""
Machine Learning Pipeline for Speech Emotion Recognition
Trains and evaluates multiple ML models for emotion classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EmotionClassifier:
    """Machine learning pipeline for emotion classification"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def load_processed_data(self, data_path):
        """Load processed data"""
        self.df = pd.read_csv(data_path)
        
        # Load metadata
        metadata_path = data_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded dataset with shape: {self.df.shape}")
        return self.df
    
    def prepare_data(self):
        """Prepare data for training"""
        # Get feature columns
        if self.metadata['selected_features']:
            feature_cols = self.metadata['selected_features']
        else:
            feature_cols = self.metadata['feature_columns']
        
        # Remove non-feature columns
        feature_cols = [col for col in feature_cols if col not in 
                       ['emotion', 'intensity', 'actor_id', 'sentence', 'filename', 'file_path']]
        
        self.X = self.df[feature_cols]
        self.y = self.df['emotion_encoded']
        
        print(f"Features: {self.X.shape}, Target: {self.y.shape}")
        return self.X, self.y
    
    def define_models(self):
        """Define machine learning models"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500
            )
        }
        return self.models
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best model"""
        print(f"\nPerforming hyperparameter tuning for {self.best_model_name}...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        elif self.best_model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.best_model_name == 'Neural Network':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'learning_rate': ['constant', 'adaptive'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        else:
            print("No hyperparameter tuning defined for this model")
            return self.best_model
        
        # Grid search
        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.tuned_model = grid_search.best_estimator_
        return self.tuned_model
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model from top performers"""
        print("\nCreating ensemble model...")
        
        # Get top 3 models by accuracy
        top_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
        
        ensemble_models = []
        for name, result in top_models:
            ensemble_models.append((name, result['model']))
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        self.ensemble_model = ensemble
        return ensemble
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def plot_confusion_matrix(self, cm, labels, title, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, save_path=None):
        """Plot model comparison"""
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        cv_means = [self.results[model]['cv_mean'] for model in models]
        cv_stds = [self.results[model]['cv_std'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8, 
                      yerr=cv_stds, capsize=5)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, output_dir):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual models
        for name, result in self.results.items():
            model_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.joblib")
            joblib.dump(result['model'], model_path)
        
        # Save best model
        if self.best_model:
            best_model_path = os.path.join(output_dir, "best_model.joblib")
            joblib.dump(self.best_model, best_model_path)
        
        # Save tuned model
        if hasattr(self, 'tuned_model'):
            tuned_model_path = os.path.join(output_dir, "tuned_model.joblib")
            joblib.dump(self.tuned_model, tuned_model_path)
        
        # Save ensemble model
        if hasattr(self, 'ensemble_model'):
            ensemble_model_path = os.path.join(output_dir, "ensemble_model.joblib")
            joblib.dump(self.ensemble_model, ensemble_model_path)
        
        # Save results
        results_path = os.path.join(output_dir, "model_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for name, result in self.results.items():
                json_results[name] = {
                    'accuracy': result['accuracy'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                }
            json.dump(json_results, f, indent=2)
        
        print(f"Models saved to: {output_dir}")

def main():
    """Main training pipeline"""
    
    # Paths
    data_path = "/Users/will/Projects/SpeechEmotionCS3540/features/processed_data.csv"
    output_dir = "/Users/will/Projects/SpeechEmotionCS3540/models"
    
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Load data
    print("Loading processed data...")
    df = classifier.load_processed_data(data_path)
    
    # Prepare data
    print("Preparing data...")
    X, y = classifier.prepare_data()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    print("Defining models...")
    models = classifier.define_models()
    
    # Train models
    print("Training models...")
    results = classifier.train_models(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    tuned_model = classifier.hyperparameter_tuning(X_train, y_train)
    
    # Create ensemble
    print("Creating ensemble model...")
    ensemble = classifier.create_ensemble_model(X_train, y_train)
    
    # Evaluate ensemble
    print("Evaluating ensemble model...")
    ensemble_results = classifier.evaluate_model(ensemble, X_test, y_test, "Ensemble")
    print(f"Ensemble accuracy: {ensemble_results['accuracy']:.4f}")
    
    # Create visualizations
    print("Creating visualizations...")
    classifier.plot_model_comparison(save_path=f"{output_dir}/model_comparison.png")
    
    # Plot confusion matrix for best model
    if hasattr(classifier, 'tuned_model'):
        best_model = classifier.tuned_model
        best_name = f"Tuned {classifier.best_model_name}"
    else:
        best_model = classifier.best_model
        best_name = classifier.best_model_name
    
    best_results = classifier.evaluate_model(best_model, X_test, y_test, best_name)
    
    # Get label names
    label_mapping = classifier.metadata['label_mapping']
    label_names = list(label_mapping.keys())
    
    classifier.plot_confusion_matrix(
        best_results['confusion_matrix'], 
        label_names,
        f"Confusion Matrix - {best_name}",
        save_path=f"{output_dir}/confusion_matrix.png"
    )
    
    # Save models
    print("Saving models...")
    classifier.save_models(output_dir)
    
    print("\nTraining completed!")
    print(f"Best model: {classifier.best_model_name}")
    print(f"Best accuracy: {classifier.best_score:.4f}")
    print(f"Ensemble accuracy: {ensemble_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
