#!/usr/bin/env python3
"""
Machine Learning Classification Pipeline
Main execution script
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluate import ModelEvaluator

def main():
    """Main function to run the ML pipeline."""
    
    print("="*80)
    print("MACHINE LEARNING CLASSIFICATION PIPELINE")
    print("="*80)
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Load data
    print("\n1. Loading Data...")
    df = data_loader.load_iris_data()
    print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare features and target
    X = df.iloc[:, :-2]  # Features
    y = df['target']     # Target variable
    
    # Split data
    print("\n2. Splitting Data...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create preprocessing pipeline
    print("\n3. Building Preprocessing Pipeline...")
    numeric_features = X.columns.tolist()
    preprocessor.create_preprocessing_pipeline(numeric_features)
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Feature selection
    print("\n4. Feature Engineering...")
    X_train_selected, feature_scores = feature_engineer.select_k_best(
        X_train_processed, y_train, k='all'
    )
    X_test_selected = feature_engineer.selector.transform(X_test_processed)
    
    # Train models
    print("\n5. Training Models...")
    
    # Random Forest
    print("\n5.1 Training Random Forest...")
    rf_model = trainer.train_random_forest(X_train_selected, y_train)
    rf_results = trainer.evaluate_model(rf_model, X_test_selected, y_test, "Random Forest")
    
    # SVM
    print("\n5.2 Training SVM...")
    svm_model = trainer.train_svm(X_train_selected, y_train)
    svm_results = trainer.evaluate_model(svm_model, X_test_selected, y_test, "SVM")
    
    # Hyperparameter Tuning
    print("\n6. Hyperparameter Tuning...")
    
    # Random Forest tuning
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_grid = trainer.tune_hyperparameters(
        rf_model, rf_param_grid, X_train_selected, y_train
    )
    
    # SVM tuning
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    svm_grid = trainer.tune_hyperparameters(
        svm_model, svm_param_grid, X_train_selected, y_train
    )
    
    # Evaluate tuned models
    print("\n7. Final Evaluation...")
    
    rf_tuned = rf_grid.best_estimator_
    rf_tuned_results = trainer.evaluate_model(
        rf_tuned, X_test_selected, y_test, "Random Forest (Tuned)"
    )
    
    svm_tuned = svm_grid.best_estimator_
    svm_tuned_results = trainer.evaluate_model(
        svm_tuned, X_test_selected, y_test, "SVM (Tuned)"
    )
    
    # Cross-validation
    print("\n8. Cross-Validation...")
    rf_cv_scores = trainer.cross_validate(rf_tuned, X_train_selected, y_train)
    svm_cv_scores = trainer.cross_validate(svm_tuned, X_train_selected, y_train)
    
    # Select best model
    best_model = rf_tuned if rf_tuned_results['accuracy'] > svm_tuned_results['accuracy'] else svm_tuned
    best_model_name = "Random Forest" if rf_tuned_results['accuracy'] > svm_tuned_results['accuracy'] else "SVM"
    
    # Save best model
    print("\n9. Saving Best Model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Best model ({best_model_name}) saved to 'models/best_model.pkl'")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {accuracy_score(y_test, best_model.predict(X_test_selected)):.4f}")
    print(f"Cross-Validation Score: {rf_cv_scores.mean() if best_model_name == 'Random Forest' else svm_cv_scores.mean():.4f}")
    print("="*80)

if __name__ == "__main__":
    main()