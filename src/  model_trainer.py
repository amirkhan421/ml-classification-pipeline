from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

class ModelTrainer:
    """Class for model training and evaluation."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, random_state=42):
        """Train a Random Forest classifier."""
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_svm(self, X_train, y_train, random_state=42):
        """Train an SVM classifier."""
        svm_model = SVC(random_state=random_state, probability=True)
        svm_model.fit(X_train, y_train)
        self.models['svm'] = svm_model
        return svm_model
    
    def tune_hyperparameters(self, model, param_grid, X_train, y_train, cv=5):
        """Perform hyperparameter tuning using GridSearchCV."""
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        return grid_search
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return self.results[model_name]
    
    def cross_validate(self, model, X, y, cv=10):
        """Perform cross-validation."""
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"Cross-validation scores (mean ± std): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        return cv_scores