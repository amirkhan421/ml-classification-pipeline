import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    """Class for data preprocessing and cleaning."""
    
    def __init__(self):
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state, stratify=y)
    
    def create_preprocessing_pipeline(self, numeric_features):
        """Create preprocessing pipeline for numerical features."""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])
        
        return self.preprocessor
    
    def fit_transform(self, X):
        """Fit and transform the data."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not created. Call create_preprocessing_pipeline first.")
        return self.preprocessor.fit_transform(X)
    
    def transform(self, X):
        """Transform the data."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not created. Call create_preprocessing_pipeline first.")
        return self.preprocessor.transform(X)