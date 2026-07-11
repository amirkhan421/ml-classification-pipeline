import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class DataLoader:
    """Class for loading and initial data exploration."""
    
    def __init__(self):
        self.data = None
        self.target = None
        
    def load_iris_data(self):
        """Load the Iris dataset."""
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df
    
    def load_csv_data(self, filepath):
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """Perform basic data exploration."""
        print("Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing Values:\n{df.isnull().sum()}")
        print(f"Data Types:\n{df.dtypes}")
        return df.describe()