from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class FeatureEngineer:
    """Class for feature engineering and selection."""
    
    def __init__(self):
        self.selector = None
        self.selected_features = None
        
    def select_k_best(self, X, y, k=10):
        """Select K best features using ANOVA F-value."""
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        # Get selected feature scores
        scores = pd.DataFrame({
            'Feature': range(X.shape[1]),
            'Score': self.selector.scores_
        }).sort_values('Score', ascending=False)
        
        self.selected_features = scores.head(k)['Feature'].tolist()
        return X_selected, scores
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction."""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        return X_pca, pca
    
    def create_polynomial_features(self, X, degree=2):
        """Create polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        return X_poly, poly