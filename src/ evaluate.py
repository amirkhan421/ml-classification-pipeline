import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd

class ModelEvaluator:
    """Class for detailed model evaluation and visualization."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix'):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    def plot_roc_curves(self, y_test, y_pred_proba, classes):
        """Plot ROC curves for multi-class classification."""
        y_test_bin = label_binarize(y_test, classes=range(len(classes)))
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                    label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), 
                      [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Importance', fontsize=14, fontweight='bold')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.tight_layout()
            plt.show()
            
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
    
    def compare_models(self, results_dict):
        """Compare multiple models' performance."""
        df = pd.DataFrame(results_dict).T
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar')
        plt.title('Model Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return df