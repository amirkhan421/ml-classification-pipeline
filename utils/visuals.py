# /utils/visuals.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_class_distribution(df, target_col):
    """Plot class distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df[target_col].value_counts().plot(kind='bar', ax=axes[0], 
                                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('Class Distribution', fontweight='bold')
    df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_ylabel('')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    sns.heatmap(numeric_df.corr(), mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    return fig

def plot_confusion_matrix(model, X_test, y_test, labels):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, 
                                          display_labels=labels, ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix', fontweight='bold')
    return fig

def plot_model_comparison(results_dict):
    """Plot model comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results_dict.keys())
    accuracies = list(results_dict.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax.bar(names, accuracies, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.8, 1.05])
    ax.axhline(y=0.92, color='red', linestyle='--', label='92% Target')
    ax.legend()
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        ax.bar(range(len(importance)), importance[indices])
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        ax.set_title('Feature Importance', fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        plt.tight_layout()
        return fig
    return None

def plot_roc_curves(model, X_test, y_test, classes):
    """Plot ROC curves for multi-class"""
    y_pred_proba = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig