"""
Model Training and Tuning Utilities

This module provides helper functions for model training, evaluation,
and hyperparameter tuning in machine learning projects.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import joblib


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size (float): Proportion of test set
        random_state (int): Random seed
        stratify: If not None, data is split in a stratified fashion
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def train_model(model, X_train, y_train):
    """
    Train a machine learning model.
    
    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X_test, y_test, average='binary'):
    """
    Evaluate a classification model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        average (str): Averaging method for multi-class classification
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average=average, zero_division=0)
    }
    
    print("Classification Metrics:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return metrics


def evaluate_regression(model, X_test, y_test):
    """
    Evaluate a regression model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    print("Regression Metrics:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """
    Perform cross-validation on a model.
    
    Args:
        model: Scikit-learn model instance
        X: Feature matrix
        y: Target vector
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        
    Returns:
        dict: Cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    results = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }
    
    print(f"Cross-Validation Results ({cv} folds):")
    print("="*50)
    print(f"Scores: {scores}")
    print(f"Mean {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
    
    return results


def grid_search_tuning(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        model: Scikit-learn model instance
        param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings
        X_train: Training features
        y_train: Training labels
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        
    Returns:
        GridSearchCV: Fitted GridSearchCV object
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Grid Search Results:")
    print("="*50)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_:.4f}")
    
    return grid_search


def random_search_tuning(model, param_distributions, X_train, y_train, 
                        n_iter=10, cv=5, scoring='accuracy', random_state=42):
    """
    Perform random search for hyperparameter tuning.
    
    Args:
        model: Scikit-learn model instance
        param_distributions (dict): Dictionary with parameters names and distributions
        X_train: Training features
        y_train: Training labels
        n_iter (int): Number of parameter settings sampled
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        random_state (int): Random seed
        
    Returns:
        RandomizedSearchCV: Fitted RandomizedSearchCV object
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    random_search.fit(X_train, y_train)
    
    print("Random Search Results:")
    print("="*50)
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_:.4f}")
    
    return random_search


def save_model(model, filepath):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def compare_models(models, X_train, y_train, X_test, y_test, task='classification'):
    """
    Compare multiple models and return their performance metrics.
    
    Args:
        models (dict): Dictionary of model names and model instances
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        task (str): 'classification' or 'regression'
        
    Returns:
        pd.DataFrame: Comparison results
    """
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        if task == 'classification':
            y_pred = model.predict(X_test)
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            y_pred = model.predict(X_test)
            metrics = {
                'Model': name,
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df
