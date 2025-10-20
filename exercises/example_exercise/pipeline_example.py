"""
Example: Complete Machine Learning Pipeline

This script demonstrates a complete machine learning workflow including
data analysis, feature engineering, model training, and evaluation.

To use this as a template for your own exercises:
1. Replace the sample data with your actual dataset
2. Adjust the feature engineering steps based on your data
3. Choose appropriate models for your problem
4. Modify evaluation metrics as needed
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Import our utility functions
import sys
sys.path.append('../..')
from utils import (
    summary_report,
    handle_missing_values,
    encode_categorical_features,
    scale_features,
    split_data,
    train_model,
    evaluate_classification,
    compare_models
)


def main():
    """Main execution function."""
    
    print("="*60)
    print("MACHINE LEARNING PIPELINE EXAMPLE")
    print("="*60)
    
    # ========================================
    # 1. LOAD AND EXPLORE DATA
    # ========================================
    print("\n1. Loading and Exploring Data...")
    print("-"*60)
    
    # Create sample dataset (replace with your actual data loading)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Generate summary report
    summary_report(df)
    
    # ========================================
    # 2. FEATURE ENGINEERING
    # ========================================
    print("\n\n2. Feature Engineering...")
    print("-"*60)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Handle missing values (if any)
    # X = handle_missing_values(X, strategy='mean')
    
    # Scale features
    X_scaled, scaler = scale_features(X, columns=X.columns.tolist(), method='standard')
    
    print("Features scaled using StandardScaler")
    
    # ========================================
    # 3. SPLIT DATA
    # ========================================
    print("\n\n3. Splitting Data...")
    print("-"*60)
    
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # ========================================
    # 4. TRAIN AND COMPARE MODELS
    # ========================================
    print("\n\n4. Training and Comparing Models...")
    print("-"*60)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = compare_models(
        models, X_train, y_train, X_test, y_test, task='classification'
    )
    
    # ========================================
    # 5. TRAIN BEST MODEL
    # ========================================
    print("\n\n5. Training Best Model...")
    print("-"*60)
    
    # Select best model (Random Forest in this example)
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    best_model = train_model(best_model, X_train, y_train)
    
    # ========================================
    # 6. EVALUATE MODEL
    # ========================================
    print("\n\n6. Final Model Evaluation...")
    print("-"*60)
    
    metrics = evaluate_classification(best_model, X_test, y_test, average='binary')
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return best_model, metrics


if __name__ == "__main__":
    model, metrics = main()
