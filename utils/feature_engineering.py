"""
Feature Engineering Utilities

This module provides helper functions for feature engineering tasks
in machine learning projects.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent', 'constant')
        columns (list): List of columns to process. If None, processes all columns with missing values
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns[df_copy.isnull().any()].tolist()
    
    for col in columns:
        if df_copy[col].dtype in ['float64', 'int64']:
            imputer = SimpleImputer(strategy=strategy)
            df_copy[col] = imputer.fit_transform(df_copy[[col]])
        else:
            # For categorical columns, use most_frequent
            imputer = SimpleImputer(strategy='most_frequent')
            df_copy[col] = imputer.fit_transform(df_copy[[col]])
    
    return df_copy


def encode_categorical_features(df, columns, method='onehot'):
    """
    Encode categorical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of categorical columns to encode
        method (str): Encoding method ('onehot' or 'label')
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    df_copy = df.copy()
    
    if method == 'onehot':
        df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
    
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    
    else:
        raise ValueError("Method must be 'onehot' or 'label'")
    
    return df_copy


def scale_features(df, columns, method='standard'):
    """
    Scale numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to scale
        method (str): Scaling method ('standard' or 'minmax')
        
    Returns:
        pd.DataFrame: Dataframe with scaled features
        scaler: Fitted scaler object
    """
    df_copy = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    
    return df_copy, scaler


def create_polynomial_features(df, columns, degree=2):
    """
    Create polynomial features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to create polynomial features from
        degree (int): Degree of polynomial features
        
    Returns:
        pd.DataFrame: Dataframe with polynomial features added
    """
    df_copy = df.copy()
    
    for col in columns:
        for d in range(2, degree + 1):
            df_copy[f'{col}_pow_{d}'] = df_copy[col] ** d
    
    return df_copy


def create_interaction_features(df, column_pairs):
    """
    Create interaction features between column pairs.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_pairs (list): List of tuples containing column pairs
        
    Returns:
        pd.DataFrame: Dataframe with interaction features added
    """
    df_copy = df.copy()
    
    for col1, col2 in column_pairs:
        df_copy[f'{col1}_x_{col2}'] = df_copy[col1] * df_copy[col2]
    
    return df_copy


def create_binned_features(df, column, bins, labels=None):
    """
    Create binned features from continuous variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to bin
        bins (int or list): Number of bins or bin edges
        labels (list): Labels for bins
        
    Returns:
        pd.DataFrame: Dataframe with binned feature added
    """
    df_copy = df.copy()
    
    df_copy[f'{column}_binned'] = pd.cut(df_copy[column], bins=bins, labels=labels)
    
    return df_copy


def create_date_features(df, date_column):
    """
    Extract date features from a datetime column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of the datetime column
        
    Returns:
        pd.DataFrame: Dataframe with extracted date features
    """
    df_copy = df.copy()
    
    # Convert to datetime if not already
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Extract features
    df_copy[f'{date_column}_year'] = df_copy[date_column].dt.year
    df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
    df_copy[f'{date_column}_day'] = df_copy[date_column].dt.day
    df_copy[f'{date_column}_dayofweek'] = df_copy[date_column].dt.dayofweek
    df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
    df_copy[f'{date_column}_is_weekend'] = df_copy[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    
    return df_copy


def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to check for outliers
        method (str): Method for outlier detection ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    df_copy = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
            df_copy = df_copy[z_scores <= threshold]
    
    print(f"Removed {len(df) - len(df_copy)} rows containing outliers")
    return df_copy


def feature_engineering_pipeline(df, config):
    """
    Apply a series of feature engineering steps based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (dict): Configuration dictionary with feature engineering steps
        
    Returns:
        pd.DataFrame: Transformed dataframe
    """
    df_transformed = df.copy()
    
    # Handle missing values
    if 'missing_values' in config:
        df_transformed = handle_missing_values(
            df_transformed,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns', None)
        )
    
    # Encode categorical features
    if 'encode_categorical' in config:
        df_transformed = encode_categorical_features(
            df_transformed,
            columns=config['encode_categorical']['columns'],
            method=config['encode_categorical'].get('method', 'onehot')
        )
    
    # Scale features
    if 'scale_features' in config:
        df_transformed, _ = scale_features(
            df_transformed,
            columns=config['scale_features']['columns'],
            method=config['scale_features'].get('method', 'standard')
        )
    
    # Create polynomial features
    if 'polynomial_features' in config:
        df_transformed = create_polynomial_features(
            df_transformed,
            columns=config['polynomial_features']['columns'],
            degree=config['polynomial_features'].get('degree', 2)
        )
    
    # Create interaction features
    if 'interaction_features' in config:
        df_transformed = create_interaction_features(
            df_transformed,
            column_pairs=config['interaction_features']['column_pairs']
        )
    
    return df_transformed
