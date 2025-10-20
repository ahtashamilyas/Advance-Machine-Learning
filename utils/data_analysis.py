"""
Data Analysis Utilities

This module provides helper functions for common data analysis tasks
in machine learning projects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_describe_data(filepath, **kwargs):
    """
    Load a dataset and provide basic information.
    
    Args:
        filepath (str): Path to the data file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath, **kwargs)
    
    print("Dataset Shape:", df.shape)
    print("\nColumn Names and Types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df


def plot_missing_values(df):
    """
    Visualize missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.plot(kind='bar')
        plt.title('Missing Values by Column')
        plt.ylabel('Count')
        plt.xlabel('Column')
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values found!")


def plot_distributions(df, columns=None, figsize=(15, 10)):
    """
    Plot distributions of numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to plot. If None, plots all numerical columns
        figsize (tuple): Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plot correlation matrix for numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        figsize (tuple): Figure size
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def identify_outliers(df, column, method='iqr', threshold=1.5):
    """
    Identify outliers in a column using IQR or Z-score method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        method (str): 'iqr' or 'zscore'
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    print(f"Found {outliers.sum()} outliers in {column} using {method} method")
    return outliers


def summary_report(df):
    """
    Generate a comprehensive summary report of the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("="*50)
    print("DATA SUMMARY REPORT")
    print("="*50)
    
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nColumn Data Types:")
    print(df.dtypes.value_counts())
    
    print("\nMissing Values Summary:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    print(missing_table[missing_table['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
    
    print("\nDuplicate Rows:", df.duplicated().sum())
    
    print("\nNumerical Columns Summary:")
    print(df.describe())
    
    print("\nCategorical Columns:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
        if df[col].nunique() < 10:
            print(df[col].value_counts())
