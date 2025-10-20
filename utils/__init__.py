"""
Utility modules for machine learning tasks.

This package contains helper functions for:
- Data analysis and exploration
- Feature engineering
- Model training and tuning
"""

from .data_analysis import (
    load_and_describe_data,
    plot_missing_values,
    plot_distributions,
    plot_correlation_matrix,
    identify_outliers,
    summary_report
)

from .feature_engineering import (
    handle_missing_values,
    encode_categorical_features,
    scale_features,
    create_polynomial_features,
    create_interaction_features,
    create_binned_features,
    create_date_features,
    remove_outliers,
    feature_engineering_pipeline
)

from .model_training import (
    split_data,
    train_model,
    evaluate_classification,
    evaluate_regression,
    cross_validate_model,
    grid_search_tuning,
    random_search_tuning,
    save_model,
    load_model,
    compare_models
)

__all__ = [
    # Data analysis
    'load_and_describe_data',
    'plot_missing_values',
    'plot_distributions',
    'plot_correlation_matrix',
    'identify_outliers',
    'summary_report',
    
    # Feature engineering
    'handle_missing_values',
    'encode_categorical_features',
    'scale_features',
    'create_polynomial_features',
    'create_interaction_features',
    'create_binned_features',
    'create_date_features',
    'remove_outliers',
    'feature_engineering_pipeline',
    
    # Model training
    'split_data',
    'train_model',
    'evaluate_classification',
    'evaluate_regression',
    'cross_validate_model',
    'grid_search_tuning',
    'random_search_tuning',
    'save_model',
    'load_model',
    'compare_models'
]
