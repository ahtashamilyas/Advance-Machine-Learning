# Data Directory

This directory is for storing datasets used in the exercises.

## Structure

You can organize your data as follows:

```
data/
├── raw/              # Original, immutable data
├── processed/        # Cleaned and processed data
└── external/         # External data sources
```

## Data Files

Add your data files here. Common formats include:
- CSV files (`.csv`)
- JSON files (`.json`)
- Excel files (`.xlsx`)
- Parquet files (`.parquet`)

## Important Notes

1. **Large Files**: For large datasets, consider using Git LFS or storing them externally
2. **Privacy**: Never commit sensitive or private data to the repository
3. **Documentation**: Document your datasets with a description, source, and any preprocessing steps

## Example Datasets

To get started, you can use popular datasets from:
- [Kaggle](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Scikit-learn datasets](https://scikit-learn.org/stable/datasets.html)
- [TensorFlow datasets](https://www.tensorflow.org/datasets)
