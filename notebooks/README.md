# Jupyter Notebooks

This directory contains Jupyter notebooks for interactive data analysis and model development.

## Getting Started

1. Install Jupyter (included in requirements.txt):
```bash
pip install jupyter
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Or use JupyterLab:
```bash
jupyter lab
```

## Notebook Organization

Organize your notebooks by exercise or topic:

```
notebooks/
├── 01_data_exploration.ipynb
├── 02_feature_engineering.ipynb
├── 03_model_training.ipynb
├── exercise_1/
│   └── analysis.ipynb
└── exercise_2/
    └── analysis.ipynb
```

## Notebook Best Practices

1. **Structure**: Use clear sections with markdown headers
2. **Documentation**: Add explanatory text and comments
3. **Reproducibility**: Set random seeds and document dependencies
4. **Clean Output**: Clear output before committing to reduce file size
5. **Naming**: Use descriptive names with numbers for ordering

## Template Notebook

A template notebook is provided (`ml_exercise_template.ipynb`) that includes:
- Data loading and exploration
- Feature engineering
- Model training
- Evaluation and visualization

Copy this template to start new exercises.
