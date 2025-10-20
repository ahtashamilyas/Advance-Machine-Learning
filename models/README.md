# Models Directory

This directory is for storing trained machine learning models.

## Structure

Organize your saved models by exercise or project:

```
models/
├── exercise_1/
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── exercise_2/
│   └── best_model.pkl
└── README.md
```

## Saving Models

Use the utility functions to save models:

```python
from utils import save_model

# Save a trained model
save_model(trained_model, 'models/my_exercise/model.pkl')
```

## Loading Models

Load saved models for inference:

```python
from utils import load_model

# Load a saved model
model = load_model('models/my_exercise/model.pkl')
```

## Model Formats

Supported formats:
- `.pkl` - Scikit-learn models (using joblib)
- `.h5` - Keras/TensorFlow models
- `.pt` or `.pth` - PyTorch models
- `.onnx` - ONNX format for cross-framework compatibility

## Best Practices

1. **Versioning**: Keep track of model versions
2. **Metadata**: Save model metadata (hyperparameters, training date, performance metrics)
3. **Size**: Consider compression for large models
4. **Documentation**: Document model architecture and training details
