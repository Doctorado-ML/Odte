# Odte

![CI](https://github.com/Doctorado-ML/Odte/workflows/CI/badge.svg)
[![CodeQL](https://github.com/Doctorado-ML/Odte/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Doctorado-ML/Odte/actions/workflows/codeql-analysis.yml)
[![codecov](https://codecov.io/gh/Doctorado-ML/odte/branch/master/graph/badge.svg)](https://codecov.io/gh/Doctorado-ML/odte)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f4b5ef87584b4095b6e49aefbe594c82)](https://www.codacy.com/gh/Doctorado-ML/Odte/dashboard?utm_source=github.com&utm_medium=referral&utm_content=Doctorado-ML/Odte&utm_campaign=Badge_Grade)
[![PyPI version](https://badge.fury.io/py/Odte.svg)](https://badge.fury.io/py/Odte)
![https://img.shields.io/badge/python-3.11%2B-blue](https://img.shields.io/badge/python-3.11%2B-brightgreen)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Doctorado-ML/Odte)
[![DOI](https://zenodo.org/badge/271595804.svg)](https://zenodo.org/badge/latestdoi/271595804)

**Odte** (Oblique Decision Tree Ensemble) is a scikit-learn compatible ensemble classifier that builds forests of oblique decision trees using [STree](https://github.com/doctorado-ml/stree) as base estimators.

## Overview

Odte combines the power of ensemble learning with oblique decision trees to create a robust and flexible classification algorithm. Unlike traditional axis-aligned decision trees, oblique trees use hyperplanes at arbitrary angles, allowing for more complex decision boundaries and potentially better performance on certain datasets.

The classifier implements bootstrap aggregating (bagging) with random subspace method, similar to Random Forests, but uses oblique decision trees as base learners instead of traditional CART trees.

## Features

- **Scikit-learn Compatible**: Fully compatible with scikit-learn's API and ecosystem
- **Flexible Base Estimators**: Works with any scikit-learn classifier (default: STree)
- **Parallel Processing**: Built-in support for parallel tree construction using joblib
- **Bootstrap Aggregating**: Implements bagging with configurable sample sizes
- **Random Subspace Method**: Feature randomization for improved generalization
- **Customizable Hyperparameters**: Pass custom hyperparameters to base estimators
- **Model Inspection**: Access to tree depth, node count, and leaf statistics

## Installation

### From PyPI

```bash
pip install Odte
```

### From Source

```bash
git clone https://github.com/Doctorado-ML/Odte.git
cd Odte
pip install -e .
```

## Quick Start

```python
from odte import Odte
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the classifier
clf = Odte(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

## Usage Examples

### Basic Usage with Default Parameters

```python
from odte import Odte

# Use default STree estimator
clf = Odte(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### Custom Base Estimator

```python
from sklearn.svm import SVC
from odte import Odte

# Use SVM as base estimator
clf = Odte(
    estimator=SVC(kernel='rbf'),
    n_estimators=50,
    random_state=42
)
clf.fit(X_train, y_train)
```

### Configuring Feature and Sample Subsampling

```python
# Use sqrt of features and 80% of samples for each tree
clf = Odte(
    n_estimators=100,
    max_features='sqrt',  # or 'log2', int, float
    max_samples=0.8,       # or int for absolute number
    random_state=42
)
clf.fit(X_train, y_train)
```

### Passing Hyperparameters to Base Estimator

```python
import json
from stree import Stree

# Configure base estimator hyperparameters
hyperparams = json.dumps({
    'kernel': 'rbf',
    'max_depth': 5
})

clf = Odte(
    estimator=Stree(),
    n_estimators=100,
    be_hyperparams=hyperparams,
    random_state=42
)
clf.fit(X_train, y_train)
```

### Model Inspection

```python
# Get model statistics
nodes, leaves = clf.nodes_leaves()
depth = clf.get_depth()

print(f"Total nodes: {nodes}")
print(f"Total leaves: {leaves}")
print(f"Total depth: {depth}")
```

### Probability Predictions

```python
# Get class probabilities
probabilities = clf.predict_proba(X_test)
print(f"Class probabilities shape: {probabilities.shape}")
```

## API Reference

### Odte Class

```python
Odte(
    n_jobs=-1,
    estimator=Stree(),
    random_state=None,
    max_features=None,
    max_samples=None,
    n_estimators=100,
    be_hyperparams="{}"
)
```

**Parameters:**

- `n_jobs` (int, default=-1): Number of parallel jobs. -1 uses all available cores.
- `estimator` (BaseEstimator, default=Stree()): Base classifier to use for each tree.
- `random_state` (int, optional): Random seed for reproducibility.
- `max_features` (int, float, str, optional): Number of features to consider for each tree:
  - `None`: Use all features
  - `int`: Use this number of features
  - `float`: Use this fraction of features
  - `'auto'` or `'sqrt'`: Use sqrt(n_features)
  - `'log2'`: Use log2(n_features)
- `max_samples` (int, float, optional): Bootstrap sample size:
  - `None`: Use all samples
  - `int`: Use this number of samples
  - `float`: Use this fraction of samples
- `n_estimators` (int, default=100): Number of trees in the ensemble.
- `be_hyperparams` (str, default="{}"): JSON string of hyperparameters for base estimator.

**Methods:**

- `fit(X, y, sample_weight=None)`: Train the ensemble classifier.
- `predict(X)`: Predict class labels for samples in X.
- `predict_proba(X)`: Predict class probabilities for samples in X.
- `nodes_leaves()`: Return tuple of (total_nodes, total_leaves).
- `get_nodes()`: Return total number of nodes across all trees.
- `get_leaves()`: Return total number of leaves across all trees.
- `get_depth()`: Return total depth across all trees.
- `version()`: Return package version string.

**Attributes (after fitting):**

- `estimators_`: List of fitted base estimators.
- `subspaces_`: List of feature subsets used for each estimator.
- `classes_`: Unique class labels.
- `n_classes_`: Number of classes.
- `max_features_`: Computed maximum number of features.
- `nodes_`: Total number of nodes across all trees.
- `leaves_`: Total number of leaves across all trees.
- `depth_`: Total depth across all trees.

## Requirements

- Python >= 3.11
- scikit-learn == 1.5.2
- stree >= 1.4

See `requirements.txt` for the complete list of dependencies.

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/Doctorado-ML/Odte.git
cd Odte

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run tests with coverage
coverage run -m unittest discover -s odte.tests
coverage report

# Or use make
make test
```

### Code Quality

```bash
# Format code with black
black odte/

# Run type checking
mypy odte/

# Run linter
flake8 odte/

# Run security audit
pip-audit
```

## Documentation

Full documentation is available at [ReadTheDocs](https://odte.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and maintain the existing code style.

## Citation

If you use Odte in your research, please cite:

```bibtex
@article{Montañana:2025,
  title = {ODTE—An ensemble of multi-class SVM-based oblique decision trees},
  journal = {Expert Systems with Applications},
  volume = {273},
  pages = {126833},
  year = {2025},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2025.126833},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417425004555},
  author = {Ricardo Montañana and José A. Gámez and José M. Puerta},
  keywords = {Oblique decision trees, Supervised classification, SVM, Ensemble, Multiclass strategies}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Ricardo Montañana Gómez
- Email: ricardo.montanana@alu.uclm.es
- ORCID: [0000-0003-3242-5452](https://orcid.org/0000-0003-3242-5452)

## Links

- **GitHub**: https://github.com/doctorado-ml/odte
- **Documentation**: https://odte.readthedocs.io
- **PyPI**: https://pypi.org/project/Odte/
- **STree Project**: https://github.com/doctorado-ml/stree
