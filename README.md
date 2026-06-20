# Machine Learning With Python

A collection of practical machine learning examples implemented in Python using scikit-learn and common datasets. Each example is self-contained and demonstrates a different algorithm or technique.

## Examples

| # | Algorithm | Dataset | File |
|---|-----------|---------|------|
| 1 | [Linear Regression](01_linear_regression/) | Diabetes | `01_linear_regression/linear_regression.py` |
| 2 | [Logistic Regression](02_logistic_regression/) | Breast Cancer | `02_logistic_regression/logistic_regression.py` |
| 3 | [Decision Tree](03_decision_tree/) | Iris | `03_decision_tree/decision_tree.py` |
| 4 | [Random Forest](04_random_forest/) | Iris | `04_random_forest/random_forest.py` |
| 5 | [K-Means Clustering](05_k_means_clustering/) | Iris (unsupervised) | `05_k_means_clustering/k_means_clustering.py` |
| 6 | [Neural Network (MLP)](06_neural_network/) | Digits | `06_neural_network/neural_network.py` |
| 7 | [Support Vector Machine](07_svm/) | Breast Cancer | `07_svm/svm.py` |
| 8 | [Naive Bayes](08_naive_bayes/) | Wine | `08_naive_bayes/naive_bayes.py` |

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alongiladi/Machine_Learning_With_Python.git
   cd Machine_Learning_With_Python
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running an Example

Navigate to any example directory and run the Python script:

```bash
python 01_linear_regression/linear_regression.py
```

## Algorithms Overview

### 1. Linear Regression
Predicts a continuous target variable by fitting a linear equation to observed data. This example predicts disease progression using the Diabetes dataset.

### 2. Logistic Regression
A classification algorithm that models the probability of a binary outcome. This example classifies breast cancer tumors as malignant or benign.

### 3. Decision Tree
A tree-shaped model that makes decisions based on feature thresholds. This example classifies Iris flower species.

### 4. Random Forest
An ensemble of decision trees that improves accuracy and reduces overfitting. This example classifies Iris flower species.

### 5. K-Means Clustering
An unsupervised algorithm that groups data into K clusters based on similarity. This example clusters Iris flowers without using the labels.

### 6. Neural Network (MLP)
A multi-layer perceptron that learns non-linear patterns through backpropagation. This example recognizes handwritten digits.

### 7. Support Vector Machine (SVM)
Finds the optimal hyperplane that separates classes with the maximum margin. This example classifies breast cancer tumors.

### 8. Naive Bayes
A probabilistic classifier based on Bayes' theorem with strong independence assumptions. This example classifies wine cultivars using chemical features.

## Dependencies

See [requirements.txt](requirements.txt) for the full list of dependencies.