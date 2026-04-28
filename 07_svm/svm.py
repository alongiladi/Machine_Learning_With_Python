"""
Support Vector Machine (SVM) Example
======================================
Classifies breast cancer tumors as malignant or benign.

Algorithm: Support Vector Classifier (SVC) with RBF kernel
Dataset:   Breast Cancer Wisconsin (sklearn built-in)
Task:      Binary Classification
"""

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = malignant, 1 = benign

    print("Breast Cancer Wisconsin Dataset")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes : {list(data.target_names)}")
    print()

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 3. Feature scaling — SVMs are sensitive to feature scale
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Hyperparameter search with cross-validation
    #    C     : regularisation strength (higher = less regularisation)
    #    gamma : RBF kernel bandwidth ('scale' = 1 / (n_features * X.var()))
    # ------------------------------------------------------------------
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    }
    grid_search = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best Parameters : {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    print()

    # ------------------------------------------------------------------
    # 5. Evaluate best model on hold-out test set
    # ------------------------------------------------------------------
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy   : {accuracy:.4f}")
    print()
    print("Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=data.target_names)
    )

    # ------------------------------------------------------------------
    # 6. Support vectors summary
    # ------------------------------------------------------------------
    print(f"Number of Support Vectors: {best_model.n_support_.sum()}")
    for cls, n_sv in zip(data.target_names, best_model.n_support_):
        print(f"  {cls}: {n_sv}")


if __name__ == "__main__":
    main()
