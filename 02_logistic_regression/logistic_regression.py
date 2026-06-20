"""
Logistic Regression Example
============================
Classifies breast cancer tumors as malignant or benign.

Algorithm: Logistic Regression (binary classification)
Dataset:   Breast Cancer Wisconsin (sklearn built-in)
Task:      Binary Classification
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
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
    # 3. Feature scaling
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  {'':12}  {'Pred Mal':>10}  {'Pred Ben':>10}")
    print(f"  {'Actual Mal':12}  {cm[0, 0]:>10}  {cm[0, 1]:>10}")
    print(f"  {'Actual Ben':12}  {cm[1, 0]:>10}  {cm[1, 1]:>10}")
    print()
    print("Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=data.target_names)
    )


if __name__ == "__main__":
    main()
