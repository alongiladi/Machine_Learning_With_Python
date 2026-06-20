"""
Decision Tree Example
=====================
Classifies Iris flower species using a Decision Tree.

Algorithm: Decision Tree Classifier (CART)
Dataset:   Iris (sklearn built-in)
Task:      Multi-class Classification
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Iris Dataset")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]} ({', '.join(iris.feature_names)})")
    print(f"  Classes : {list(iris.target_names)}")
    print()

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 3. Train model
    #    max_depth limits tree size to avoid overfitting
    # ------------------------------------------------------------------
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"Test Accuracy      : {accuracy:.4f}")
    print(f"5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    print("Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=iris.target_names)
    )

    # ------------------------------------------------------------------
    # 5. Print tree structure
    # ------------------------------------------------------------------
    print("Decision Tree Structure (depth ≤ 4):")
    tree_text = export_text(model, feature_names=iris.feature_names)
    print(tree_text)

    # ------------------------------------------------------------------
    # 6. Feature importances
    # ------------------------------------------------------------------
    print("Feature Importances:")
    for name, importance in sorted(
        zip(iris.feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        bar = "█" * int(importance * 40)
        print(f"  {name:25s} {importance:.4f}  {bar}")


if __name__ == "__main__":
    main()
