"""
Random Forest Example
=====================
Classifies Iris flower species using a Random Forest ensemble.

Algorithm: Random Forest Classifier
Dataset:   Iris (sklearn built-in)
Task:      Multi-class Classification
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
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
    #    n_estimators: number of trees in the forest
    # ------------------------------------------------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
    # 5. Feature importances (aggregated across all trees)
    # ------------------------------------------------------------------
    print("Feature Importances (mean decrease in impurity across all trees):")
    for name, importance in sorted(
        zip(iris.feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        bar = "█" * int(importance * 40)
        print(f"  {name:25s} {importance:.4f}  {bar}")


if __name__ == "__main__":
    main()
