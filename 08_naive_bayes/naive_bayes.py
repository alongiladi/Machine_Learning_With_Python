"""
Naive Bayes Example
====================
Classifies wines into three cultivar categories using chemical features.

Algorithm: Gaussian Naive Bayes
Dataset:   Wine (sklearn built-in)
Task:      Multi-class Classification
"""

from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    wine = load_wine()
    X, y = wine.data, wine.target

    print("Wine Dataset")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]} ({', '.join(wine.feature_names)})")
    print(f"  Classes : {list(wine.target_names)}")
    print()

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 3. Feature scaling
    #    GaussianNB models each feature as a Gaussian distribution,
    #    so scaling does not change the model but keeps outputs comparable.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    print(f"Test Accuracy      : {accuracy:.4f}")
    print(f"5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    # ------------------------------------------------------------------
    # 6. Per-class prior probabilities learned by the model
    # ------------------------------------------------------------------
    print("Class Prior Probabilities:")
    for cls, prior in zip(wine.target_names, model.class_prior_):
        print(f"  {cls}: {prior:.4f}")

    # ------------------------------------------------------------------
    # 7. Sample predictions with probabilities
    # ------------------------------------------------------------------
    print()
    print("Sample Predictions (first 5 test samples):")
    print(f"  {'Actual':>12}  {'Predicted':>12}  Probability")
    probs = model.predict_proba(X_test_scaled[:5])
    for actual, predicted, prob in zip(y_test[:5], y_pred[:5], probs):
        prob_str = "  ".join(f"{wine.target_names[i]}={p:.2f}" for i, p in enumerate(prob))
        correct = "✓" if actual == predicted else "✗"
        print(f"  {wine.target_names[actual]:>12}  {wine.target_names[predicted]:>12}  {correct}  {prob_str}")


if __name__ == "__main__":
    main()
