"""
Neural Network (MLP) Example
=============================
Recognizes handwritten digits (0–9) using a Multi-Layer Perceptron.

Algorithm: MLPClassifier (feed-forward neural network)
Dataset:   Digits (sklearn built-in, 8×8 pixel grayscale images)
Task:      Multi-class Classification (10 classes)
"""

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    digits = load_digits()
    X, y = digits.data, digits.target  # X: 64 pixel values per image

    print("Digits Dataset (handwritten digits 0–9)")
    print(f"  Samples       : {X.shape[0]}")
    print(f"  Features      : {X.shape[1]}  (8×8 pixel images flattened)")
    print(f"  Classes       : {sorted(set(y.tolist()))}")
    print()

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 3. Feature scaling — neural networks are sensitive to feature scale
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Train model
    #    Architecture: 64 inputs → 128 → 64 → 10 outputs
    # ------------------------------------------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy  : {accuracy:.4f}")
    print(f"Training Iters : {model.n_iter_}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # 6. Sample predictions
    # ------------------------------------------------------------------
    print("Sample Predictions (first 10 test images):")
    print(f"  {'Actual':>8}  {'Predicted':>10}  {'Correct':>8}")
    for actual, predicted in zip(y_test[:10], y_pred[:10]):
        correct = "✓" if actual == predicted else "✗"
        print(f"  {actual:>8}  {predicted:>10}  {correct:>8}")


if __name__ == "__main__":
    main()
