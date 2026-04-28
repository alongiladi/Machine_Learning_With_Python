"""
Linear Regression Example
=========================
Predicts disease progression using the Diabetes dataset.

Algorithm: Ordinary Least Squares Linear Regression
Dataset:   Diabetes (sklearn built-in)
Task:      Regression — predict a continuous target value
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target  # y: disease progression measure

    print("Diabetes Dataset")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]} ({', '.join(diabetes.feature_names)})")
    print()

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # 3. Feature scaling — Linear Regression converges faster when
    #    features are on a similar scale.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Model Coefficients:")
    for name, coef in zip(diabetes.feature_names, model.coef_):
        print(f"  {name:12s}: {coef:+.4f}")
    print(f"  Intercept   : {model.intercept_:+.4f}")
    print()
    print("Test-Set Performance:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    # ------------------------------------------------------------------
    # 6. Sample predictions
    # ------------------------------------------------------------------
    print()
    print("Sample Predictions (first 5 test samples):")
    print(f"  {'Actual':>10}  {'Predicted':>10}")
    for actual, predicted in zip(y_test[:5], y_pred[:5]):
        print(f"  {actual:>10.4f}  {predicted:>10.4f}")


if __name__ == "__main__":
    main()
