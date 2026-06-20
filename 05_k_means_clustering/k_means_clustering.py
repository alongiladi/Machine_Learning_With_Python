"""
K-Means Clustering Example
===========================
Clusters Iris flowers into groups without using the class labels,
then compares the discovered clusters to the true species.

Algorithm: K-Means Clustering
Dataset:   Iris (sklearn built-in, labels used only for evaluation)
Task:      Unsupervised Clustering
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset (labels withheld during training)
    # ------------------------------------------------------------------
    iris = load_iris()
    X, y_true = iris.data, iris.target

    print("Iris Dataset (unsupervised — labels NOT used for training)")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]} ({', '.join(iris.feature_names)})")
    print()

    # ------------------------------------------------------------------
    # 2. Feature scaling
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 3. Fit K-Means with k=3 (we know there are 3 species)
    # ------------------------------------------------------------------
    model = KMeans(n_clusters=3, n_init=10, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)

    # ------------------------------------------------------------------
    # 4. Evaluate clustering quality
    # ------------------------------------------------------------------
    ari = adjusted_rand_score(y_true, cluster_labels)
    sil = silhouette_score(X_scaled, cluster_labels)

    print("Clustering Metrics:")
    print(f"  Adjusted Rand Index : {ari:.4f}  (1.0 = perfect match with true labels)")
    print(f"  Silhouette Score    : {sil:.4f}  (1.0 = perfect separation)")
    print()

    # ------------------------------------------------------------------
    # 5. Choosing the right k — Elbow method (inertia vs k)
    # ------------------------------------------------------------------
    print("Elbow Method (Within-cluster Sum of Squares):")
    print(f"  {'k':>3}  {'Inertia':>12}")
    for k in range(1, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        print(f"  {k:>3}  {km.inertia_:>12.2f}")
    print()

    # ------------------------------------------------------------------
    # 6. Compare discovered clusters to true species
    # ------------------------------------------------------------------
    print("Cluster vs True Species (rows=cluster, cols=species):")
    species = iris.target_names
    # Build a contingency-like table
    header = f"  {'':>9}  " + "  ".join(f"{s:>12}" for s in species)
    print(header)
    for c in range(3):
        row = f"  Cluster {c}  "
        mask = cluster_labels == c
        for s in range(3):
            count = int(np.sum(y_true[mask] == s))
            row += f"  {count:>12}"
        print(row)


if __name__ == "__main__":
    main()
