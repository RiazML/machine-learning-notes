# K-Means Clustering Algorithm from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/130_k-means_clustering_algorithm_from_scratch_in.ipynb)

In this study guide, we build a fully functional, object-oriented K-Means clustering algorithm from scratch in Python using only NumPy. We design the class structure to match Scikit-Learn's `KMeans` API, implement the distance calculations and convergence criteria, and verify our implementation by asserting parity against Scikit-Learn's output.

---

## Object-Oriented Lifecycle of Custom KMeans

```mermaid
graph TD
    A["Initialize CustomKMeans("n_clusters, max_iter, tol")"] --> B["fit("X")"]
    B --> C["Set initial centroids copy"]
    C --> D["Loop for iteration in max_iter"]
    D --> E["Calculate Euclidean distance matrix: D["i, k"] = ||x_i - mu_k||"]
    E --> F["Assign each sample: label_i = argmin D["i, :"]"]
    F --> G["Calculate new centroids: mu_k = mean of assigned points"]
    G --> H{"Centroid shift < tol or max_iter reached?"}
    H -- No --> I["Update centroids and continue loop"]
    I --> D
    H -- Yes --> J["Save final cluster_centers_, labels_"]
    J --> K["Compute final inertia_"]
    K --> L["predict("X_new"): Assign new points to nearest centers"]
```

---

## Mathematical Formulation

### 1. Distance Metric

The distance from point $i$ to centroid $k$ is the Euclidean distance:
$$d(x_i, \mu_k) = \sqrt{\sum_{d=1}^D (x_{i, d} - \mu_{k, d})^2} = \|x_i - \mu_k\|_2$$

### 2. Centroid Update

If $I_k$ is the set of indices of samples assigned to cluster $k$:
$$\mu_k = \frac{1}{|I_k|} \sum_{i \in I_k} x_i$$

### 3. Inertia (Within-Cluster Sum of Squares)

$$\text{Inertia} = \sum_{k=1}^K \sum_{i \in I_k} \|x_i - \mu_k\|^2$$

---

## Python Implementation and Parity Verification

The following code implements the `CustomKMeans` class, fits it on a synthetic dataset using explicit initial centroids, and compares its cluster centers, label assignments, and inertia to Scikit-Learn's `KMeans` using strict assertion checks.

```python
import numpy as np
from sklearn.cluster import KMeans

# 1. Generate a synthetic dataset
X = np.array([
    [1.0, 2.0], [1.5, 1.8], [3.0, 5.0],
    [8.0, 8.0], [8.5, 8.5], [9.0, 8.0]
])

# Manually define starting centroids to ensure identical starting points
initial_centroids = np.array([
    [1.0, 2.0],
    [8.0, 8.0]
])

# 2. Custom KMeans Implementation
class CustomKMeans:
    def __init__(self, n_clusters=2, init_centroids=None, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        # Set starting centroids
        self.cluster_centers_ = self.init_centroids.copy()

        for iteration in range(self.max_iter):
            # Compute Euclidean distances: shape (n_samples, n_clusters)
            # Using broadcasting:
            # X[:, np.newaxis, :] is shape (n_samples, 1, n_features)
            # self.cluster_centers_[np.newaxis, :, :] is shape (1, n_clusters, n_features)
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)

            # Assignment Step: Assign to closest centroid
            new_labels = np.argmin(distances, axis=1)

            # Update Step: Recalculate centroids
            new_centroids = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                assigned_points = X[new_labels == k]
                if len(assigned_points) > 0:
                    new_centroids[k] = np.mean(assigned_points, axis=0)
                else:
                    # Keep old centroid if no points are assigned
                    new_centroids[k] = self.cluster_centers_[k]

            # Check convergence (centroid shift < tolerance)
            shift = np.linalg.norm(self.cluster_centers_ - new_centroids)
            self.cluster_centers_ = new_centroids
            self.labels_ = new_labels

            if shift < self.tol:
                break

        # Compute final inertia (sum of squared distances to centroids)
        self.inertia_ = 0.0
        for k in range(self.n_clusters):
            assigned_points = X[self.labels_ == k]
            if len(assigned_points) > 0:
                self.inertia_ += np.sum((assigned_points - self.cluster_centers_[k]) ** 2)

        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

# 3. Fit Scikit-Learn KMeans and CustomKMeans
sk_kmeans = KMeans(
    n_clusters=2,
    init=initial_centroids,
    n_init=1,
    max_iter=100,
    tol=1e-4,
    random_state=42
).fit(X)

custom_kmeans = CustomKMeans(
    n_clusters=2,
    init_centroids=initial_centroids,
    max_iter=100,
    tol=1e-4
).fit(X)

# 4. Verify parity using assertions
assert np.allclose(sk_kmeans.cluster_centers_, custom_kmeans.cluster_centers_), \
    f"Centroids mismatch!\nSk-learn:\n{sk_kmeans.cluster_centers_}\nCustom:\n{custom_kmeans.cluster_centers_}"

assert np.array_equal(sk_kmeans.labels_, custom_kmeans.labels_), \
    f"Labels mismatch!\nSk-learn: {sk_kmeans.labels_}\nCustom: {custom_kmeans.labels_}"

assert np.isclose(sk_kmeans.inertia_, custom_kmeans.inertia_), \
    f"Inertia mismatch: Sk-learn={sk_kmeans.inertia_}, Custom={custom_kmeans.inertia_}"

print("Parity verification passed! Custom KMeans implementation matches Scikit-Learn exactly.")
```

---

## Previous and Next Days

- **Previous Day**: [Day 129: Silhouette Analysis & Elbow Method in Python](file:///Users/prime/Developer/ml/129_k-means_clustering_algorithm_in_python.md)
- **Next Day**: [Day 131: Agglomerative Hierarchical Clustering](file:///Users/prime/Developer/ml/131_agglomerative_hierarchical_clustering.md)
