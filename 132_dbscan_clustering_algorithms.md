# DBSCAN Clustering Algorithms

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a powerful density-based clustering algorithm. Unlike K-Means, which assumes clusters are spherical and requires specifying the number of clusters $K$ in advance, DBSCAN groups points together based on their density in the feature space and can identify clusters of arbitrary shapes while naturally handling outliers (noise).

---

## Why DBSCAN? K-Means Limitations

1. **Pre-specifying $K$**: K-Means requires specifying $K$ beforehand, which is hard to estimate in high dimensions. Techniques like the Elbow Method or Silhouette Analysis can be ambiguous.
2. **Sensitivity to Outliers**: K-Means update step shifts cluster centroids based on mean positions, meaning outliers pull centroids away from their dense cluster cores.
3. **Assumes Spherical Clusters**: K-Means uses Euclidean distance from centroids, failing on non-spherical clusters (e.g., concentric circles, moons, or crescent shapes).

---

## DBSCAN Point Classification

```mermaid
graph TD
    subgraph Point Classification
        A["Check Epsilon Neighborhood of Point p: N_eps("p")"] --> B{Is |N_eps("p")| >= MinPts?}
        B -- Yes --> C["Label p as CORE Point"]
        B -- No --> D{Does N_eps("p") contain at least one CORE Point?}
        D -- Yes --> E["Label p as BORDER Point"]
        D -- No --> F["Label p as NOISE/Outlier Point"]
    end
```

---

## Mathematical Formulations: Density and Reachability

Let $X = \{x_1, x_2, \dots, x_N\}$ be the dataset. The algorithm relies on two parameters: $\epsilon$ (epsilon, the radius of the neighborhood) and $\text{MinPts}$ (minimum number of points within the neighborhood).

### 1. $\epsilon$-Neighborhood

The $\epsilon$-neighborhood of a point $x \in X$, denoted by $N_{\epsilon}(x)$, is defined as:
$$N_{\epsilon}(x) = \{y \in X \mid d(x, y) \le \epsilon\}$$
where $d(x, y)$ is the Euclidean distance:
$$d(x, y) = \sqrt{\sum_{i=1}^D (x_i - y_i)^2}$$

### 2. Core Point

A point $x$ is a core point if its neighborhood contains at least $\text{MinPts}$ points:
$$|N_{\epsilon}(x)| \ge \text{MinPts}$$

### 3. Border Point

A point $x$ is a border point if it is not a core point but lies within the $\epsilon$-neighborhood of some core point $c$:
$$|N_{\epsilon}(x)| < \text{MinPts} \quad \text{and} \quad \exists c \in X \text{ s.t. } |N_{\epsilon}(c)| \ge \text{MinPts} \text{ and } x \in N_{\epsilon}(c)$$

### 4. Noise Point (Outlier)

A point $x$ is a noise point if it is neither a core point nor a border point.

### 5. Density-Reachability & Connectivity

- **Direct Density-Reachability**: A point $p$ is directly density-reachable from a point $q$ if $p \in N_{\epsilon}(q)$ and $q$ is a core point.
- **Density-Reachability**: A point $p$ is density-reachable from $q$ if there is a sequence of points $p_1, p_2, \dots, p_n$ with $p_1 = q$ and $p_n = p$ such that each $p_{i+1}$ is directly density-reachable from $p_i$.
- **Density-Connectivity**: Two points $p$ and $q$ are density-connected if there exists a point $o$ such that both $p$ and $q$ are density-reachable from $o$.

---

## DBSCAN Algorithm Flowchart

```mermaid
graph TD
    A["Start: Initialize all points as unvisited"] --> B["Loop through each unvisited point p"]
    B --> C{"Is p a CORE point?"}
    C -- Yes --> D["Create new cluster C_id"]
    D --> E["Add p to C_id and initialize queue with N_eps("p")"]
    E --> F["Process queue BFS: for each neighbor n"]
    F --> G{"Is n unvisited?"}
    G -- Yes --> H["Set n as visited, assign to C_id"]
    H --> I{"Is n a CORE point?"}
    I -- Yes --> J["Add N_eps("n") to queue"]
    I -- No --> K["Continue"]
    G -- No --> L{"Is n labeled as noise?"}
    L -- Yes --> M["Relabel n to C_id as Border point"]
    L -- No --> K
    F --> N{"Queue empty?"}
    N -- No --> F
    N -- Yes --> O["Increment C_id, return to loop"]
    C -- No --> P["Temporarily label p as noise"]
    P --> B
    O --> Q{"All points visited?"}
    Q -- No --> B
    Q -- Yes --> R["Finalize clusters and leave noise as -1"]
```

---

## Python Implementation and Parity Verification

The following code implements DBSCAN from scratch and asserts partition connectivity and noise-assignment parity against Scikit-Learn's `DBSCAN` implementation.

```python
import numpy as np
from sklearn.cluster import DBSCAN

# 1. Generate toy dataset
X = np.array([
    [1.0, 1.0], [1.1, 1.0], [1.0, 1.1],
    [5.0, 5.0], [5.1, 5.0],
    [9.0, 9.0] # outlier
])

eps = 1.5
min_samples = 3

# 2. Fit Scikit-Learn DBSCAN
sk_db = DBSCAN(eps=eps, min_samples=min_samples)
sk_labels = sk_db.fit_predict(X)

# 3. Custom DBSCAN Implementation
def custom_dbscan(X, eps, min_samples):
    n_samples = X.shape[0]
    labels = np.full(n_samples, -2) # -2: unvisited, -1: noise, >=0: cluster ID

    # Precompute epsilon neighborhoods
    dists = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
    neighborhoods = [np.where(dists[i] <= eps)[0] for i in range(n_samples)]

    # Identify Core Points
    is_core = np.array([len(neighborhoods[i]) >= min_samples for i in range(n_samples)])

    cluster_id = 0
    for i in range(n_samples):
        if labels[i] != -2:
            continue

        if not is_core[i]:
            labels[i] = -1 # temporarily label as noise
            continue

        # Found a new core point - start a new cluster!
        labels[i] = cluster_id

        # Expand cluster using BFS/queue
        queue = list(neighborhoods[i])
        idx = 0
        while idx < len(queue):
            neighbor = queue[idx]
            idx += 1

            # If labeled noise, it's a border point (belongs to current cluster)
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id

            # If unvisited
            elif labels[neighbor] == -2:
                labels[neighbor] = cluster_id
                # If neighbor is also a core point, add its neighbors to queue
                if is_core[neighbor]:
                    for n_pt in neighborhoods[neighbor]:
                        if labels[n_pt] == -2 or labels[n_pt] == -1:
                            if n_pt not in queue:
                                queue.append(n_pt)

        cluster_id += 1

    return labels

custom_labels = custom_dbscan(X, eps, min_samples)

# 4. Parity Verification (checking cluster connectivity and noise matching)
def get_connectivity_matrix(labels):
    # Noise (-1) should not connect to anything, even to other noise points
    conn = labels[:, np.newaxis] == labels[np.newaxis, :]
    noise_mask = (labels == -1)
    conn[noise_mask, :] = False
    conn[:, noise_mask] = False
    return conn

sk_conn = get_connectivity_matrix(sk_labels)
custom_conn = get_connectivity_matrix(custom_labels)

assert np.array_equal(sk_conn, custom_conn), \
    f"Cluster connectivity mismatch!\nSk-learn: {sk_labels}\nCustom: {custom_labels}"

assert np.array_equal(sk_labels == -1, custom_labels == -1), \
    f"Noise assignment mismatch!\nSk-learn: {sk_labels}\nCustom: {custom_labels}"

print("Parity verification passed! Custom DBSCAN matches Scikit-Learn exactly.")
```

---

## Previous and Next Days

- **Previous Day**: [Day 131: Agglomerative Hierarchical Clustering](file:///Users/prime/Developer/ml/131_agglomerative_hierarchical_clustering.md)
- **Next Day**: [Day 133: Imbalanced Data in Machine Learning](file:///Users/prime/Developer/ml/133_imbalanced_data_in_machine_learning.md)
