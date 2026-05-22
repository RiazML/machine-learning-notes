# Decision Trees: Split Metrics & Hyperparameter Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/098_decision_trees.ipynb)

Decision Trees are non-parametric models that partition the feature space recursively. At each internal node, the algorithm chooses a feature and a threshold that split the incoming data into two subsets, attempting to maximize the "purity" of the resulting nodes. This guide details the mathematical criteria for splitting and the hyperparameters used to control tree growth and prevent overfitting.

---

## 1. Mathematical Split Criteria

To select the optimal feature and threshold to split a node $S$, we measure the impurity of class distributions before and after the split.

### Gini Impurity

Gini Impurity measures the probability of misclassifying a randomly chosen element from the set if it were randomly labeled according to the distribution of labels in the subset. For a node $S$ containing classes $c \in \{1, 2, \dots, C\}$:

$$G(S) = 1 - \sum_{c=1}^C p_c^2$$

where $p_c$ is the proportion of samples belonging to class $c$ in node $S$. Gini impurity ranges from $0$ (pure node, all samples belong to one class) to $1 - 1/C$ (completely uniform distribution).

### Entropy

Entropy, derived from information theory, measures the average level of "information" or "uncertainty" in the node's class distribution:

$$H(S) = -\sum_{c=1}^C p_c \log_2(p_c)$$

If $p_c = 0$, we define $0 \log_2(0) = 0$. Entropy ranges from $0$ (pure node) to $\log_2(C)$ (uniform distribution).

### Information Gain

Information Gain ($IG$) is the expected reduction in impurity achieved by splitting the parent node $S$ into left and right child nodes ($S_L$ and $S_R$) using a feature split:

$$IG(S, f, t) = I(S) - \left( \frac{|S_L|}{|S|} I(S_L) + \frac{|S_R|}{|S|} I(S_R) \right)$$

where $I(\cdot)$ represents either Gini Impurity $G(\cdot)$ or Entropy $H(\cdot)$, and $|S|$ is the number of samples in the node. The decision tree algorithm greedily searches over all features $f$ and thresholds $t$ to find the split that maximizes $IG(S, f, t)$.

```mermaid
graph TD
    Parent["Parent Node S<br/>Impurity: I("S")"]
    Parent -->|f <= t| LeftChild["Left Child S_L<br/>Weight: |S_L|/|S|<br/>Impurity: I("S_L")"]
    Parent -->|f > t| RightChild["Right Child S_R<br/>Weight: |S_R|/|S|<br/>Impurity: I("S_R")"]
end
```

---

## 2. Controlling Tree Growth: Hyperparameters

By default, decision trees are greedily grown until all leaf nodes are pure. This behavior leads to **overfitting** (low bias, high variance), where the model fits the noise in the training data and fails to generalize to test data.

We use the following hyperparameters to control tree growth and prevent overfitting:

- **`max_depth`**: The maximum depth of the tree. Limiting depth halts splitting early, preventing the tree from growing complex boundaries around isolated noisy points.
- **`min_samples_split`**: The minimum number of samples required to split an internal node. If a node has fewer samples than this threshold, it becomes a leaf node without further splitting.
- **`min_samples_leaf`**: The minimum number of samples required to be present in a leaf node. A split will only be allowed if both the left and right child branches contain at least this number of samples.
- **`max_features`**: The number of features to consider when looking for the best split. Restricting this (e.g., to $\sqrt{p}$) introduces randomness and is a core component of Random Forests.
- **`splitter`**: Specifies the strategy used to choose the split at each node. `'best'` evaluates all possible split points, while `'random'` selects a random subset of splits, reducing overfitting and training time.
- **`criterion`**: The impurity measure used to evaluate splits (`'gini'` vs `'entropy'`). Gini is computationally cheaper as it avoids logarithmic calculations.

---

## 3. Python Verification: Greedy Split Finder from Scratch

The following script implements a greedy split finder from scratch to find the optimal split (feature and threshold) on a synthetic dataset, and asserts that the calculated split matches Scikit-Learn's `DecisionTreeClassifier` (configured as a decision stump with `max_depth=1`).

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# 1. Generate a synthetic 2D classification dataset
X, y = make_classification(n_samples=60, n_features=2, n_redundant=0, n_informative=2, random_state=42)

# 2. Fit Scikit-Learn's DecisionTreeClassifier (Decision Stump)
clf = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=42)
clf.fit(X, y)

# 3. Retrieve Sklearn's split parameters
sk_feature = clf.tree_.feature[0]
sk_threshold = clf.tree_.threshold[0]

# 4. Gini Impurity Calculator
def gini_impurity(y_subset):
    if len(y_subset) == 0:
        return 0.0
    proportions = np.bincount(y_subset) / len(y_subset)
    return 1.0 - np.sum(proportions ** 2)

# 5. Greedy Split Finder from Scratch
best_gain = -1.0
best_feature = -1
best_threshold = -1.0

n_samples, n_features = X.shape
parent_gini = gini_impurity(y)

for feature in range(n_features):
    # Sort unique values to compute split midpoints
    unique_vals = np.unique(X[:, feature])
    unique_vals = sorted(unique_vals)
    midpoints = [(unique_vals[i] + unique_vals[i+1]) / 2.0 for i in range(len(unique_vals) - 1)]

    for t in midpoints:
        left_mask = X[:, feature] <= t
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        left_gini = gini_impurity(y_left)
        right_gini = gini_impurity(y_right)

        w_left = len(y_left) / n_samples
        w_right = len(y_right) / n_samples

        # Calculate Information Gain
        gain = parent_gini - (w_left * left_gini + w_right * right_gini)

        # Match with a tiny tolerance to account for floating-point precision
        if gain > best_gain + 1e-10:
            best_gain = gain
            best_feature = feature
            best_threshold = t

# 6. Verify correctness using assertions
print(f"Sklearn Split: Feature {sk_feature}, Threshold {sk_threshold:.6f}")
print(f"Scratch Split: Feature {best_feature}, Threshold {best_threshold:.6f}")

assert sk_feature == best_feature, "Split features do not match!"
assert np.abs(sk_threshold - best_threshold) < 1e-5, "Split thresholds do not match!"
print("Assertion Passed: Custom split finder matches Scikit-Learn's split exactly!")
```

---

## 4. Next Steps

- To see how decision trees extend to continuous numerical outputs using regression split criteria, proceed to [Day 99: MSE Split Regression Tree](file:///Users/prime/Developer/ml/099_regression_trees.md).
- To review the geometric intuition behind axis-aligned partitions, refer back to [Day 97: Decision Tree Geometric Split Regions](file:///Users/prime/Developer/ml/097_decision_trees_geometric_intuition.md).
