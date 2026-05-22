# Regression Trees: Split Criteria & Variance Reduction

Regression Trees are a variant of Decision Trees designed to predict continuous numerical targets. While Classification Trees assign labels based on majority voting in leaf nodes, Regression Trees partition the input space into regional boxes and predict a constant continuous value for each box, typically corresponding to the mean value of the targets in that region.

---

## 1. Split Selection Criteria

To determine the best feature $f$ and threshold $t$ for a split at node $S$, the algorithm seeks to minimize the variance of the target variable $y$ in the resulting child nodes. This is mathematically equivalent to maximizing **Variance Reduction** (or MSE Reduction).

### Mean Squared Error (Node Variance)

The impurity of a node $S$ containing target values $\{y_1, y_2, \dots, y_N\}$ is measured using the Mean Squared Error (MSE), which represents the variance of the target values within the node:

$$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

where $|S|$ is the number of samples in node $S$, and $\bar{y}$ is the mean target value of those samples:

$$\bar{y} = \frac{1}{|S|} \sum_{i \in S} y_i$$

### Variance Reduction (MSE Reduction)

When splitting a parent node $S$ into a left child $S_L$ and a right child $S_R$, we compute the reduction in MSE:

$$\text{VR}(S, f, t) = \text{MSE}(S) - \left( \frac{|S_L|}{|S|} \text{MSE}(S_L) + \frac{|S_R|}{|S|} \text{MSE}(S_R) \right)$$

The algorithm greedily searches over all features $f$ and thresholds $t$ to find the split that maximizes $\text{VR}(S, f, t)$.

```mermaid
graph TD
    Parent["Parent Node S<br/>Variance: MSE("S")<br/>Prediction: mean("y")"]
    Parent -->|f <= t| LeftChild["Left Child S_L<br/>Variance: MSE("S_L")<br/>Prediction: mean("y_L")"]
    Parent -->|f > t| RightChild["Right Child S_R<br/>Variance: MSE("S_R")<br/>Prediction: mean("y_R")"]
end
```

---

## 2. Leaf Value Assignment

Once a leaf node is established (e.g., when the tree reaches `max_depth` or when the number of samples in a node is less than `min_samples_split`), the constant prediction value $\hat{y}$ returned for any query point falling into that region is the mean of the training targets in that leaf node:

$$\hat{y}_{\text{leaf}} = \frac{1}{|S_{\text{leaf}}|} \sum_{i \in S_{\text{leaf}}} y_i$$

Unlike classification where we predict class labels or probability distributions, regression trees fit a piecewise constant function to the target variables.

---

## 3. Python Verification: MSE Split Search from Scratch

The following script fits a Scikit-Learn `DecisionTreeRegressor` (max depth of 1) on synthetic non-linear data, implements the MSE split search and leaf prediction calculations from scratch, and asserts exact numerical parity between the scratch calculations and Scikit-Learn.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# 1. Generate synthetic 1D regression data (noisy sine wave)
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 2. Fit Scikit-Learn's DecisionTreeRegressor stump (max_depth=1)
reg = DecisionTreeRegressor(max_depth=1, random_state=42)
reg.fit(X, y)

# 3. Retrieve Scikit-Learn's split parameters
sk_feature = reg.tree_.feature[0]
sk_threshold = reg.tree_.threshold[0]
sk_left_val = reg.tree_.value[1][0][0]
sk_right_val = reg.tree_.value[2][0][0]

# 4. MSE / Variance Calculator
def node_mse(y_subset):
    if len(y_subset) == 0:
        return 0.0
    mean = np.mean(y_subset)
    return np.mean((y_subset - mean) ** 2)

# 5. Greedy Regression Split Finder from Scratch
best_mse_reduction = -1.0
best_threshold = -1.0
best_feature = -1
best_left_val = 0.0
best_right_val = 0.0

n_samples, n_features = X.shape
parent_mse = node_mse(y)

for feature in range(n_features):
    unique_vals = np.unique(X[:, feature])
    # Compute midpoints between adjacent sorted unique values
    midpoints = [(unique_vals[i] + unique_vals[i+1]) / 2.0 for i in range(len(unique_vals) - 1)]

    for t in midpoints:
        left_mask = X[:, feature] <= t
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        left_mse = node_mse(y_left)
        right_mse = node_mse(y_right)

        w_left = len(y_left) / n_samples
        w_right = len(y_right) / n_samples

        # Calculate MSE Reduction
        mse_reduction = parent_mse - (w_left * left_mse + w_right * right_mse)

        if mse_reduction > best_mse_reduction + 1e-10:
            best_mse_reduction = mse_reduction
            best_threshold = t
            best_feature = feature
            best_left_val = np.mean(y_left)
            best_right_val = np.mean(y_right)

# 6. Verify correctness using assertions
print(f"Sklearn Split: Feature {sk_feature}, Threshold {sk_threshold:.6f}, Left Val {sk_left_val:.6f}, Right Val {sk_right_val:.6f}")
print(f"Scratch Split: Feature {best_feature}, Threshold {best_threshold:.6f}, Left Val {best_left_val:.6f}, Right Val {best_right_val:.6f}")

assert sk_feature == best_feature, "Split features do not match!"
assert np.abs(sk_threshold - best_threshold) < 1e-5, "Split thresholds do not match!"
assert np.abs(sk_left_val - best_left_val) < 1e-5, "Left predicted values do not match!"
assert np.abs(sk_right_val - best_right_val) < 1e-5, "Right predicted values do not match!"
print("Assertion Passed: Custom regression split finder matches Scikit-Learn exactly!")
```

---

## 4. Next Steps

- To visualize decision trees and trace how decision rules are applied to single samples, proceed to [Day 100: Internal Tree Traversal Rule-Path](file:///Users/prime/Developer/ml/100_awesome_decision_tree_visualization_using_dtreeviz.md).
- To review how Classification Trees optimize class distribution purity, refer back to [Day 98: Gini & Entropy Split Classifier](file:///Users/prime/Developer/ml/098_decision_trees.md).
