# Gradient Boosting Regression - Leaf Optimization

In this guide, we dive deeper into the mathematical optimization of leaf values in Gradient Boosting Regression. We will analyze the line-search optimization step for general loss functions, compare the derivations for Squared Error (L2) loss and Absolute Error (L1) loss, and implement a custom gradient booster that directly updates the leaf values of fitted Scikit-Learn `DecisionTreeRegressor` models. We will assert parity with Scikit-Learn's `GradientBoostingRegressor` for both L2 and L1 loss types.

---

## 1. Mathematical Formulation

Let the training dataset be $\mathcal{D} = \{(x_1, y_1), \dots, (x_N, y_N)\}$.
In each boosting step $m$, a regression tree $h_m(x)$ is fit to the pseudo-residuals $r_{im}$. This tree partitions the feature space into $J_m$ disjoint terminal leaf regions:

$$R_{1m}, R_{2m}, \dots, R_{J_m m}$$

The output of the tree for any point $x$ falling in leaf $R_{jm}$ is a constant value $\gamma_{jm}$.
Instead of simply using the tree's default output, Gradient Boosting performs a **line-search optimization** to find the optimal output $\gamma_{jm}$ for each leaf region that minimizes the loss over the training samples in that leaf:

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

### Case A: Squared Error (L2) Loss

For $L(y, F(x)) = \frac{1}{2}(y - F(x))^2$, the optimization is:

$$\gamma_{jm} = \arg\min_\gamma \frac{1}{2} \sum_{x_i \in R_{jm}} (y_i - F_{m-1}(x_i) - \gamma)^2$$

Taking the derivative with respect to $\gamma$ and setting it to 0:

$$\sum_{x_i \in R_{jm}} (y_i - F_{m-1}(x_i) - \gamma) = 0$$

$$\sum_{x_i \in R_{jm}} (r_{im} - \gamma) = 0 \implies \gamma_{jm} = \frac{\sum_{x_i \in R_{jm}} r_{im}}{|R_{jm}|}$$

For L2 loss, the optimal leaf value is the mean of the residuals in that leaf.

### Case B: Absolute Error (L1) Loss

For $L(y, F(x)) = |y - F(x)|$, the optimization is:

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} |y_i - F_{m-1}(x_i) - \gamma|$$

$$\gamma_{jm} = \text{median} \left( \{ y_i - F_{m-1}(x_i) \mid x_i \in R_{jm} \} \right)$$

For L1 loss, the optimal leaf value is the median of the residuals in that leaf.

---

## 2. Process Flowchart

```mermaid
graph TD
    A["Start: Fit tree h_m on pseudo-residuals r_im"] --> B["Partition training samples into terminal leaves R_jm"]
    B --> C["For each leaf j = 1 to J_m"]
    C --> D{"Loss Function?"}
    D -->|L2 Loss| E["Compute mean of residuals in R_jm"]
    D -->|L1 Loss| F["Compute median of actual y_i - predictions F_m-1 in R_jm"]
    E --> G["Update leaf node values: tree.tree_.value["leaf_id"] = gamma_jm"]
    F --> G
    G --> H["Ensemble prediction update: F_m("x") = F_m-1("x") + lr * h_m("x")"]
    H --> I["End"]
```

---

## 3. Python Verification Script

The following script implements Gradient Boosting with custom leaf value updates for L2 and L1 losses and validates it against Scikit-Learn.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

class CustomGBRWithLeafOpt:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=2, loss='l2'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.trees = []
        self.init_val = None

    def _compute_pseudo_residuals(self, y, f_m):
        if self.loss == 'l2':
            return y - f_m
        elif self.loss == 'l1':
            return np.sign(y - f_m)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def fit(self, X, y):
        # Step 1: Initialize
        if self.loss == 'l2':
            self.init_val = np.mean(y)
        elif self.loss == 'l1':
            self.init_val = np.median(y)

        f_m = np.full(X.shape[0], self.init_val, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Step 2: Compute pseudo-residuals
            r_im = self._compute_pseudo_residuals(y, f_m)

            # Fit base tree to pseudo-residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, r_im)

            # Step 3: Leaf-wise line search optimization
            # Find which leaf node each training sample falls into
            leaf_indices = tree.apply(X)

            # Update each unique leaf value
            unique_leaves = np.unique(leaf_indices)
            for leaf in unique_leaves:
                # Find indices of samples belonging to this leaf
                in_leaf = (leaf_indices == leaf)

                # Optimize value according to the loss function
                if self.loss == 'l2':
                    gamma = np.mean(y[in_leaf] - f_m[in_leaf])
                elif self.loss == 'l1':
                    res_in_leaf = y[in_leaf] - f_m[in_leaf]
                    gamma = np.sort(res_in_leaf)[(len(res_in_leaf) - 1) // 2]

                # Write to the tree structure directly
                # value array is of shape (node_count, outputs, values)
                tree.tree_.value[leaf, 0, 0] = gamma

            # Step 4: Update prediction
            f_m += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_val, dtype=np.float64)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Generate toy dataset
X, y = make_regression(n_samples=50, n_features=3, noise=0.5, random_state=42)

# --- Test Parity for L2 Loss ---
custom_l2 = CustomGBRWithLeafOpt(n_estimators=5, learning_rate=0.1, max_depth=2, loss='l2')
custom_l2.fit(X, y)
sklearn_l2 = GradientBoostingRegressor(
    n_estimators=5, learning_rate=0.1, max_depth=2, loss='squared_error', random_state=42
)
sklearn_l2.fit(X, y)
assert np.allclose(custom_l2.predict(X), sklearn_l2.predict(X)), "L2 Predictions Mismatch!"

# --- Test Parity for L1 Loss ---
custom_l1 = CustomGBRWithLeafOpt(n_estimators=5, learning_rate=0.1, max_depth=2, loss='l1')
custom_l1.fit(X, y)
sklearn_l1 = GradientBoostingRegressor(
    n_estimators=5, learning_rate=0.1, max_depth=2, loss='absolute_error', random_state=42
)
sklearn_l1.fit(X, y)
assert np.allclose(custom_l1.predict(X), sklearn_l1.predict(X)), "L1 Predictions Mismatch!"

print("Parity verification passed! Custom leaf value optimization matches Scikit-Learn for both L2 and L1 loss functions.")
```

---

## Navigation Links

- **Previous**: [Day 120: Gradient Boosting Regression (Squared Loss)](file:///Users/prime/Developer/ml/120_gradient_boosting_explained.md)
- **Next**: [Day 122: Gradient Boosting for Classification](file:///Users/prime/Developer/ml/122_gradient_boosting_for_classification.md)
