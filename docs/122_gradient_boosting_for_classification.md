# Gradient Boosting for Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/122_gradient_boosting_for_classification.ipynb)

In this guide, we explore Gradient Boosting for Binary Classification. We will cover the mathematical formulation of Log-Loss, the Newton-Raphson approximation used to optimize leaf values, and implement a custom gradient boosting classifier from scratch. Finally, we will verify its outputs and prediction probabilities against Scikit-Learn's `GradientBoostingClassifier` to assert exact mathematical parity.

---

## 1. Mathematical Formulation

Let the training dataset be $\mathcal{D} = \{(x_1, y_1), \dots, (x_N, y_N)\}$ where $y_i \in \{0, 1\}$.
In binary classification, the ensemble prediction $F(x)$ represents the predicted log-odds:

$$F(x) = \ln\left( \frac{p}{1 - p} \right)$$

where $p = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$ is the probability of the positive class.

### Log-Loss Function

To train the classifier, we minimize the negative log-likelihood (log-loss):

$$L(y, F(x)) = -y \ln p - (1 - y) \ln(1 - p)$$

Substituting $p = \frac{1}{1 + e^{-F(x)}}$ gives:

$$L(y, F(x)) = -y F(x) + \ln\left(1 + e^{F(x)}\right)$$

### Step 1: Initial Prediction

The initial constant model $F_0(x)$ is the log-odds of the positive class in the training data:

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^N L(y_i, \gamma) = \ln\left( \frac{\sum_{i=1}^N y_i}{N - \sum_{i=1}^N y_i} \right)$$

### Step 2: Compute Pseudo-Residuals

For each iteration $m = 1 \dots M$, the pseudo-residuals $r_{im}$ are the negative gradients of the loss function with respect to the current prediction $F_{m-1}(x_i)$:

$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$

Differentiating the loss:

$$\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} = -y_i + \frac{e^{F_{m-1}(x_i)}}{1 + e^{F_{m-1}(x_i)}} = p_{i, m-1} - y_i$$

$$r_{im} = y_i - p_{i, m-1}$$

### Step 3: Leaf-wise Line Search Optimization

We fit a regression tree $h_m(x)$ to the pseudo-residuals $r_{im}$, partitioning the space into leaf regions $R_{jm}$ for $j = 1 \dots J_m$.
The optimal leaf value $\gamma_{jm}$ minimizes the loss:

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

Because this equation cannot be solved analytically, we apply a single Newton-Raphson step:

$$\gamma_{jm} \approx \frac{-\sum_{i \in R_{jm}} g_i}{\sum_{i \in R_{jm}} h_i}$$

where:

- $g_i = \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} = p_{i, m-1} - y_i$
- $h_i = \frac{\partial^2 L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)^2} = p_{i, m-1} (1 - p_{i, m-1})$

Thus, the optimal update for each leaf is:

$$\gamma_{jm} = \frac{\sum_{x_i \in R_{jm}} (y_i - p_{i, m-1})}{\sum_{x_i \in R_{jm}} p_{i, m-1} (1 - p_{i, m-1})}$$

---

## 2. Process Flowchart

```mermaid
graph TD
    A["Start: Calculate Initial Log-Odds F_0"] --> B["Set current ensemble F_m-1 = F_0"]
    B --> C["Compute probabilities p_i = sigmoid("F_m-1")"]
    C --> D["Calculate residuals r_im = y_i - p_i"]
    D --> E["Fit Decision Tree h_m to r_im"]
    E --> F["For each leaf R_jm in h_m"]
    F --> G["Compute optimal leaf value gamma_jm using Newton-Raphson formula"]
    G --> H["Update tree.tree_.value with gamma_jm"]
    H --> I["Update ensemble: F_m = F_m-1 + learning_rate * h_m"]
    I --> J{"More iterations?"}
    J -->|Yes| C
    J -->|No| K["End"]
```

---

## 3. Python Verification Script

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier

class CustomGBClassifier:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_val = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        # Step 1: Initialize raw predictions F_0 as log-odds
        p_mean = np.mean(y)
        self.init_val = np.log(p_mean / (1.0 - p_mean))

        f_m = np.full(X.shape[0], self.init_val, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Step 2: Compute predicted probabilities and pseudo-residuals
            p = self._sigmoid(f_m)
            r_im = y - p

            # Fit base tree to pseudo-residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, r_im)

            # Step 3: Newton-Raphson leaf optimization
            leaf_indices = tree.apply(X)
            unique_leaves = np.unique(leaf_indices)

            for leaf in unique_leaves:
                in_leaf = (leaf_indices == leaf)
                numerator = np.sum(y[in_leaf] - p[in_leaf])
                denominator = np.sum(p[in_leaf] * (1.0 - p[in_leaf]))

                # Newton-Raphson update
                gamma = numerator / denominator if denominator > 0 else 0.0

                # Overwrite internal leaf value
                tree.tree_.value[leaf, 0, 0] = gamma

            # Step 4: Update prediction
            f_m += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict_raw(self, X):
        raw_predictions = np.full(X.shape[0], self.init_val, dtype=np.float64)
        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X)
        return raw_predictions

    def predict_proba(self, X):
        raw = self.predict_raw(X)
        p = self._sigmoid(raw)
        return np.column_stack((1.0 - p, p))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

# Generate synthetic binary classification dataset
X, y = make_classification(n_samples=60, n_features=4, n_informative=3, n_redundant=1, random_state=42)

# Fit Custom Classifier
custom = CustomGBClassifier(n_estimators=5, learning_rate=0.1, max_depth=2)
custom.fit(X, y)

# Fit Scikit-Learn Classifier
sklearn_model = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=2, random_state=42)
sklearn_model.fit(X, y)

# Parity Assertions
custom_raw = custom.predict_raw(X)
sklearn_raw = sklearn_model.decision_function(X)
assert np.allclose(custom_raw, sklearn_raw), "Raw prediction (decision function) mismatch!"

custom_proba = custom.predict_proba(X)
sklearn_proba = sklearn_model.predict_proba(X)
assert np.allclose(custom_proba, sklearn_proba), "Probability predictions mismatch!"

custom_preds = custom.predict(X)
sklearn_preds = sklearn_model.predict(X)
assert np.array_equal(custom_preds, sklearn_preds), "Class predictions mismatch!"

print("Parity verification passed! Custom Gradient Boosting Binary Classifier matches Scikit-Learn exactly.")
```

---

## Navigation Links

- **Previous**: [Day 121: Gradient Boosting Regression - Leaf Optimization](file:///Users/prime/Developer/ml/121_gradient_boosting_regression_part_2.md)
- **Next**: [Day 123: Introduction to XGBoost](file:///Users/prime/Developer/ml/123_introduction_to_xgboost.md)
