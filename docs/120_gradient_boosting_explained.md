# Gradient Boosting Regression (Squared Loss)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/120_gradient_boosting_explained.ipynb)

This guide introduces the core intuition and mathematics of Gradient Boosting for regression problems using the Mean Squared Error (MSE) loss function. We will walk through the step-by-step process of sequential additive modeling, derive the pseudo-residuals, construct a step-by-step process flowchart, and build a custom implementation from scratch that matches Scikit-Learn's `GradientBoostingRegressor` output.

---

## 1. Mathematical Formulation (Squared Error Loss)

Gradient Boosting is an additive ensemble method that builds models sequentially:

$$F_m(x) = F_{m-1}(x) + \nu h_m(x)$$

where $h_m(x)$ is a weak learner (usually a decision tree) and $\nu \in (0, 1]$ is the learning rate (shrinkage).

For regression with Squared Error Loss:

$$L(y, F(x)) = \frac{1}{2}(y - F(x))^2$$

### Step 1: Initialize the Base Predictor

We initialize the ensemble with a constant prediction that minimizes the loss over the entire dataset:

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^N L(y_i, \gamma) = \arg\min_\gamma \frac{1}{2}\sum_{i=1}^N (y_i - \gamma)^2$$

Taking the derivative with respect to $\gamma$ and setting it to 0 yields:

$$F_0(x) = \bar{y} \quad (\text{the mean of the target variable } y)$$

### Step 2: Compute Pseudo-Residuals

For each iteration $m = 1 \dots M$, we compute the negative gradient of the loss function with respect to the current predictions. These are called **pseudo-residuals**:

$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$

For Squared Error Loss:

$$\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} = -(y_i - F(x_i))$$

$$r_{im} = y_i - F_{m-1}(x_i)$$

Thus, for squared error loss, the pseudo-residuals are exactly the standard prediction residuals.

### Step 3: Train Weak Learner on Residuals

We fit a regression tree $h_m(x)$ using the features $X$ to predict the pseudo-residuals $r_{im}$.

### Step 4: Update the Ensemble Prediction

We update the model by adding the scaled predictions of the new tree:

$$F_m(x) = F_{m-1}(x) + \nu h_m(x)$$

---

## 2. Process Flowchart

```mermaid
graph TD
    A["Start: Dataset X, y"] --> B["Initialize F_0("x") = Mean("y")"]
    B --> C["For m = 1 to M"]
    C --> D["Compute residuals: r_im = y_i - F_m-1("x_i")"]
    D --> E["Fit Regression Tree h_m("x") to targets r_im"]
    E --> F["Update Ensemble: F_m("x") = F_m-1("x") + lr * h_m("x")"]
    F --> G{"m == M?"}
    G -- No --> C
    G -- Yes --> H["Output final ensemble prediction F_M_x"]
    H --> I["End"]
```

---

## 3. Python Verification Script

The following script implements Gradient Boosting from scratch for MSE loss and asserts mathematical prediction parity against Scikit-Learn's `GradientBoostingRegressor`.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_val = None

    def fit(self, X, y):
        # Step 1: Initialize with mean
        self.init_val = np.mean(y)
        f_m = np.full(X.shape[0], self.init_val, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Step 2: Compute pseudo-residuals
            residuals = y - f_m

            # Step 3: Fit regression tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)

            # Step 4: Update ensemble
            f_m += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_val, dtype=np.float64)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Generate synthetic regression dataset
X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=42)

# Custom Gradient Boosting Regressor
custom_gbr = CustomGradientBoostingRegressor(n_estimators=5, learning_rate=0.1, max_depth=2)
custom_gbr.fit(X, y)
custom_preds = custom_gbr.predict(X)

# Scikit-Learn Gradient Boosting Regressor (squared error loss)
sklearn_gbr = GradientBoostingRegressor(
    n_estimators=5, learning_rate=0.1, max_depth=2, loss='squared_error', random_state=42
)
sklearn_gbr.fit(X, y)
sklearn_preds = sklearn_gbr.predict(X)

# Assert that predictions are identical
assert np.allclose(custom_preds, sklearn_preds), "Predictions mismatch!"
print("Parity verification passed! Custom Gradient Boosting Regressor matches Scikit-Learn output exactly.")
```

---

## Navigation Links

- **Previous**: [Day 119: Bagging vs Boosting](file:///Users/prime/Developer/ml/119_bagging_vs_boosting.md)
- **Next**: [Day 121: Gradient Boosting Regression Part 2](file:///Users/prime/Developer/ml/121_gradient_boosting_regression_part_2.md)
