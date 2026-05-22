# Non-Linear Boundaries: Polynomial Logistic Regression

Standard Logistic Regression models a linear decision boundary: $w^T x + b = 0$. However, many real-world classification problems are non-linearly separable (e.g., concentric circles, spirals, XOR patterns). To model these distributions without resorting to complex neural networks, we map inputs into a higher-dimensional space using **Polynomial Feature expansion**, enabling a linear classifier to learn curved decision boundaries.

---

## 1. Feature Expansion Mechanics

Let's assume a 2D input vector $x = [x_1, x_2]^T$. A standard logistic model uses the linear combination:
$$z = w_1 x_1 + w_2 x_2 + b$$

This only allows a straight line boundary.

### Polynomial Feature Mapping

To allow curved boundaries, we define a mapping function $\phi(x)$ that expands our input features to degree $d = 2$:
$$\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]^T$$

Applying logistic regression to this expanded feature space yields the decision boundary equation:
$$w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2 + b = 0$$

While the boundary is **linear with respect to the weights** $w$, it represents a **quadratic curve (circle, ellipse, parabola, or hyperbola)** in the original 2D input space.

```mermaid
flowchart TD
    In["Input Space: X = [x₁, x₂] (Non-linear separation)"] --> Map["Polynomial Mapping: φ("X") = [x₁, x₂, x₁², x₁x₂, x₂²]"]
    Map --> Fit["Logistic Regression fits hyperplane in 5D space"]
    Fit --> Project["Projection to 2D space results in curved boundary"]
```

#### Example: Separating Concentric Circles

Consider a dataset where class 1 lies inside a circle of radius $r$ and class 0 lies outside. The separator is:
$$x_1^2 + x_2^2 = r^2 \implies x_1^2 + x_2^2 - r^2 = 0$$

By mapping inputs to $[x_1^2, x_2^2]$ and setting weights to $w_1 = 1$, $w_2 = 1$, and bias $b = -r^2$, logistic regression classifies this dataset perfectly.

---

## 2. Python Implementation: Classifying Concentric Circles

The following runnable Python script generates a non-linear "concentric circles" dataset, fits both a standard (linear) Logistic Regression model and a Polynomial Logistic Regression pipeline, and asserts the superior performance of the polynomial model.

```python
import numpy as np
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Generate Concentric Circles Dataset
# Class 0 represents outer circle, Class 1 represents inner circle
X, y = make_circles(n_samples=200, noise=0.08, factor=0.4, random_seed=42) if 'random_seed' in make_circles.__code__.co_varnames else make_circles(n_samples=200, noise=0.08, factor=0.4, random_state=42)

# 2. Fit standard Linear Logistic Regression
linear_model = LogisticRegression(penalty=None)
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
acc_linear = accuracy_score(y, y_pred_linear)

# 3. Fit Polynomial Logistic Regression (Degree = 2) using a Pipeline
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('logreg', LogisticRegression(penalty=None))
])
poly_pipeline.fit(X, y)
y_pred_poly = poly_pipeline.predict(X)
acc_poly = accuracy_score(y, y_pred_poly)

# 4. Print results & verify coefficients
print("=== Linear vs. Polynomial Logistic Regression ===")
print(f"Linear Model Accuracy:     {acc_linear * 100.0:.2f}%")
print(f"Polynomial Model Accuracy: {acc_poly * 100.0:.2f}%")

# Retrieve coefficients from the pipeline to demonstrate weights mapping
poly_features = poly_pipeline.named_steps['poly']
logreg_model = poly_pipeline.named_steps['logreg']
feature_names = poly_features.get_feature_names_out(['x1', 'x2'])
weights = logreg_model.coef_[0]

print("\nLearned Polynomial Weights:")
for name, weight in zip(feature_names, weights):
    print(f"Feature '{name}': {weight:.4f}")
print(f"Intercept (b):     {logreg_model.intercept_[0]:.4f}")

# Assert that polynomial model successfully captures the circular relationship
assert acc_poly > 0.95, "Polynomial model failed to achieve high accuracy on circular data"
assert acc_linear < 0.60, "Linear model unexpectedly fit circular data"
print("\n[SUCCESS] Polynomial feature expansion successfully projected non-linear data for linear classification!")
```

---

- **Next Topic**: [081_logistic_regression_hyperparameters.md](file:///Users/prime/Developer/ml/081_logistic_regression_hyperparameters.md) - Hyperparameter Tuning and Regularization in Logistic Regression.
