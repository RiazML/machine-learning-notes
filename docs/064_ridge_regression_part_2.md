# Ridge Regression: Closed-form Mathematical Matrix Derivation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/064_ridge_regression_part_2.ipynb)

In simple linear regression, the normal equation solves for the parameter vector $\theta$. In this guide, we derive the closed-form matrix solution for Ridge Regression and demonstrate how to implement it from scratch using two mathematically equivalent methods, comparing them directly to Scikit-Learn's implementation.

---

## 1. Mathematical Derivation

Let $X$ be the $N \times (p + 1)$ design matrix (including a column of ones for the bias/intercept), $Y$ be the $N \times 1$ target vector, and $\theta$ be the $(p + 1) \times 1$ parameter vector:

$$\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_p \end{bmatrix}$$

We define the regularized objective function as:
$$J(\theta) = (Y - X\theta)^T (Y - X\theta) + \lambda \theta^T I_0 \theta$$

Where:

- $\lambda \ge 0$ is the regularization strength.
- $I_0$ is a modified $(p+1) \times (p+1)$ identity matrix that prevents penalizing the intercept $\theta_0$:
  $$I_0 = \begin{bmatrix} 0 & 0 & 0 & \cdots & 0 \\ 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \end{bmatrix}$$

### Step-by-Step Gradient Derivation

First, expand the objective function:
$$J(\theta) = (Y^T - \theta^T X^T)(Y - X\theta) + \lambda \theta^T I_0 \theta$$
$$J(\theta) = Y^TY - Y^T X\theta - \theta^T X^T Y + \theta^T X^T X \theta + \lambda \theta^T I_0 \theta$$

Since $Y^T X\theta$ is a scalar, it equals its transpose $\theta^T X^T Y$. Thus:
$$J(\theta) = Y^TY - 2\theta^T X^T Y + \theta^T X^T X \theta + \lambda \theta^T I_0 \theta$$

We take the gradient with respect to the vector $\theta$:
$$\nabla_\theta J(\theta) = \frac{\partial}{\partial \theta} \left( Y^TY - 2\theta^T X^T Y + \theta^T X^T X \theta + \lambda \theta^T I_0 \theta \right)$$

Using matrix calculus rules:

- $\frac{\partial}{\partial \theta} (\theta^T A) = A$
- $\frac{\partial}{\partial \theta} (\theta^T A \theta) = (A + A^T)\theta$ (and if $A$ is symmetric, $2A\theta$)

We obtain:
$$\nabla_\theta J(\theta) = -2X^TY + 2X^TX\theta + 2\lambda I_0 \theta$$

Setting the gradient to the zero vector to find the minimum:
$$-2X^TY + 2(X^TX + \lambda I_0)\theta = 0$$
$$(X^TX + \lambda I_0)\theta = X^TY$$

Assuming $(X^TX + \lambda I_0)$ is invertible, we solve for $\theta$:
$$\theta = (X^TX + \lambda I_0)^{-1} X^TY$$

---

## 2. Invertibility and Numerical Stability

In OLS, the matrix $X^TX$ is singular (not invertible) if:

1. We have more features than samples ($p > N$).
2. Features are perfectly collinear (linear dependency).

By adding $\lambda I_0$ (where $\lambda > 0$), we add positive values along the diagonal of $X^TX$. This shifts the eigenvalues of the matrix, making the system strictly positive definite and guaranteeing that $(X^TX + \lambda I_0)$ is non-singular and invertible.

---

## 3. Python Implementation: Scratch vs. Scikit-Learn

Scikit-Learn implements Ridge regression by first **centering** the features $X$ and target $Y$ (subtracting their respective column means), solving the OLS equation on the centered variables with a standard identity matrix $I$, and then computing the intercept.

Below, we implement both the **$I_0$ matrix projection** and the **centering method** from scratch, and verify that both match Scikit-Learn's `Ridge` coefficients exactly.

```python
import numpy as np
from sklearn.linear_model import Ridge

# 1. Generate Synthetic Dataset
np.random.seed(42)
N = 80
p = 5
X = np.random.normal(loc=0.0, scale=1.5, size=(N, p))
y = 2.0 + 1.5 * X[:, 0] - 3.0 * X[:, 1] + 0.8 * X[:, 2] + np.random.normal(0, 0.5, size=N)

lambda_val = 15.0

# 2. Method A: Matrix Projection (Using I_0)
# Append column of ones to X for the intercept
X_bias = np.hstack([np.ones((N, 1)), X])
I_0 = np.eye(p + 1)
I_0[0, 0] = 0.0  # Do not penalize the intercept

theta_projection = np.linalg.inv(X_bias.T @ X_bias + lambda_val * I_0) @ X_bias.T @ y
intercept_proj = theta_projection[0]
coef_proj = theta_projection[1:]

# 3. Method B: Mean Centering (Scikit-Learn's internal logic)
X_mean = np.mean(X, axis=0)
y_mean = np.mean(y)

X_centered = X - X_mean
y_centered = y - y_mean

# In the centered space, we solve: theta_coef = (X_c^T X_c + lambda * I)^-1 X_c^T y_c
I_p = np.eye(p)
coef_centering = np.linalg.inv(X_centered.T @ X_centered + lambda_val * I_p) @ X_centered.T @ y_centered
intercept_centering = y_mean - np.dot(X_mean, coef_centering)

# 4. Method C: Scikit-Learn Ridge
# Note: Sklearn does not divide the penalty by N in Ridge (unlike some loss equations),
# so alpha is exactly lambda_val
sklearn_ridge = Ridge(alpha=lambda_val, fit_intercept=True)
sklearn_ridge.fit(X, y)

# 5. Output Verification & Assertions
print("=== Solver Comparison Results ===")
print("Method A (I_0 Projection):")
print(f"  Intercept: {intercept_proj:.8f}")
print(f"  Coefficients: {coef_proj}")
print("Method B (Centering):")
print(f"  Intercept: {intercept_centering:.8f}")
print(f"  Coefficients: {coef_centering}")
print("Method C (Scikit-Learn Ridge):")
print(f"  Intercept: {sklearn_ridge.intercept_:.8f}")
print(f"  Coefficients: {sklearn_ridge.coef_}")

# Assert equality
assert np.allclose(coef_proj, sklearn_ridge.coef_, rtol=1e-10)
assert np.isclose(intercept_proj, sklearn_ridge.intercept_, rtol=1e-10)
assert np.allclose(coef_centering, sklearn_ridge.coef_, rtol=1e-10)
assert np.isclose(intercept_centering, sklearn_ridge.intercept_, rtol=1e-10)

print("\n[SUCCESS] Custom mathematical solvers match Scikit-Learn's Ridge regression output exactly!")
```

---

- **Next Topic**: [065_ridge_regression_part_3.md](file:///Users/prime/Developer/ml/065_ridge_regression_part_3.md) - Ridge Regression: Geometric Interpretation of L2 Regularization.
