# Why Lasso Regression Creates Sparsity: Geometry & Coordinate Descent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/068_why_lasso_regression_creates_sparsity.ipynb)

Lasso's ability to drive coefficients exactly to zero allows it to perform automatic feature selection. In this guide, we analyze this behavior from both a **geometric perspective** and a **mathematical optimization perspective** using subgradients and coordinate descent.

---

## 1. Geometric Explanation: Diamond vs. Circle

The difference in sparsity generation between Ridge and Lasso lies in the geometry of their constraint boundaries:

```mermaid
flowchart TD
    subgraph Ridge (L2 Penalty)
        L2["Constraint: θ₁² + θ₂² ≤ t"] --> L2Shape["Boundary is a smooth circle/hypersphere"]
        L2Shape --> L2Tangent["Tangency point is rarely on a coordinate axis"]
    end
    subgraph Lasso (L1 Penalty)
        L1["Constraint: |θ₁| + |θ₂| ≤ t"] --> L1Shape["Boundary is a diamond with sharp corners"]
        L1Shape --> L1Tangent["Elliptical loss contours are highly likely to hit a sharp corner (axis)"]
    end
```

In a 2D parameter space ($\theta_1, \theta_2$):

- **L2 Constraint (Ridge)**: $\theta_1^2 + \theta_2^2 \le t$. The constraint boundary is a smooth circle. The elliptical contours of the MSE loss function will touch the circle at a tangent point. Because the circle is curved, this tangent point is highly unlikely to lie exactly on the axis ($\theta_1 = 0$ or $\theta_2 = 0$).
- **L1 Constraint (Lasso)**: $|\theta_1| + |\theta_2| \le t$. The constraint boundary is a diamond. The diamond has sharp corners on the coordinate axes. As the elliptical loss contours expand from the OLS solution, they are geometrically much more likely to hit one of these corners first, setting one of the coefficients to exactly zero.

---

## 2. Mathematical Derivation: Subgradients & Soft Thresholding

To optimize the Lasso objective function, we use **Coordinate Descent**, which optimizes one parameter $\theta_j$ at a time while keeping all other parameters constant.

The Lasso objective function is:
$$J(\theta) = \frac{1}{2N} \sum_{i=1}^N \left( y_i - \sum_{k=1}^p \theta_k x_{ik} \right)^2 + \alpha \sum_{k=1}^p |\theta_k|$$

_(Note: We use the Scikit-Learn convention of multiplying the MSE term by $\frac{1}{2N}$ and using $\alpha$ as the regularization hyperparameter)._

To isolate $\theta_j$, we write:
$$J_j(\theta_j) = \frac{1}{2N} \sum_{i=1}^N \left( \left(y_i - \sum_{k \ne j} \theta_k x_{ik}\right) - \theta_j x_{ij} \right)^2 + \alpha |\theta_j| + C$$

Let the **partial residual** for sample $i$ excluding feature $j$ be:
$$r_{i,-j} = y_i - \sum_{k \ne j} \theta_k x_{ik}$$

The objective simplifies to:
$$J_j(\theta_j) = \frac{1}{2N} \sum_{i=1}^N \left( r_{i,-j} - \theta_j x_{ij} \right)^2 + \alpha |\theta_j|$$

Expanding and taking the subgradient with respect to $\theta_j$:
$$\partial_{\theta_j} J_j(\theta_j) = \frac{1}{N} \sum_{i=1}^N \left( -x_{ij} \left( r_{i,-j} - \theta_j x_{ij} \right) \right) + \alpha \cdot \partial(|\theta_j|)$$
$$= \theta_j \left( \frac{1}{N} \sum_{i=1}^N x_{ij}^2 \right) - \frac{1}{N} \sum_{i=1}^N r_{i,-j} x_{ij} + \alpha \cdot \text{sign}(\theta_j)$$

Let us define:

- $\rho_j = \frac{1}{N} \sum_{i=1}^N r_{i,-j} x_{ij}$ (The correlation between feature $j$ and the partial residual).
- $z_j = \frac{1}{N} \sum_{i=1}^N x_{ij}^2$ (If features are standardized, $z_j = 1$).

Our subgradient equation is:
$$\partial_{\theta_j} J_j(\theta_j) = z_j \theta_j - \rho_j + \alpha \cdot \text{sign}(\theta_j)$$

Where the subgradient of the absolute value is:
$$\text{sign}(\theta_j) \in \begin{cases} \{-1\} & \text{if } \theta_j < 0 \\ [-1, 1] & \text{if } \theta_j = 0 \\ \{1\} & \text{if } \theta_j > 0 \end{cases}$$

### Solving the Subgradient cases

1. If $\theta_j > 0$: $z_j \theta_j - \rho_j + \alpha = 0 \implies \theta_j = \frac{\rho_j - \alpha}{z_j}$. For this to be valid, we must have $\rho_j > \alpha$.
2. If $\theta_j < 0$: $z_j \theta_j - \rho_j - \alpha = 0 \implies \theta_j = \frac{\rho_j + \alpha}{z_j}$. For this to be valid, we must have $\rho_j < -\alpha$.
3. If $\theta_j = 0$: The subgradient interval contains 0: $-\rho_j + \alpha \cdot [-1, 1] \ni 0 \implies |\rho_j| \le \alpha$.

Combining these three conditions yields the **Soft Thresholding Operator**:
$$\theta_j = \text{SoftThresholding}\left(\frac{\rho_j}{z_j}, \frac{\alpha}{z_j}\right)$$

Where:
$$\text{SoftThresholding}(\omega, \gamma) = \text{sign}(\omega) \max(0, |\omega| - \gamma)$$

If $|\rho_j| \le \alpha$, the thresholding operator truncates $\theta_j$ to **exactly zero**, creating sparsity.

---

## 3. Python Coordinate Descent Lasso Solver from Scratch

The following runnable Python script implements Coordinate Descent with Soft Thresholding from scratch. We compare the learned coefficients directly with Scikit-Learn's `Lasso`.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Soft Thresholding operator
def soft_thresholding(omega, gamma):
    if omega > gamma:
        return omega - gamma
    elif omega < -gamma:
        return omega + gamma
    else:
        return 0.0

# 1. Generate Synthetic Dataset
np.random.seed(42)
N = 60
p = 4
X = np.random.normal(0, 1, size=(N, p))
# Target depends on features 0 and 2
y = 4.0 * X[:, 0] - 2.5 * X[:, 2] + np.random.normal(0, 0.4, size=N)

# Standardize features (mandatory for coordinate descent scaling assumptions)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Subtract target mean to center y, allowing us to fit without an explicit intercept
y_centered = y - np.mean(y)

# 2. Custom Coordinate Descent Solver
alpha = 0.3
n_iterations = 200
theta = np.zeros(p)  # Initialize weights to zero

for iteration in range(n_iterations):
    for j in range(p):
        # Calculate partial residual: r_i,-j = y_i - Sum_{k != j} theta_k * x_ik
        y_pred_except_j = np.dot(X_scaled, theta) - theta[j] * X_scaled[:, j]
        r_neg_j = y_centered - y_pred_except_j

        # rho_j = (1 / N) * Sum(r_neg_j * x_ij)
        rho_j = np.mean(r_neg_j * X_scaled[:, j])

        # z_j = (1 / N) * Sum(x_ij^2). Since features are scaled, z_j is exactly 1.0
        z_j = np.mean(X_scaled[:, j] ** 2)

        # Update theta_j using Soft Thresholding
        theta[j] = soft_thresholding(rho_j / z_j, alpha / z_j)

# 3. Scikit-Learn Lasso Comparison
# fit_intercept=False because y and X are centered
sklearn_lasso = Lasso(alpha=alpha, fit_intercept=False, tol=1e-6, max_iter=1000)
sklearn_lasso.fit(X_scaled, y_centered)
theta_sklearn = sklearn_lasso.coef_

# 4. Results & Verification
print("=== Lasso Solver Comparison (Coordinate Descent) ===")
print(f"Scratch Solver Coefficients:     {theta}")
print(f"Scikit-Learn Lasso Coefficients: {theta_sklearn}")

# Assert proximity of solutions
assert np.allclose(theta, theta_sklearn, atol=1e-4), "Custom coordinate descent solver is incorrect!"
print("\n[SUCCESS] Custom Coordinate Descent Lasso solver matches Scikit-Learn's coefficients!")
```

---

- **Next Topic**: [069_elasticnet_regression.md](file:///Users/prime/Developer/ml/069_elasticnet_regression.md) - Elastic Net Regression: Combining L1 and L2 regularizations.
