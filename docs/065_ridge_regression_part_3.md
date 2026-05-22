# Ridge Regression: Geometric Interpretation & Lagrange Multipliers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/065_ridge_regression_part_3.ipynb)

Understanding Ridge Regression geometrically provides the foundation for why L2 regularization behaves the way it does. We will analyze the optimization problem from both the unconstrained penalization perspective and the constrained optimization perspective using Lagrange multipliers.

---

## 1. Constrained Optimization & Lagrange Formulation

The standard Ridge formulation minimizes:
$$\min_{\theta} J(\theta) = \frac{1}{N} \|Y - X\theta\|_2^2 + \lambda \|\theta\|_2^2$$

This formulation is mathematically equivalent to the constrained optimization problem:
$$\min_{\theta} L(\theta) = \frac{1}{N} \|Y - X\theta\|_2^2 \quad \text{subject to} \quad g(\theta) = \|\theta\|_2^2 \le t$$

Where $t > 0$ represents the "budget" or radius of the parameter constraint. The relationship between $\lambda$ and $t$ is monotonic and inverse: as $\lambda \to 0$, $t \to \|\theta_{\text{OLS}}\|_2^2$, and as $\lambda \to \infty$, $t \to 0$.

### The Lagrangian Function

To solve this constrained problem, we construct the Lagrangian:
$$\mathcal{L}(\theta, \tilde{\lambda}) = \frac{1}{N} \|Y - X\theta\|_2^2 + \tilde{\lambda} \left( \|\theta\|_2^2 - t \right)$$

Where $\tilde{\lambda} \ge 0$ is the Lagrange multiplier. According to the Karush-Kuhn-Tucker (KKT) conditions:

1. **Stationarity**: $\nabla_\theta \mathcal{L}(\theta, \tilde{\lambda}) = 0 \implies \nabla_\theta L(\theta) + 2\tilde{\lambda}\theta = 0$
2. **Primal Feasibility**: $\|\theta\|_2^2 - t \le 0$
3. **Dual Feasibility**: $\tilde{\lambda} \ge 0$
4. **Complementary Slackness**: $\tilde{\lambda} (\|\theta\|_2^2 - t) = 0$

If the OLS solution lies outside the budget sphere (i.e. $\|\theta_{\text{OLS}}\|_2^2 > t$), the constraint is active, meaning the optimal Ridge solution lies **exactly on the boundary** of the sphere: $\|\theta\|_2^2 = t$. In this case, $\tilde{\lambda} > 0$.

---

## 2. Geometric Interpretation

In a 2D parameter space ($\theta_1, \theta_2$), we can visualize the optimization landscape:

```mermaid
flowchart TD
    OLS["OLS Solution θ_OLS (Center of Loss Ellipses)"] -.->|Increasing Loss| Contours["Elliptical Loss Contours: MSE("θ") = c"]
    Origin["Origin (0,0) (Center of L2 Constraint)"] --->|Radius √t| Sphere["L2 Constraint Circle: θ₁² + θ₂² = t"]
    Contours ---|Tangency Point| Sphere
    Sphere ---|Ridge Solution θ_Ridge| Contours
```

- **OLS Loss Contours**: The MSE loss is a quadratic function, drawing concentric ellipses centered at the unconstrained OLS solution $\theta_{\text{OLS}}$.
- **L2 Constraint Circle**: The constraint $\|\theta\|_2^2 \le t$ represents a solid circle (in 2D) centered at the origin.
- **The Intersection**: The Ridge regression solution is the point where the smallest possible loss ellipse just touches (is tangent to) the boundary of the constraint circle.
- **No Sparsity**: Because the L2 circle is smooth and has no corners, the tangent point of contact rarely occurs exactly on a coordinate axis (where a coefficient would be zero), which is why Ridge shrinks coefficients but does not perform variable selection.

---

## 3. Mathematical Proof of Tangency (Collinearity of Gradients)

At the tangency point $\theta_{\text{Ridge}}$, the boundary of the constraint and the contour of the loss function share the same tangent line. Therefore, their gradient vectors must be **collinear** (pointing in opposite directions):

$$\nabla_\theta L(\theta_{\text{Ridge}}) \propto -\theta_{\text{Ridge}}$$

Let's verify this analytically. The gradient of the loss is:
$$\nabla_\theta L(\theta) = \frac{2}{N} \left( X^TX\theta - X^TY \right)$$

Substituting the closed-form Ridge solution $(X^TX + N\tilde{\lambda} I)\theta_{\text{Ridge}} = X^TY$, we get:
$$\nabla_\theta L(\theta_{\text{Ridge}}) = \frac{2}{N} \left( X^TX\theta_{\text{Ridge}} - (X^TX + N\tilde{\lambda} I)\theta_{\text{Ridge}} \right)$$
$$\nabla_\theta L(\theta_{\text{Ridge}}) = -2\tilde{\lambda} \theta_{\text{Ridge}}$$

This proves that the loss gradient is pointing directly in the opposite direction of the coefficient vector $\theta_{\text{Ridge}}$, confirming the geometric tangency condition.

---

## 4. Python Demonstration: Gradient Collinearity Verification

The following runnable Python script fits a Ridge model, computes the gradient of the MSE loss at the Ridge coefficients from scratch, and verifies that the cosine similarity between the loss gradient and the coefficient vector is exactly $-1.0$.

```python
import numpy as np
from sklearn.linear_model import Ridge

# 1. Generate Synthetic Dataset
np.random.seed(42)
N = 50
X = np.random.normal(loc=0.0, scale=1.0, size=(N, 2))
# Ground truth: y = 5*x1 - 3*x2 + noise
y = 5.0 * X[:, 0] - 3.0 * X[:, 1] + np.random.normal(0, 0.5, size=N)

# Fit Ridge Regression (fit_intercept=False to simplify geometry to the origin)
alpha = 25.0
ridge = Ridge(alpha=alpha, fit_intercept=False)
ridge.fit(X, y)
theta_ridge = ridge.coef_

# 2. Compute the Gradient of the MSE Loss at the Ridge Solution
# MSE Loss: L(theta) = (1/N) * ||y - X*theta||₂²
# Gradient: grad_L = -(2/N) * X.T * (y - X*theta)
y_pred = X @ theta_ridge
residuals = y - y_pred
grad_loss = -(2.0 / N) * (X.T @ residuals)

# 3. Compute Cosine Similarity between the loss gradient and theta_ridge
# Cosine similarity = (A . B) / (||A|| * ||B||)
dot_product = np.dot(grad_loss, theta_ridge)
norm_grad = np.linalg.norm(grad_loss)
norm_theta = np.linalg.norm(theta_ridge)
cosine_sim = dot_product / (norm_grad * norm_theta)

print("=== Geometric Tangency Verification ===")
print(f"Ridge Coefficients θ_ridge:        {theta_ridge}")
print(f"MSE Loss Gradient at θ_ridge:      {grad_loss}")
print(f"Cosine Similarity (Grad vs θ):     {cosine_sim:.8f}")

# 4. Verify Lagrange Multiplier Relation
# grad_loss + 2 * (alpha / N) * theta_ridge should be close to zero vector
multiplier = alpha / N
reconstruction = grad_loss + 2 * multiplier * theta_ridge
print(f"Lagrange Optimality Sum:           {reconstruction}")

# Assertions
assert np.isclose(cosine_sim, -1.0, rtol=1e-10), "Gradients are not pointing in opposite directions!"
assert np.allclose(reconstruction, np.zeros(2), atol=1e-10), "Lagrange optimality condition failed!"

print("\n[SUCCESS] Geometric tangency verified: Gradient of loss is collinear and opposite to the coefficient vector!")
```

---

- **Next Topic**: [066_5_key_points.md](file:///Users/prime/Developer/ml/066_5_key_points.md) - 5 Key Practical Points on Regularization and Scale Sensitivity.
