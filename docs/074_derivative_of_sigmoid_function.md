# Calculus of the Sigmoid Function: Derivation & Numerical Verification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/074_derivative_of_sigmoid_function.ipynb)

The derivative of the sigmoid function is a central component in updating model parameters during training. Its clean mathematical structure makes it highly efficient to compute. This guide walks through the step-by-step calculus derivation of this derivative and verifies it numerically.

---

## 1. Mathematical Derivation of the Derivative

Let the Sigmoid function be defined as:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$

We wish to calculate the first derivative with respect to $z$:
$$\frac{d}{dz}\sigma(z) = \sigma'(z)$$

### Method 1: Using the Chain Rule

Expressing the sigmoid function as $u(z)^{-1}$ where $u(z) = 1 + e^{-z}$:
$$\frac{d}{dz} u(z)^{-1} = -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z})$$
$$\frac{d}{dz}(1 + e^{-z}) = -e^{-z}$$

Combine the terms:
$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

### Method 2: Using the Quotient Rule

Let $f(z) = 1$ and $g(z) = 1 + e^{-z}$. The quotient rule states:
$$\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$$

Since $f'(z) = 0$ and $g'(z) = -e^{-z}$:
$$\sigma'(z) = \frac{0 \cdot (1 + e^{-z}) - 1 \cdot (-e^{-z})}{(1 + e^{-z})^2} = \frac{e^{-z}}{(1 + e^{-z})^2}$$

### Expressing the Derivative in Terms of $\sigma(z)$

We can rewrite this expression to formulate it in terms of the original function $\sigma(z)$:
$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \left(\frac{1}{1 + e^{-z}}\right) \left(\frac{e^{-z}}{1 + e^{-z}}\right)$$

Notice that:
$$\frac{e^{-z}}{1 + e^{-z}} = \frac{1 + e^{-z} - 1}{1 + e^{-z}} = \frac{1 + e^{-z}}{1 + e^{-z}} - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$$

Therefore:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

```mermaid
flowchart TD
    Sig["Sigmoid: σ("z")"] -->|Compute output| Out["s = σ("z")"]
    Out -->|Derivative Formula| Deriv["σ'(z) = s * (1 - s)"]
```

---

## 2. Python Verification: Analytical vs. Numerical Derivative

To verify the analytical derivative formula, we compare it against a numerical approximation using the **finite difference method** (central difference):
$$\sigma'(z) \approx \frac{\sigma(z + h) - \sigma(z - h)}{2h}$$
where $h$ is a very small step size (e.g., $10^{-6}$).

The following Python script computes both values across a range of $z$ inputs and checks their convergence.

```python
import numpy as np

# 1. Implement Sigmoid and its Derivatives
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative_analytical(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def sigmoid_derivative_numerical(z, h=1e-6):
    return (sigmoid(z + h) - sigmoid(z - h)) / (2.0 * h)

# 2. Evaluate and Compare over a range of z values
z_values = np.linspace(-6.0, 6.0, 13)
analytical_derivs = sigmoid_derivative_analytical(z_values)
numerical_derivs = sigmoid_derivative_numerical(z_values)

print("=== Comparing Analytical and Numerical Derivatives of Sigmoid ===")
print(f"{'Input z':<8} | {'Sigmoid value':<15} | {'Analytical Deriv':<18} | {'Numerical Deriv':<18} | {'Absolute Error':<16}")
print("-" * 85)

for i, z in enumerate(z_values):
    error = np.abs(analytical_derivs[i] - numerical_derivs[i])
    print(f"{z:<8.1f} | {sigmoid(z):<15.6f} | {analytical_derivs[i]:<18.10f} | {numerical_derivs[i]:<18.10f} | {error:<16.2e}")

# Assert that difference is extremely small (well within the discretization error order h^2)
assert np.allclose(analytical_derivs, numerical_derivs, atol=1e-10)
print("\n[SUCCESS] The analytical derivative formula σ'(z) = σ(z)(1 - σ(z)) is verified and matches numerical estimates!")
```

---

- **Next Topic**: [075_logistic_regression_part_5.md](file:///Users/prime/Developer/ml/075_logistic_regression_part_5.md) - Logistic Regression Part 5: Gradient Descent from Scratch.
