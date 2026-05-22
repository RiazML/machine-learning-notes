# Gradient Descent Optimization from Scratch

While Ordinary Least Squares (OLS) closed-form solutions work well for small datasets, they become computationally prohibitive when the number of features $p$ is extremely large (due to the $O(p^3)$ matrix inversion step). **Gradient Descent** is an iterative first-order optimization algorithm that scales much better to massive datasets and forms the backbone of modern machine learning and deep learning training.

---

## 1. Why Gradient Descent?

```mermaid
graph TD
    A["Optimization Strategy"] --> B["Closed-Form OLS"]
    A --> C["Gradient Descent"]
    B --> B1["Complexity: O("p^3") matrix inversion"]
    B --> B2["Infeasible for very large feature counts"]
    C --> C1["Complexity: O("K * N * p") iterative updates"]
    C --> C2["Highly scalable and memory efficient"]
```

---

## 2. Mathematical Derivation of Gradients for Simple Linear Regression

For a simple linear regression model $y_i = m x_i + c + \epsilon_i$, we define the Mean Squared Error (MSE) cost function as:
$$J(m, c) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_{i=1}^N (y_i - (m x_i + c))^2$$

To minimize $J(m, c)$, we calculate the partial derivatives with respect to the parameters $m$ and $c$ using the chain rule:

### Derivative with respect to slope $m$

$$\frac{\partial J}{\partial m} = \frac{\partial}{\partial m} \left[ \frac{1}{N} \sum_{i=1}^N (y_i - (m x_i + c))^2 \right]$$
$$\frac{\partial J}{\partial m} = \frac{1}{N} \sum_{i=1}^N 2 (y_i - (m x_i + c)) \cdot \frac{\partial (y_i - m x_i - c)}{\partial m}$$
$$\frac{\partial J}{\partial m} = -\frac{2}{N} \sum_{i=1}^N x_i (y_i - (m x_i + c))$$

### Derivative with respect to intercept $c$

$$\frac{\partial J}{\partial c} = \frac{\partial}{\partial c} \left[ \frac{1}{N} \sum_{i=1}^N (y_i - (m x_i + c))^2 \right]$$
$$\frac{\partial J}{\partial c} = \frac{1}{N} \sum_{i=1}^N 2 (y_i - (m x_i + c)) \cdot \frac{\partial (y_i - m x_i - c)}{\partial c}$$
$$\frac{\partial J}{\partial c} = -\frac{2}{N} \sum_{i=1}^N (y_i - (m x_i + c))$$

---

## 3. Parameter Update Equation

At each iteration (epoch), parameters are updated in the direction of steepest descent (opposite to the gradient vector) scaled by the **learning rate** $\alpha$:
$$m \leftarrow m - \alpha \frac{\partial J}{\partial m}$$
$$c \leftarrow c - \alpha \frac{\partial J}{\partial c}$$

- If the learning rate $\alpha$ is **too small**, convergence is slow.
- If the learning rate $\alpha$ is **too large**, the algorithm can overshoot the minimum and diverge.

```mermaid
graph TD
    Start["Initialize slope m & intercept c randomly"] --> ComputePred["Compute predictions: y_pred = m*x + c"]
    ComputePred --> ComputeCost["Compute Cost J("m,c") = MSE"]
    ComputeCost --> CheckConv{"Cost delta < tolerance or max epochs?"}
    CheckConv -- Yes --> End["Model converged. Return parameters"]
    CheckConv -- No --> CompGrad["Compute partial derivatives: dJ/dm & dJ/dc"]
    CompGrad --> UpdateParam["Update: m = m - alpha*dJ/dm, c = c - alpha*dJ/dc"]
    UpdateParam --> ComputePred
```

---

## 4. Custom Python Implementation with Training Loop

Below is a complete, self-contained Python script implementing Simple Linear Regression via Gradient Descent, tracking cost history, and demonstrating convergence compared to Scikit-Learn.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

class SimpleRegressorGD:
    """
    Simple Linear Regression model optimized via Gradient Descent from scratch.
    """
    def __init__(self, learning_rate=0.01, epochs=1000, tolerance=1e-7):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.m = 0.0 # Initial slope
        self.c = 0.0 # Initial intercept
        self.cost_history_ = []

    def fit(self, X, y):
        """
        Fit the model using Batch Gradient Descent for Simple Linear Regression.
        """
        X_arr = np.asarray(X, dtype=np.float64).flatten()
        y_arr = np.asarray(y, dtype=np.float64).flatten()
        N = len(X_arr)

        for epoch in range(self.epochs):
            # 1. Compute predictions
            y_pred = (self.m * X_arr) + self.c

            # 2. Compute cost (MSE)
            cost = np.mean((y_arr - y_pred) ** 2)
            self.cost_history_.append(cost)

            # 3. Compute gradients (partial derivatives)
            dm = (-2.0 / N) * np.sum(X_arr * (y_arr - y_pred))
            dc = (-2.0 / N) * np.sum(y_arr - y_pred)

            # 4. Check for convergence based on parameter update step size
            prev_m, prev_c = self.m, self.c

            # 5. Apply gradient updates
            self.m -= self.learning_rate * dm
            self.c -= self.learning_rate * dc

            # Break if parameter updates are below tolerance
            if max(abs(self.m - prev_m), abs(self.c - prev_c)) < self.tolerance:
                print(f"Converged early at epoch {epoch}!")
                break

        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64).flatten()
        return (self.m * X_arr) + self.c

# 1. Generate Synthetic Data
# Ground truth: y = 4.2 * x + 5.0
np.random.seed(42)
n_samples = 150
X_raw = np.random.uniform(0.0, 10.0, size=n_samples)
y_raw = 4.2 * X_raw + 5.0 + np.random.normal(loc=0.0, scale=1.5, size=n_samples)

# Normalize data to prevent gradient explosion/divergence
# (Standard feature scaling is highly recommended for gradient descent)
X_mean, X_std = np.mean(X_raw), np.std(X_raw)
X_scaled = (X_raw - X_mean) / X_std

# 2. Fit Custom Model
gd_model = SimpleRegressorGD(learning_rate=0.05, epochs=3000)
gd_model.fit(X_scaled, y_raw)

# 3. Fit Scikit-Learn Model
sklearn_model = LinearRegression()
sklearn_model.fit(X_scaled.reshape(-1, 1), y_raw)

# 4. Verify Results
print("=== Parameter Tuning Verification ===")
print(f"Custom GD Model Slope (m):       {gd_model.m:.6f}")
print(f"Sklearn Model Slope (coef_):     {sklearn_model.coef_[0]:.6f}")
print(f"Custom GD Model Intercept (c):   {gd_model.c:.6f}")
print(f"Sklearn Model Intercept:         {sklearn_model.intercept_:.6f}")
print(f"Initial Cost:                    {gd_model.cost_history_[0]:.6f}")
print(f"Final Cost:                      {gd_model.cost_history_[-1]:.6f}")

# Assert proximity to Sklearn optimal parameters
assert np.isclose(gd_model.m, sklearn_model.coef_[0], rtol=1e-4)
assert np.isclose(gd_model.c, sklearn_model.intercept_, rtol=1e-4)

print("\n[SUCCESS] Custom gradient descent solver matched standard Scikit-Learn coefficients closely!")
```

---

- **Next Topic**: [058_batch_gradient_descent_with_code_demo.md](file:///Users/prime/Developer/ml/058_batch_gradient_descent_with_code_demo.md) - Vectorized Batch Gradient Descent for Multiple Features.
