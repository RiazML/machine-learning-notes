# Logistic Regression Part 5: Gradient Derivation & Gradient Descent from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/075_logistic_regression_part_5.ipynb)

Training a Logistic Regression model means minimizing the Binary Cross-Entropy (BCE) Loss function. Since there is no closed-form analytical solution (like the normal equation in linear regression), we use **Gradient Descent** to find the optimal parameters. This guide derives the gradient of the BCE loss step-by-step and implements Logistic Regression from scratch.

---

## 1. Step-by-Step Gradient Derivation

Let our cost function for $N$ samples be:
$$J(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

Where the prediction is:
$$p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$
And the linear combination is:
$$z_i = w^T x_i + b = \sum_{j=1}^M w_j x_{ij} + b$$

We want to find the partial derivatives of $J$ with respect to weight $w_j$ using the **Chain Rule**:
$$\frac{\partial J}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N \frac{\partial J_i}{\partial w_j}$$
Where $J_i$ is the loss of a single sample $i$. Applying the chain rule:
$$\frac{\partial J_i}{\partial w_j} = \frac{\partial J_i}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_j}$$

Let's compute each component individually:

1. **Component 1: $\frac{\partial J_i}{\partial p_i}$** (derivative of loss with respect to probability)
    $$\frac{\partial J_i}{\partial p_i} = \frac{\partial}{\partial p_i} \left( - \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] \right) = - \frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i} = \frac{-y_i(1 - p_i) + p_i(1 - y_i)}{p_i(1 - p_i)} = \frac{p_i - y_i}{p_i(1 - p_i)}$$

2. **Component 2: $\frac{\partial p_i}{\partial z_i}$** (derivative of Sigmoid)
    As derived in [Day 74](file:///Users/prime/Developer/ml/074_derivative_of_sigmoid_function.md):
    $$\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$$

3. **Component 3: $\frac{\partial z_i}{\partial w_j}$** (derivative of linear equation)
    $$\frac{\partial z_i}{\partial w_j} = \frac{\partial}{\partial w_j} \left( \sum_{k=1}^M w_k x_{ik} + b \right) = x_{ij}$$

### Combining the Components

Substitute the three components back into the chain rule product:
$$\frac{\partial J_i}{\partial w_j} = \left[ \frac{p_i - y_i}{p_i(1 - p_i)} \right] \cdot \left[ p_i(1 - p_i) \right] \cdot x_{ij}$$

Notice that the term $p_i(1 - p_i)$ in the numerator and denominator cancels out:
$$\frac{\partial J_i}{\partial w_j} = (p_i - y_i)x_{ij}$$

Taking the average over all samples $N$:
$$\frac{\partial J}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i) x_{ij}$$

Similarly, for the bias $b$ (since $\frac{\partial z_i}{\partial b} = 1$):
$$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)$$

### Vectorized Formulation

Let $X$ be the feature matrix of shape $(N, M)$, $y$ be the label vector of shape $(N, 1)$, and $p$ be the prediction vector of shape $(N, 1)$:
$$\nabla_w J = \frac{1}{N} X^T (p - y)$$
$$\frac{\partial J}{\partial b} = \frac{1}{N} \sum (p_i - y_i)$$

```mermaid
flowchart TD
    Init["Initialize Weights w and Bias b"] --> Predictions["Compute p = σ("Xw + b")"]
    Predictions --> Gradients["Compute Gradients: dw = (1/N) Xᵀ("p - y"), db = (1/N) ∑(p - y)"]
    Gradients --> Update["Update parameters: w = w - η * dw, b = b - η * db"]
    Update --> Check{"Epoch limit reached?"}
    Check -->|No| Predictions
    Check -->|Yes| End["Final Weights and Bias"]
```

---

## 2. Python Implementation from Scratch

The following runnable Python script implements a `CustomLogisticRegression` classifier utilizing vectorized Gradient Descent. It trains on synthetic 2D data and validates its parameters and prediction accuracy against Scikit-Learn's `LogisticRegression` (with regularization turned off).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Define custom logistic regression model
class CustomLogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient Descent loop
        for _ in range(self.n_iters):
            # Compute linear combination
            z = np.dot(X, self.weights) + self.bias
            # Predict probabilities
            p = self._sigmoid(z)

            # Compute gradients
            dw = (1.0 / n_samples) * np.dot(X.T, (p - y))
            db = (1.0 / n_samples) * np.sum(p - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        p = self._sigmoid(z)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.where(probs[:, 1] >= 0.5, 1, 0)

# 2. Generate Synthetic 2D Classification Dataset with Noise
np.random.seed(42)
n_samples = 150
X = np.random.randn(n_samples, 2)
# True separation line with noise
y = (X[:, 0] + 2.0 * X[:, 1] - 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

# 3. Train both classifiers
custom_model = CustomLogisticRegression(lr=2.0, n_iters=15000)
custom_model.fit(X, y)

sklearn_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000, tol=1e-7)
sklearn_model.fit(X, y)

# 4. Compare and Assert Correctness
custom_w = custom_model.weights
custom_b = custom_model.bias
sklearn_w = sklearn_model.coef_[0]
sklearn_b = sklearn_model.intercept_[0]

print("=== Custom vs Scikit-Learn Parameter Match ===")
print(f"Custom weights:  {custom_w}")
print(f"Sklearn weights: {sklearn_w}")
print(f"Custom bias:     {custom_b:.6f}")
print(f"Sklearn bias:    {sklearn_b:.6f}")

# Assert parameters are very close
assert np.allclose(custom_w, sklearn_w, atol=1e-2), "Weights do not match"
assert np.isclose(custom_b, sklearn_b, atol=1e-2), "Bias does not match"

# Assert prediction probabilities match
custom_probs = custom_model.predict_proba(X)
sklearn_probs = sklearn_model.predict_proba(X)
assert np.allclose(custom_probs, sklearn_probs, atol=1e-2)
print("\n[SUCCESS] Custom Gradient Descent implementation matches Scikit-Learn predictions and weights perfectly!")
```

---

- **Next Topic**: [076_accuracy_and_confusion_matrix.md](file:///Users/prime/Developer/ml/076_accuracy_and_confusion_matrix.md) - Model Evaluation: Confusion Matrix and Accuracy.
