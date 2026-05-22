# Softmax Regression (Multinomial Logistic Regression)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/079_softmax_regression.ipynb)

Binary Logistic Regression classifies instances into two categories. When we need to classify instances into three or more classes ($K > 2$), we extend the model to **Softmax Regression** (also known as Multinomial Logistic Regression). Instead of applying the binary sigmoid function, we map scores using the **Softmax function** to generate a probability distribution over all $K$ classes.

---

## 1. Mathematical Formulation

For an input vector $x$ and $K$ target classes, the model maintains a separate weight vector $w_k$ and bias $b_k$ for each class $k \in \{1, \ldots, K\}$.

```mermaid
flowchart TD
    X["Input vector x"] --> L1["z₁ = w₁ᵀx + b₁"]
    X --> L2["z₂ = w₂ᵀx + b₂"]
    X --> LK["z_K = w_Kᵀx + b_K"]
    L1 & L2 & LK --> Softmax{"Softmax Function"}
    Softmax --> P1["P("y=1|x")"]
    Softmax --> P2["P("y=2|x")"]
    Softmax --> PK["P("y=K|x")"]
```

### The Softmax Function

The linear score (logit) for class $k$ is $z_k = w_k^T x + b_k$. The probability that the input $x$ belongs to class $k$ is:
$$\hat{y}_k = P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^K e^{w_j^T x + b_j}}$$

- All predicted probabilities $\hat{y}_k$ are in the range $(0, 1)$.
- The sum of all predicted probabilities is exactly 1: $\sum_{k=1}^K \hat{y}_k = 1$.

### Categorical Cross-Entropy Loss

To train the model, we minimize the **Categorical Cross-Entropy Loss** over $N$ samples:
$$J(W, B) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik})$$

Where:

- $y_{ik}$ is 1 if the true class of sample $i$ is $k$, and 0 otherwise (one-hot encoded targets).
- $\hat{y}_{ik}$ is the model's predicted probability that sample $i$ belongs to class $k$.

### Gradient of Loss with Respect to Weights

Using the chain rule, the gradient of the loss function with respect to the weights $w_k$ of class $k$ simplifies to:
$$\nabla_{w_k} J = \frac{1}{N} \sum_{i=1}^N (\hat{y}_{ik} - y_{ik}) x_i$$

In vectorized form, where $X$ is shape $(N, M)$, $\hat{Y}$ is shape $(N, K)$, and $Y$ is shape $(N, K)$:
$$\nabla_W J = \frac{1}{N} X^T (\hat{Y} - Y)$$
$$\nabla_B J = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

---

## 2. Python Implementation from Scratch

The following runnable Python script implements a Softmax Regression classifier from scratch using NumPy. It includes one-hot encoding, stable Softmax computation (preventing numerical overflow), and vectorized Gradient Descent, comparing final weights and predictions against Scikit-Learn.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Implement Helper Functions
def to_one_hot(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot

# 2. Implement Softmax Regression Class
class SoftmaxRegressionScratch:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _softmax(self, Z):
        # Subtract max for numerical stability (prevents overflow of e^Z)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        # Initialize weights and biases
        # weights shape: (n_features, num_classes), bias shape: (num_classes,)
        self.weights = np.zeros((n_features, num_classes))
        self.bias = np.zeros(num_classes)

        Y_one_hot = to_one_hot(y, num_classes)

        for _ in range(self.n_iters):
            # Compute scores: Z = X * W + B
            Z = np.dot(X, self.weights) + self.bias
            # Compute probabilities
            probs = self._softmax(Z)

            # Compute gradients
            dw = (1.0 / n_samples) * np.dot(X.T, (probs - Y_one_hot))
            db = (1.0 / n_samples) * np.sum(probs - Y_one_hot, axis=0)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        Z = np.dot(X, self.weights) + self.bias
        return self._softmax(Z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# 3. Generate 3-class synthetic classification dataset with noise
np.random.seed(42)
n_samples = 150
X = np.random.randn(n_samples, 2)
# Determine classes based on spatial division with noise to ensure finite weights convergence
y = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    val = X[i, 0] + 0.5 * X[i, 1] + np.random.randn() * 0.3
    if val < -0.5:
        y[i] = 0
    elif val < 0.5:
        y[i] = 1
    else:
        y[i] = 2

# 4. Train Scratch and Scikit-Learn Models
scratch_model = SoftmaxRegressionScratch(lr=1.0, n_iters=15000)
scratch_model.fit(X, y)

# Scikit-Learn Logistic Regression with lbfgs solver and no regularization
sklearn_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000, tol=1e-7)
sklearn_model.fit(X, y)

# 5. Evaluate and Verify
probs_scratch = scratch_model.predict_proba(X)
probs_sklearn = sklearn_model.predict_proba(X)

# Compare first 5 probability vectors
print("=== Softmax Regression Predictions (First 5 samples) ===")
for i in range(5):
    print(f"Scratch: {probs_scratch[i].round(4)} | Sklearn: {probs_sklearn[i].round(4)}")

# Assert that predictions are aligned
assert np.allclose(probs_scratch, probs_sklearn, atol=1e-2)
print("\n[SUCCESS] Custom Multiclass Softmax Regression predicts identical class probabilities to Scikit-Learn!")
```

---

- **Next Topic**: [080_polynomial_features_in_logistic_regression.md](file:///Users/prime/Developer/ml/080_polynomial_features_in_logistic_regression.md) - Non-linear classification with Polynomial Logistic Regression.
