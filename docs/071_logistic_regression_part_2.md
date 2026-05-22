# The Perceptron Trick: Binary Classification from a Geometric Perspective

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/071_logistic_regression_part_2.ipynb)

The **Perceptron** is one of the earliest and simplest forms of an artificial neural network, introduced by Frank Rosenblatt in 1958. It serves as a foundational step toward understanding modern classification models like Logistic Regression and Support Vector Machines. Instead of estimating probabilities, the Perceptron finds a linear decision boundary directly using a feedback mechanism known as the **Perceptron Trick**.

---

## 1. Geometric Intuition of the Decision Boundary

In a 2D feature space with inputs $x = [x_1, x_2]^T$, a linear decision boundary is a straight line that splits the space into two halves. This boundary is defined by the equation:
$$w_1 x_1 + w_2 x_2 + b = 0$$

Where:

- $w_1, w_2$ are the **weights** determining the orientation/slope of the boundary line.
- $b$ is the **bias** determining the offset (intercept) of the line from the origin.
- Vector notation: $w^T x + b = 0$, where $w = [w_1, w_2]^T$ and $x = [x_1, x_2]^T$.

The vector $w$ is normal (perpendicular) to the decision boundary line and points toward the positive half-space (where $w^T x + b > 0$).

```mermaid
flowchart TD
    subgraph Perceptron Model Architecture
        X1["Input x₁"] -->|w₁| Sum["Sum: z = w₁x₁ + w₂x₂ + b"]
        X2["Input x₂"] -->|w₂| Sum
        Bias["Bias b"] -->|1| Sum
        Sum --> Act{"Step Activation f("z")"}
        Act -->|z ≥ 0| Output1["Class 1 (ŷ = 1)"]
        Act -->|z < 0| Output0["Class 0 (ŷ = 0)"]
    end
```

---

## 2. Mathematical Formulation & Update Rules

The Perceptron uses a step function to convert the linear combination $z = w^T x + b$ into a binary prediction $\hat{y} \in \{0, 1\}$:
$$\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$

### The Perceptron Trick (Feedback Loop)

When the Perceptron misclassifies a training instance, it adjusts its weights and bias to nudge the boundary line toward the correct classification.

Let $(x_i, y_i)$ be a training sample where $y_i \in \{0, 1\}$.

1. **If the sample is positive ($y_i = 1$) but predicted negative ($\hat{y}_i = 0$):**
    We need $w^T x_i + b$ to increase. We update the weights and bias by adding a fraction of the input vector:
    $$w_j \leftarrow w_j + \eta \cdot x_{ij}$$
    $$b \leftarrow b + \eta$$
    Where $\eta$ is the learning rate ($0 < \eta \le 1$).

2. **If the sample is negative ($y_i = 0$) but predicted positive ($\hat{y}_i = 1$):**
    We need $w^T x_i + b$ to decrease. We update the weights and bias by subtracting a fraction of the input vector:
    $$w_j \leftarrow w_j - \eta \cdot x_{ij}$$
    $$b \leftarrow b - \eta$$

### Generalized Perceptron Update Equation

We can unify these two rules into a single equation using the prediction error $(y_i - \hat{y}_i) \in \{-1, 0, 1\}$:
$$w_j \leftarrow w_j + \eta \cdot (y_i - \hat{y}_i) \cdot x_{ij}$$
$$b \leftarrow b + \eta \cdot (y_i - \hat{y}_i)$$

- If predictions are correct, $y_i - \hat{y}_i = 0$, and no updates occur.
- If a point is misclassified, the line shifts slightly. If the dataset is linearly separable, the Perceptron is guaranteed to converge to a separating line in a finite number of steps (Perceptron Convergence Theorem).

---

## 3. Python Implementation from Scratch

The following self-contained Python script implements the Perceptron Trick from scratch on a synthetic, linearly separable 2D dataset. It shows how the weights converge and confirms that the final accuracy reaches 100%.

```python
import numpy as np

# 1. Generate linearly separable 2D synthetic data
np.random.seed(42)
n_samples = 100
# Generate coordinates clustered in two regions
class_0 = np.random.multivariate_normal([1.0, 1.0], [[0.15, 0], [0, 0.15]], n_samples // 2)
class_1 = np.random.multivariate_normal([3.0, 3.0], [[0.15, 0], [0, 0.15]], n_samples // 2)

X = np.vstack((class_0, class_1))
y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2))).astype(int)

# Shuffle dataset
indices = np.arange(n_samples)
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# 2. Implement Perceptron from scratch using the Perceptron Trick
class PerceptronFromScratch:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias to zeros (or small random numbers)
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            errors_in_epoch = 0
            for idx, x_i in enumerate(X):
                # Calculate linear output z = w^T x + b
                z = np.dot(x_i, self.weights) + self.bias
                # Apply step function prediction
                y_pred = 1 if z >= 0 else 0

                # Compute error
                error = y[idx] - y_pred
                if error != 0:
                    # Update weights and bias: w = w + lr * error * x
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    errors_in_epoch += 1

            # Early stopping if fully converged (no errors)
            if errors_in_epoch == 0:
                print(f"Converged early at epoch {epoch + 1}!")
                break

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.where(z >= 0, 1, 0)

# 3. Train the model
model = PerceptronFromScratch(learning_rate=0.05, epochs=100)
model.fit(X, y)

# 4. Evaluate the model
predictions = model.predict(X)
accuracy = np.mean(predictions == y) * 100.0

print("\n=== Perceptron Trick Training Summary ===")
print(f"Learned Weights (w): {model.weights}")
print(f"Learned Bias (b):    {model.bias:.4f}")
print(f"Final Accuracy:      {accuracy:.2f}%")

# Assert that convergence was successful and reached 100% accuracy
assert accuracy == 100.0, "Model did not separate linearly separable classes"
print("\n[SUCCESS] The Perceptron classifier successfully converged to a 100% separating boundary!")
```

---

- **Next Topic**: [072_logistic_regression_part_3.md](file:///Users/prime/Developer/ml/072_logistic_regression_part_3.md) - Logistic Regression Part 3: Sigmoid, Odds, and Log-Odds.
