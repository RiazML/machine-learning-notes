# Introduction to Logistic Regression: Odds, Log-Odds, & The Sigmoid Function

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/070_logistic_regression_part_1.ipynb)

Linear regression works well for predicting continuous targets. However, when predicting binary outcomes (e.g., $y \in \{0, 1\}$), applying standard linear regression fails because it can produce outputs less than 0 or greater than 1, and is highly sensitive to outliers. **Logistic Regression** solves this by mapping linear predictions into probabilities using the logistic (sigmoid) function.

---

## 1. Mathematical Formulation & Derivation

Instead of modeling the binary variable $y$ directly, we model the **probability** that $y = 1$ given input vector $x$. Let this probability be:
$$p(x) = P(y = 1 \mid x)$$

### Probability, Odds, and Log-Odds

The **Odds** of an event is the ratio of the probability of the event occurring to the probability of it not occurring:
$$\text{Odds} = \frac{p(x)}{1 - p(x)}$$

If the probability of an event is $0.8$, the odds are $\frac{0.8}{0.2} = 4$ (or $4:1$).

To map this ratio (which ranges from $0$ to $\infty$) to a symmetric range $(-\infty, \infty)$ suitable for linear modeling, we take the natural logarithm, obtaining the **Log-Odds** (or **Logit** function):
$$\log(\text{Odds}) = \log\left(\frac{p(x)}{1 - p(x)}\right) = \theta_0 + \theta_1 x_1 + \cdots + \theta_p x_p = \theta^T x$$

### Derivation of the Sigmoid Function

To predict probabilities, we must invert the logit function to solve for $p(x)$ in terms of the linear combination $z = \theta^T x$:
$$\log\left(\frac{p(x)}{1 - p(x)}\right) = z$$

Exponentiate both sides:
$$\frac{p(x)}{1 - p(x)} = e^z$$

Multiply by $1 - p(x)$:
$$p(x) = e^z (1 - p(x))$$
$$p(x) = e^z - p(x) e^z$$

Group the $p(x)$ terms:
$$p(x) + p(x) e^z = e^z$$
$$p(x) (1 + e^z) = e^z$$

Solve for $p(x)$:
$$p(x) = \frac{e^z}{1 + e^z}$$

Divide numerator and denominator by $e^z$:
$$p(x) = \frac{1}{1 + e^{-z}} = \sigma(z)$$

This is the **Sigmoid Function** (or Logistic function), denoted as $\sigma(z)$.

---

## 2. Properties of the Sigmoid Function

The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ has several key properties:

- **Domain**: $(-\infty, \infty)$
- **Range**: $(0, 1)$, making it ideal for representing probabilities.
- **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$.
- **Midpoint**: $\sigma(0) = 0.5$. This acts as the standard classification threshold:
  - If $\theta^T x \ge 0 \implies \sigma(\theta^T x) \ge 0.5 \implies \text{Class } 1$
  - If $\theta^T x < 0 \implies \sigma(\theta^T x) < 0.5 \implies \text{Class } 0$

---

## 3. Classification Pipeline Architecture

The logistic regression classification workflow maps inputs to class decisions:

```mermaid
flowchart LR
    X["Input Vector X"] --> Z["Linear Combination: z = θᵀ x"]
    Z --> Sig["Sigmoid Activation: p = σ("z")"]
    Sig --> Thresh{"p ≥ 0.5?"}
    Thresh -->|Yes| C1["Class 1 (ŷ = 1)"]
    Thresh -->|No| C0["Class 0 (ŷ = 0)"]
```

---

## 4. Python Demonstration: Probability Estimation from Scratch

The following runnable Python script generates a 1D synthetic binary classification dataset, fits Scikit-Learn's `LogisticRegression`, and verifies that computing probabilities from scratch using our derived Sigmoid function matches Scikit-Learn's prediction probabilities exactly.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sigmoid function definition
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 1. Generate Synthetic 1D Classification Dataset
np.random.seed(42)
n_samples = 80
X = np.random.uniform(-4.0, 4.0, size=(n_samples, 1))
# Ground truth boundary at x = 0: y is Class 1 if x > 0 (with some noise)
y = (X[:, 0] + np.random.normal(0, 0.8, size=n_samples) > 0.0).astype(int)

# 2. Fit Logistic Regression Model
# penalty=None to fit a pure unregularized logistic model
model = LogisticRegression(penalty=None, fit_intercept=True)
model.fit(X, y)

# 3. Predict Probabilities using Scikit-Learn
probs_sklearn = model.predict_proba(X)

# 4. Predict Probabilities from Scratch using Sigmoid and Coefficients
intercept = model.intercept_[0]
coef = model.coef_[0, 0]

# Linear combination: z = theta_0 + theta_1 * x
z = intercept + coef * X[:, 0]
probs_scratch_class_1 = sigmoid(z)
# Class 0 probability is 1 - p(class_1)
probs_scratch_class_0 = 1.0 - probs_scratch_class_1

# Combine into matching shape: (n_samples, 2)
probs_scratch = np.column_stack([probs_scratch_class_0, probs_scratch_class_1])

# 5. Display Parameter Outputs & Verify Correctness
print("=== Logistic Regression Sigmoid Probability Verification ===")
print(f"Model Intercept (theta_0):  {intercept:.6f}")
print(f"Model Coefficient (theta_1): {coef:.6f}")
print("\nSample Probabilities (First 5 samples):")
print(f"{'Input X':<10} | {'Sklearn Prob (C0, C1)':<24} | {'Scratch Prob (C0, C1)':<24}")
print("-" * 65)
for i in range(5):
    print(f"{X[i, 0]:<10.4f} | [{probs_sklearn[i,0]:.5f}, {probs_sklearn[i,1]:.5f}] | [{probs_scratch[i,0]:.5f}, {probs_scratch[i,1]:.5f}]")

# Assert matching probabilities
assert np.allclose(probs_sklearn, probs_scratch, rtol=1e-10)
print("\n[SUCCESS] Custom Sigmoid probability predictions match Scikit-Learn's predict_proba outputs exactly!")
```

---

- **Next Topic**: [071_logistic_regression_part_2.md](file:///Users/prime/Developer/ml/071_logistic_regression_part_2.md) - Logistic Regression Part 2: Cost Function and Maximum Likelihood Estimation.
