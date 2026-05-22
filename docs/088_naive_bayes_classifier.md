# Naive Bayes Classifier Part 7: Log Likelihood & Laplace Smoothing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/088_naive_bayes_classifier.ipynb)

When implementing the Naive Bayes classifier in practice, we encounter two critical challenges: **numerical underflow** and the **zero-probability problem**. This guide explains how to formulate the decision boundary using log-probabilities and how to apply **Laplace Smoothing** to resolve zero-probabilities.

---

## 1. Log Likelihood Formulation

The standard Naive Bayes classification rule predicts the class label $\hat{y}$ that maximizes the posterior probability:
$$\hat{y} = \arg\max_c P(y=c) \prod_{j=1}^M P(x_j \mid y=c)$$

### The Underflow Problem

Because probabilities are numbers in the range $[0, 1]$, multiplying many features' probabilities together (e.g., $M = 1000$ words in a text classification task) leads to an extremely small number that exceeds the floating-point precision of computers, rounding down to exactly `0.0`.

### The Log-Space Solution

To prevent underflow, we take the natural logarithm of the numerator. Since the logarithm is a strictly increasing function, the value that maximizes the log-probability will also maximize the original probability:
$$\log\left(P(y=c) \prod_{j=1}^M P(x_j \mid y=c)\right) = \log P(y=c) + \sum_{j=1}^M \log P(x_j \mid y=c)$$

This transforms multiplications of small probabilities into additions of negative numbers, which are numerically stable.
$$\hat{y} = \arg\max_c \left[ \log P(y=c) + \sum_{j=1}^M \log P(x_j \mid y=c) \right]$$

---

## 2. The Zero-Probability Problem & Laplace Smoothing

If a specific feature value $v$ never appears alongside class $c$ in the training dataset, the empirical probability estimate is:
$$P(x_j = v \mid y = c) = \frac{\text{count}(x_j=v, y=c)}{\text{count}(y=c)} = 0$$

Without smoothing, if we encounter this feature value in test data, the joint probability for class $c$ becomes 0 (and in log-space, $\log(0) = -\infty$). A single unseen feature value completely overrides all other strong signals in the data.

### The Laplace Smoothing Formula

To prevent zero-probabilities, we add a small smoothing parameter $\alpha > 0$ to the numerator, and scale the denominator to compensate:
$$P(x_j = v \mid y = c) = \frac{\text{count}(x_j = v, y = c) + \alpha}{\text{count}(y = c) + \alpha \cdot V_j}$$

Where:

- $\alpha$ is the smoothing parameter. Setting $\alpha = 1$ is known as **Laplace Smoothing** (or add-one smoothing). Setting $\alpha < 1$ is **Lidstone Smoothing**.
- $V_j$ is the **cardinality** of feature $j$ (the number of unique possible values feature $j$ can take). This ensures that the smoothed probabilities still sum to exactly $1.0$ over all possible values of $x_j$.

```mermaid
flowchart TD
    FeatureCount["Count of (x_j = v, y = c)"] -->|Add α| Num["Numerator: Count + α"]
    ClassCount["Count of class y = c"] -->|Add α * V_j| Den["Denominator: ClassCount + α * V_j"]
    Num & Den --> Div["Divide"]
    Div --> Prob["Smoothed Probability P("x_j=v | y=c") > 0"]
```

---

## 3. Python Implementation from Scratch

The following runnable Python script implements a categorical Naive Bayes classifier from scratch. It calculates probabilities both with and without Laplace smoothing to show how the smoothed version handles previously unseen categories without crashing or predicting zero.

```python
import numpy as np

# 1. Custom Categorical Naive Bayes Classifier
class CategoricalNaivesBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        # Likelihood dict: class -> feature_idx -> feature_value -> probability
        self.likelihoods = {}
        self.classes = None
        self.feature_cardinalities = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Calculate feature cardinalities (V_j) for denominator scaling
        self.feature_cardinalities = [len(np.unique(X[:, j])) for j in range(n_features)]

        # 1. Compute class priors P(y = c)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
            self.likelihoods[c] = {}

            # 2. Compute class-conditional probabilities P(x_j = v | y = c)
            X_c = X[y == c]
            for j in range(n_features):
                self.likelihoods[c][j] = {}
                unique_vals = np.unique(X[:, j]) # All possible values for this feature

                for val in unique_vals:
                    count_val = np.sum(X_c[:, j] == val)
                    # Apply Laplace Smoothing formula
                    num = count_val + self.alpha
                    den = len(X_c) + self.alpha * self.feature_cardinalities[j]
                    self.likelihoods[c][j][val] = num / den

    def predict_joint_log_prob(self, X):
        n_samples, n_features = X.shape
        joint_log_probs = []

        for x in X:
            sample_log_probs = {}
            for c in self.classes:
                # Prior
                log_prob = np.log(self.class_priors[c])
                # Add conditional log-likelihoods
                for j in range(n_features):
                    val = x[j]
                    # If value was never seen in training at all, fallback to smoothing baseline
                    if val in self.likelihoods[c][j]:
                        prob = self.likelihoods[c][j][val]
                    else:
                        prob = self.alpha / (np.sum(y == c) + self.alpha * self.feature_cardinalities[j])
                    log_prob += np.log(prob)
                sample_log_probs[c] = log_prob
            joint_log_probs.append(sample_log_probs)
        return joint_log_probs

    def predict(self, X):
        joint_log_probs = self.predict_joint_log_prob(X)
        preds = []
        for probs in joint_log_probs:
            preds.append(max(probs, key=probs.get))
        return np.array(preds)

# 2. Setup Synthetic weather dataset (Outlook: 0=Sunny, 1=Overcast, 2=Rainy)
# Label: Play Tennis (0=No, 1=Yes)
X_train = np.array([
    [0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [1, 1], [0, 0]
])
# Let's say classes are:
y_train = np.array([0, 0, 1, 1, 1, 1, 0])

# Fit classifier with Laplace Smoothing (alpha=1.0)
nb_smoothed = CategoricalNaivesBayesScratch(alpha=1.0)
nb_smoothed.fit(X_train, y_train)

# Fit classifier without smoothing (alpha=0.0)
nb_unsmoothed = CategoricalNaivesBayesScratch(alpha=0.0)
nb_unsmoothed.fit(X_train, y_train)

# Test instance with unseen combination in class 0: Outlook = 1 (Overcast)
# In training, class 0 (No) has Outlook values [0, 0, 0] (never Outlook=1).
X_test = np.array([[1, 0]])

# 3. Evaluate and Compare
print("=== Unsmoothed (alpha=0) vs. Smoothed (alpha=1) Likelihoods ===")
print("Unsmoothed probability P(Outlook=Overcast | Class=No):", nb_unsmoothed.likelihoods[0][0].get(1, 0.0))
print("Smoothed probability P(Outlook=Overcast | Class=No):  ", nb_smoothed.likelihoods[0][0].get(1, 0.0))

# Assert that unsmoothed probability is 0.0 and smoothed is > 0.0
assert nb_unsmoothed.likelihoods[0][0].get(1, 0.0) == 0.0
assert nb_smoothed.likelihoods[0][0].get(1, 0.0) > 0.0
print("\n[SUCCESS] Laplace smoothing successfully prevented zero-probabilities for unseen categories!")
```

---

- **Next Topic**: [089_naive_bayes_classifier.md](file:///Users/prime/Developer/ml/089_naive_bayes_classifier.md) - Naive Bayes Classifier Part 8: Bag-of-Words Text Classification.
