# Logistic Regression Part 3: Deep Dive into Probability, Odds, Log-Odds, and Sigmoid Symmetry

In linear regression, predictions range from $-\infty$ to $\infty$. To use a linear framework for binary classification, we must map this infinite range to the interval $[0, 1]$, representing the probability of a positive class. This transition involves three key concepts: **Probability**, **Odds**, and **Log-Odds (Logit)**, culminating in the **Sigmoid function**.

---

## 1. Mathematical Concepts & Transformations

### Probability ($p$)

The probability $p$ of an event is a value in the range $[0, 1]$.
$$0 \le p \le 1$$

### Odds

The **Odds** of an event represents the ratio of the probability of occurrence to the probability of non-occurrence:
$$\text{Odds} = \frac{p}{1 - p}$$

- If $p = 0.5$, $\text{Odds} = \frac{0.5}{0.5} = 1$ (even odds, or 1:1).
- If $p \to 1$, $\text{Odds} \to \infty$.
- If $p \to 0$, $\text{Odds} \to 0$.
- The domain of Odds is $[0, \infty)$. This solves the upper-bound issue of probability but is still bounded below by 0.

### Log-Odds (Logit Function)

To map odds to a symmetric range of $(-\infty, \infty)$, we take the natural logarithm. The **logit** function of $p$ is defined as:
$$\text{logit}(p) = \log(\text{Odds}) = \log\left(\frac{p}{1 - p}\right)$$

This maps probability $p \in (0, 1)$ to a real number $z \in (-\infty, \infty)$. We can now set up a linear relationship:
$$\log\left(\frac{p}{1 - p}\right) = z = \theta^T x$$

```mermaid
flowchart TD
    Prob["Probability p ∈ (0, 1)"] -->|p / 1-p| Odds["Odds ∈ (0, ∞)"]
    Odds -->|log| LogOdds["Log-Odds (logit) ∈ (-∞, ∞)"]
    LogOdds -->|Inverse logit (Sigmoid)| Prob
```

---

## 2. Inverting the Logit to Derive the Sigmoid Function

To get the probability $p$ from the linear combination $z$, we invert the logit function:
$$\log\left(\frac{p}{1 - p}\right) = z$$

Exponentiating both sides:
$$\frac{p}{1 - p} = e^z$$

Rearranging terms to solve for $p$:
$$p = e^z(1 - p) \implies p = e^z - p \cdot e^z$$
$$p(1 + e^z) = e^z \implies p = \frac{e^z}{1 + e^z}$$

Dividing the numerator and denominator by $e^z$:
$$p = \frac{1}{1 + e^{-z}} = \sigma(z)$$

This is the **Sigmoid (or Logistic) function**.

### Mathematical Properties of Sigmoid

1. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$
    _Proof:_
    $$\sigma(-z) = \frac{1}{1 + e^{-(-z)}} = \frac{1}{1 + e^z}$$
    $$1 - \sigma(z) = 1 - \frac{1}{1 + e^{-z}} = \frac{1 + e^{-z} - 1}{1 + e^{-z}} = \frac{e^{-z}}{1 + e^{-z}}$$
    Multiply numerator and denominator by $e^z$:
    $$\frac{e^{-z} \cdot e^z}{(1 + e^{-z}) \cdot e^z} = \frac{1}{e^z + 1} = \sigma(-z)$$
    Hence, the symmetry holds.

2. **Asymptotes**:
    - $\lim_{z \to \infty} \sigma(z) = 1$
    - $\lim_{z \to -\infty} \sigma(z) = 0$

---

## 3. Python Verification & Curve Plotting Code

The following runnable Python script validates the mathematical relationships between probability, odds, and log-odds. It also programmatically verifies the Sigmoid symmetry property and displays numerical evaluations.

```python
import numpy as np

# 1. Math Function Definitions
def probability_to_odds(p):
    # Bound to prevent division by zero
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return p / (1.0 - p)

def odds_to_log_odds(odds):
    # Bound to prevent log(0)
    odds = np.clip(odds, 1e-12, None)
    return np.log(odds)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 2. Mathematical Validation & Assertion tests
print("=== Verifying Probability, Odds, and Log-Odds Transformations ===")

# Test 1: Probability of 0.5 should correspond to Odds of 1.0 and Log-Odds of 0.0
p_test = 0.5
odds_test = probability_to_odds(p_test)
log_odds_test = odds_to_log_odds(odds_test)
print(f"Prob: {p_test:.2f} -> Odds: {odds_test:.2f} -> Log-Odds: {log_odds_test:.6f}")
assert np.isclose(odds_test, 1.0)
assert np.isclose(log_odds_test, 0.0)

# Test 2: Inverting Sigmoid (mapping z back to probability)
z_vals = np.array([-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
probabilities = sigmoid(z_vals)
reconstructed_odds = probability_to_odds(probabilities)
reconstructed_z = odds_to_log_odds(reconstructed_odds)

# Assert reconstruction matches original z values
assert np.allclose(z_vals, reconstructed_z, rtol=1e-10)
print("[SUCCESS] Bidirectional transformations (Probability <-> Odds <-> Log-Odds) are perfectly consistent!")

# Test 3: Verifying the Symmetry Property: sigma(-z) == 1 - sigma(z)
for z in z_vals:
    sig_pos = sigmoid(z)
    sig_neg = sigmoid(-z)
    symmetric_val = 1.0 - sig_pos
    print(f"z = {z:>4.1f} | sigma(z) = {sig_pos:.5f} | sigma(-z) = {sig_neg:.5f} | 1 - sigma(z) = {symmetric_val:.5f}")
    assert np.isclose(sig_neg, symmetric_val, rtol=1e-12)

print("[SUCCESS] Mathematical proof of Sigmoid symmetry verified numerically!")
```

---

- **Next Topic**: [073_logistic_regression_part_4.md](file:///Users/prime/Developer/ml/073_logistic_regression_part_4.md) - Logistic Regression Part 4: Loss Function and Maximum Likelihood Estimation.
