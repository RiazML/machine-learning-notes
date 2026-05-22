# Naive Bayes Classifier Part 3: Mutually Exclusive Events

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/084_naive_bayes_classifier.ipynb)

A common point of confusion in probability theory is the distinction between **Independent Events** and **Mutually Exclusive Events**. Both concepts describe relationships between events, but they are mathematically opposite. This guide defines mutually exclusive events and explores this distinction.

---

## 1. Mathematical Definition of Mutual Exclusivity

Two events $A$ and $B$ are **mutually exclusive** (or **disjoint**) if they cannot occur at the same time. In set theory terms, their intersection is the empty set:
$$A \cap B = \emptyset$$

Consequently, the joint probability of both events occurring is zero:
$$P(A \cap B) = 0$$

```mermaid
flowchart TD
    subgraph Mutually Exclusive Events (Disjoint)
        A["Event A"]
        B["Event B"]
        A -.->|Intersection is Empty| B
    end
```

### The Addition Rule

For any two events, the probability of event $A$ **or** event $B$ occurring (their union) is:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

If the events are mutually exclusive, $P(A \cap B) = 0$, simplifying the formula to:
$$P(A \cup B) = P(A) + P(B)$$

---

## 2. Independence vs. Mutual Exclusivity

- **Independence** states that the occurrence of $A$ tells us _nothing_ about the likelihood of $B$:
  $$P(B \mid A) = P(B)$$
- **Mutual Exclusivity** states that the occurrence of $A$ tells us _with absolute certainty_ that $B$ did not occur:
  $$P(B \mid A) = 0$$

If $P(A) > 0$ and $P(B) > 0$:

- If $A$ and $B$ are mutually exclusive, they **cannot** be independent.
- If $A$ and $B$ are independent, they **cannot** be mutually exclusive.

---

## 3. Python Verification: Single Die Roll Simulation

The following runnable Python script simulates 100,000 rolls of a single fair six-sided die to demonstrate mutual exclusivity and verify that disjoint events are highly dependent.

```python
import numpy as np

# 1. Simulate 100,000 rolls of a single die
np.random.seed(42)
n_rolls = 100000
rolls = np.random.randint(1, 7, size=n_rolls)

# 2. Define Events
# Event A: Roll is a 2
event_A = (rolls == 2)
# Event B: Roll is a 4
event_B = (rolls == 4)

# 3. Calculate Empirical Probabilities
p_A = np.mean(event_A)
p_B = np.mean(event_B)

# Joint Probability (Roll is both 2 and 4)
event_joint = event_A & event_B
p_joint = np.mean(event_joint)

# Union Probability (Roll is 2 or 4)
event_union = event_A | event_B
p_union = np.mean(event_union)

print("=== Mutually Exclusive Events Simulation ===")
print(f"P(A) (Roll is 2):              {p_A:.5f}")
print(f"P(B) (Roll is 4):              {p_B:.5f}")
print(f"Joint P(A ∩ B) (Both):          {p_joint:.5f} (Must be exactly 0)")
print(f"Union P(A ∪ B) (Either):        {p_union:.5f}")
print(f"Sum P(A) + P(B):               {p_A + p_B:.5f}")

# 4. Programmatic Verification of Mutual Exclusivity and Dependence
# Check addition rule holds
assert p_joint == 0.0, "Joint probability of mutually exclusive events is not zero"
assert np.isclose(p_union, p_A + p_B), "Addition rule P(A ∪ B) = P(A) + P(B) failed"
print("\n[SUCCESS] Mutual exclusivity verified: P(A ∩ B) == 0 and P(A ∪ B) == P(A) + P(B).")

# Check dependence: P(A | B) = P(A ∩ B) / P(B)
# If B occurred (roll is a 4), the probability that roll is a 2 must be 0
p_A_given_B = p_joint / p_B if p_B > 0 else 0.0
print(f"\nConditional Probability P(A | B): {p_A_given_B:.5f}")
print(f"Marginal Probability P(A):       {p_A:.5f}")

assert p_A_given_B == 0.0
assert p_A_given_B != p_A, "Events are independent, but they should be dependent"
print("[SUCCESS] Disjoint events are dependent: P(A | B) = 0, which does not equal P(A)!")
```

---

- **Next Topic**: [085_naive_bayes_classifier.md](file:///Users/prime/Developer/ml/085_naive_bayes_classifier.md) - Naive Bayes Classifier Part 4: Bayes' Theorem.
