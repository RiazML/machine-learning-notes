# Naive Bayes Classifier Part 5: The Law of Total Probability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/086_naive_bayes_classifier.ipynb)

To calculate the denominator (evidence $P(B)$) in Bayes' Theorem for complex datasets, we must use the **Law of Total Probability**. This law allows us to find the marginal probability of an event by summing its joint probabilities across a set of partitioning sub-events.

---

## 1. Mathematical Formulation

Let the sample space $S$ be partitioned into $K$ mutually exclusive and collectively exhaustive events $A_1, A_2, \ldots, A_K$.
This means:

1. **Mutually Exclusive**: $A_i \cap A_j = \emptyset$ for all $i \ne j$ (events do not overlap).
2. **Collectively Exhaustive**: $A_1 \cup A_2 \cup \cdots \cup A_K = S$ (the events cover the entire sample space).

For any event $B$ in the same sample space:
$$P(B) = \sum_{k=1}^K P(B \cap A_k)$$

Using the definition of conditional probability ($P(B \cap A_k) = P(B \mid A_k) \cdot P(A_k)$), we obtain the **Law of Total Probability**:
$$P(B) = \sum_{k=1}^K P(B \mid A_k) \cdot P(A_k)$$

```mermaid
flowchart TD
    subgraph Sample Space Partitioning
        A1["Partition A₁"]
        A2["Partition A₂"]
        AK["Partition A_K"]
    end
    A1 -->|P("B|A₁")P("A₁")| Sum["Summing Joint Probabilities P("B ∩ A_k")"]
    A2 -->|P("B|A₂")P("A₂")| Sum
    AK -->|P("B|A_K")P("A_K")| Sum
    Sum --> Result["Total Marginal Probability P("B")"]
```

### Connection to Bayes' Theorem

For any single partition $A_j$, we can find its posterior probability given evidence $B$ by expanding the denominator using the Law of Total Probability:
$$P(A_j \mid B) = \frac{P(B \mid A_j) \cdot P(A_j)}{\sum_{k=1}^K P(B \mid A_k) \cdot P(A_k)}$$

---

## 2. Practical Application: Factory Defect Analysis

Suppose a factory uses three machines ($M_1$, $M_2$, and $M_3$) to manufacture components.

- **Production Shares (Priors)**:
  - $M_1$ produces $50\%$ of all parts ($P(M_1) = 0.50$).
  - $M_2$ produces $30\%$ of all parts ($P(M_2) = 0.30$).
  - $M_3$ produces $20\%$ of all parts ($P(M_3) = 0.20$).
- **Defect Rates (Likelihoods)**:
  - $M_1$ parts are defective $1\%$ of the time ($P(D \mid M_1) = 0.01$).
  - $M_2$ parts are defective $2\%$ of the time ($P(D \mid M_2) = 0.02$).
  - $M_3$ parts are defective $3\%$ of the time ($P(D \mid M_3) = 0.03$).

### Question 1: What is the total probability that a randomly selected part is defective ($P(D)$)?

Applying the Law of Total Probability:
$$P(D) = P(D \mid M_1) \cdot P(M_1) + P(D \mid M_2) \cdot M_2 + P(D \mid M_3) \cdot M_3$$
$$P(D) = (0.01 \cdot 0.50) + (0.02 \cdot 0.30) + (0.03 \cdot 0.20) = 0.005 + 0.006 + 0.006 = 0.017 \text{ (or } 1.7\%)$$

### Question 2: If a selected part is defective, what is the probability it came from $M_3$?

Applying the expanded Bayes' Theorem:
$$P(M_3 \mid D) = \frac{P(D \mid M_3) \cdot P(M_3)}{P(D)} = \frac{0.03 \cdot 0.20}{0.017} = \frac{0.006}{0.017} \approx 0.3529 \text{ (or } 35.29\%)$$

---

## 3. Python Solver Code

The following runnable Python script implements the factory defect calculation, outputs probabilities for all partitions, and asserts that the posterior probabilities sum to exactly 1.0.

```python
import numpy as np

# 1. Define inputs
priors = np.array([0.50, 0.30, 0.20])       # P(M1), P(M2), P(M3)
defect_rates = np.array([0.01, 0.02, 0.03]) # P(D|M1), P(D|M2), P(D|M3)

# 2. Compute Total Probability of defect P(D)
# P(D) = sum(P(D|M_i) * P(M_i))
total_defect_probability = np.sum(defect_rates * priors)

# 3. Compute Posterior Probabilities for each machine
# P(M_i|D) = P(D|M_i) * P(M_i) / P(D)
posteriors = (defect_rates * priors) / total_defect_probability

# 4. Display Results
print("=== Total Probability & Multiclass Bayes Solver ===")
print(f"Total Defect Probability P(D): {total_defect_probability * 100.0:.2f}%")
print("\nPosterior probabilities of source machine given a defective component:")
for i in range(len(priors)):
    print(f"  - Machine M{i+1}: {posteriors[i] * 100.0:.2f}%")

# 5. Programmatic Verification
# Check that P(D) matches theoretical value 0.017
assert np.isclose(total_defect_probability, 0.017)
# Check that posterior probabilities sum to 1.0 (collectively exhaustive)
posterior_sum = np.sum(posteriors)
print(f"\nSum of Posteriors:              {posterior_sum:.6f}")
assert np.isclose(posterior_sum, 1.0)
print("[SUCCESS] Law of Total Probability verified, and posterior distribution is mathematically consistent!")
```

---

- **Next Topic**: [087_naive_bayes_classifier.md](file:///Users/prime/Developer/ml/087_naive_bayes_classifier.md) - Naive Bayes Classifier Part 6: The "Naive" Assumption.
