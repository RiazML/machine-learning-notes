# Naive Bayes Classifier Part 4: Derivation of Bayes' Theorem

**Bayes' Theorem** is the mathematical core of the Naive Bayes classifier. It provides a formal framework for updating our belief in a hypothesis as new evidence or data becomes available. This guide derives the theorem and walks through a classic medical screening diagnostic application.

---

## 1. Mathematical Derivation of Bayes' Theorem

The derivation follows directly from the definition of conditional probability.

For any two events $A$ (hypothesis) and $B$ (evidence):

1. Write the conditional probability of $A$ given $B$:
    $$P(A \mid B) = \frac{P(A \cap B)}{P(B)} \implies P(A \cap B) = P(A \mid B) \cdot P(B)$$

2. Write the conditional probability of $B$ given $A$:
    $$P(B \mid A) = \frac{P(B \cap A)}{P(A)} \implies P(B \cap A) = P(B \mid A) \cdot P(A)$$

3. Since joint probability is commutative, the probability of both events occurring is the same regardless of order:
    $$P(A \cap B) = P(B \cap A)$$

4. Equate the two expressions:
    $$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

5. Solve for $P(A \mid B)$ by dividing both sides by $P(B)$ (assuming $P(B) > 0$):
    $$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

```mermaid
flowchart TD
    Prior["Prior P("A"): Belief before evidence"] --> Update["Bayes' Theorem Update"]
    Likelihood["Likelihood P("B|A"): Probability of evidence given hypothesis"] --> Update
    Evidence["Evidence P("B"): Normalizing constant"] --> Update
    Update --> Posterior["Posterior P("A|B"): Belief after evidence"]
```

### Terminology Breakdown

- **$P(A \mid B)$ (Posterior Probability)**: The updated probability of event $A$ occurring _after_ taking event $B$ into account.
- **$P(B \mid A)$ (Likelihood)**: The probability of observing event $B$ _given_ that event $A$ is true.
- **$P(A)$ (Prior Probability)**: The baseline probability of event $A$ occurring _before_ any evidence is observed.
- **$P(B)$ (Evidence / Marginal Likelihood)**: The total probability of observing event $B$ across all possible hypotheses.

---

## 2. Classic Application: Medical Diagnostic Screening

Suppose a rare disease affects $0.5\%$ of the population. A clinical test exists to screen for it:

- **Sensitivity (True Positive Rate)**: If a patient has the disease, the test is positive $99\%$ of the time ($P(\text{Positive} \mid \text{Disease}) = 0.99$).
- **Specificity (True Negative Rate)**: If a patient is healthy, the test is negative $98\%$ of the time ($P(\text{Negative} \mid \text{Healthy}) = 0.98$).
  - This implies a **False Positive Rate** of $2\%$ ($P(\text{Positive} \mid \text{Healthy}) = 0.02$).

If a randomly selected patient tests positive, what is the probability they actually have the disease?

Applying Bayes' Theorem:
$$P(\text{Disease} \mid \text{Positive}) = \frac{P(\text{Positive} \mid \text{Disease}) \cdot P(\text{Disease})}{P(\text{Positive})}$$

To find the denominator $P(\text{Positive})$, we use the Law of Total Probability:
$$P(\text{Positive}) = P(\text{Positive} \mid \text{Disease}) \cdot P(\text{Disease}) + P(\text{Positive} \mid \text{Healthy}) \cdot P(\text{Healthy})$$
$$P(\text{Positive}) = (0.99 \cdot 0.005) + (0.02 \cdot 0.995) = 0.00495 + 0.0199 = 0.02485$$

Now, calculate the posterior probability:
$$P(\text{Disease} \mid \text{Positive}) = \frac{0.99 \cdot 0.005}{0.02485} = \frac{0.00495}{0.02485} \approx 0.199197 \text{ (or } 19.92\%)$$

Despite the test being $99\%$ sensitive and $98\%$ specific, a positive test only yields a $\sim 20\%$ chance of having the disease, because the disease's prior prevalence is extremely low.

---

## 3. Python Solver Code

The following runnable Python script implements a general solver for this class of conditional diagnostic screening problems, prints the outputs, and asserts mathematical correctness.

```python
import numpy as np

# 1. Define Bayes Solver Function
def compute_posterior(prior_disease, sensitivity, specificity):
    # Disease prevalence (Prior)
    p_disease = prior_disease
    # Healthy prevalence
    p_healthy = 1.0 - prior_disease

    # Likelihoods
    p_pos_given_disease = sensitivity
    p_pos_given_healthy = 1.0 - specificity

    # Marginal Likelihood of Positive Test (Evidence) via Total Probability
    p_positive = (p_pos_given_disease * p_disease) + (p_pos_given_healthy * p_healthy)

    # Posterior Probability P(Disease | Positive)
    p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive

    return p_disease_given_positive, p_positive

# 2. Set Parameters
prior_val = 0.005  # 0.5% disease prevalence
sens_val = 0.99   # 99% sensitivity
spec_val = 0.98   # 98% specificity

# 3. Solve
posterior, total_evidence = compute_posterior(prior_val, sens_val, spec_val)

print("=== Bayes' Theorem Medical Screening Diagnostic Solver ===")
print(f"Disease Prevalence (Prior): {prior_val * 100.0:.2f}%")
print(f"Test Sensitivity (TPR):     {sens_val * 100.0:.2f}%")
print(f"Test Specificity (TNR):     {spec_val * 100.0:.2f}%")
print(f"Total Positive Rate (P(B)): {total_evidence * 100.0:.4f}%")
print(f"Posterior P(Disease|Pos):   {posterior * 100.0:.4f}%")

# 4. Programmatic Verification
# Expected posterior is 0.00495 / 0.02485 = 0.199195171
expected_posterior = 0.199195171
assert np.isclose(posterior, expected_posterior, rtol=1e-6)
print("\n[SUCCESS] Posterior probability calculated correctly and verified against standard Bayes' Theorem formula!")
```

---

- **Next Topic**: [086_naive_bayes_classifier.md](file:///Users/prime/Developer/ml/086_naive_bayes_classifier.md) - Naive Bayes Classifier Part 5: Law of Total Probability.
