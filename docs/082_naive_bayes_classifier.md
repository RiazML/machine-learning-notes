# Naive Bayes Classifier Part 1: Conditional Probability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/082_naive_bayes_classifier.ipynb)

The **Naive Bayes Classifier** is a probabilistic machine learning algorithm rooted in probability theory. To understand how it operates, we must first master its mathematical foundations. The first of these is **Conditional Probability**.

---

## 1. Mathematical Formulation

Conditional probability measures the probability of an event $A$ occurring, **given** that another event $B$ has already occurred. This is denoted as $P(A \mid B)$ (read: "probability of $A$ given $B$").

```mermaid
flowchart TD
    TotalSample["Total Sample Space (S)"] --> EventB["Event B (New Sample Space)"]
    EventB --> Intersection["Intersection of A and B (A ∩ B)"]
    Intersection --> Formula["P("A|B") = P("A ∩ B") / P("B")"]
```

### The Conditional Probability Formula

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Where:

- $P(A \cap B)$ is the joint probability that **both** event $A$ and event $B$ occur.
- $P(B)$ is the marginal probability that event $B$ occurs, satisfying $P(B) > 0$.

### Asymmetry of Conditional Probability: $P(A \mid B) \ne P(B \mid A)$

A common logical fallacy is to assume that $P(A \mid B) = P(B \mid A)$. This is generally false.

- _Example_: Let $A$ be "a person is a professional basketball player" and $B$ be "a person is tall".
  - $P(B \mid A)$: The probability that someone is tall given they are a pro basketball player is very close to $1.0$ (nearly all are tall).
  - $P(A \mid B)$: The probability that someone is a pro basketball player given they are tall is extremely small (there are millions of tall people, but very few are in the NBA).
- **Proof**:
  $$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
  $$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$
  Therefore, $P(A \mid B) = P(B \mid A)$ **only if** $P(A) = P(B)$.

---

## 2. Python Demonstration: Computing Probabilities from Contingency Tables

A contingency table cross-tabulates frequency distributions of variables. The following runnable Python script builds a contingency table of mock emails (classified by whether they are Spam and whether they contain the word "Offer") and computes conditional probabilities to demonstrate asymmetry.

```python
import pandas as pd
import numpy as np

# 1. Create a synthetic dataset of 1000 emails
np.random.seed(42)
n_emails = 1000

# True labels: 0 = Ham (normal email), 1 = Spam
# Assume 20% of all emails are Spam
is_spam = np.random.binomial(n=1, p=0.2, size=n_emails)

# Feature: Contains the word "Offer"
# Spam is highly likely to contain "Offer" (75%), Ham is less likely (15%)
has_offer = np.zeros(n_emails, dtype=int)
for i in range(n_emails):
    prob_offer = 0.75 if is_spam[i] == 1 else 0.15
    has_offer[i] = np.random.binomial(n=1, p=prob_offer)

# Create DataFrame
df = pd.DataFrame({'Spam': is_spam, 'Offer': has_offer})

# 2. Build Contingency Table (Cross-Tabulation)
contingency_table = pd.crosstab(df['Spam'], df['Offer'], margins=True)
print("=== Contingency Table (Frequencies) ===")
print(contingency_table)
print("\n" + "="*40 + "\n")

# 3. Calculate Conditional Probabilities
# P(Spam = 1 | Offer = 1) = Count(Spam=1 and Offer=1) / Count(Offer=1)
count_both = contingency_table.loc[1, 1]
count_offer = contingency_table.loc['All', 1]
p_spam_given_offer = count_both / count_offer

# P(Offer = 1 | Spam = 1) = Count(Spam=1 and Offer=1) / Count(Spam=1)
count_spam = contingency_table.loc[1, 'All']
p_offer_given_spam = count_both / count_spam

# Marginal Probabilities
p_spam = count_spam / n_emails
p_offer = count_offer / n_emails

print("=== Calculated Probabilities ===")
print(f"P(Spam) (Prior):                 {p_spam:.4f}")
print(f"P(Offer):                       {p_offer:.4f}")
print(f"P(Spam | Offer) (Posterior):     {p_spam_given_offer:.4f}")
print(f"P(Offer | Spam) (Likelihood):    {p_offer_given_spam:.4f}")

# 4. Programmatic Verification
# Assert that conditional probabilities are asymmetric: P(A|B) != P(B|A)
print(f"\nIs P(Spam|Offer) != P(Offer|Spam)? {p_spam_given_offer != p_offer_given_spam}")
assert not np.isclose(p_spam_given_offer, p_offer_given_spam), "Conditional probabilities are symmetric, but they should be asymmetric here"

# Verify using the formula P(Spam|Offer) = P(Offer|Spam) * P(Spam) / P(Offer)
calculated_via_formula = (p_offer_given_spam * p_spam) / p_offer
assert np.isclose(p_spam_given_offer, calculated_via_formula)
print("[SUCCESS] Asymmetry demonstrated, and formula verified against contingency table frequencies!")
```

---

- **Next Topic**: [083_naive_bayes_classifier.md](file:///Users/prime/Developer/ml/083_naive_bayes_classifier.md) - Naive Bayes Classifier Part 2: Independent Events.
