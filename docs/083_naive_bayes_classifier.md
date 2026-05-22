# Naive Bayes Classifier Part 2: Independent Events

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/083_naive_bayes_classifier.ipynb)

The "Naive" assumption in the Naive Bayes classifier relies on the concept of statistical independence. Understanding independence is crucial for simplifying complex probability calculations. This guide defines independent events mathematically and verifies their properties using simulation.

---

## 1. Mathematical Definition of Independence

Two events $A$ and $B$ are **independent** if the occurrence of event $B$ does not alter the probability of event $A$ occurring.

### Conditional Formulation

$$P(A \mid B) = P(A)$$
$$P(B \mid A) = P(B)$$

_(Assuming $P(A) > 0$ and $P(B) > 0$.)_

### The Product Rule of Joint Probability

If we substitute $P(A \mid B) = P(A)$ into the general conditional probability equation:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} \implies P(A) = \frac{P(A \cap B)}{P(B)}$$

This yields the **Product Rule** for independent events:
$$P(A \cap B) = P(A) \cdot P(B)$$

This is a necessary and sufficient condition for independence. If $P(A \cap B) \ne P(A) \cdot P(B)$, the events are **dependent**.

### Physical Analogy

- **Independent**: Rolling a die, recording the result, and then rolling it again. The second roll is unaffected by the first.
- **Dependent**: Drawing a card from a deck and drawing a second card _without replacing the first_. The first draw changes the composition of the deck, altering the probability of the second draw.

---

## 2. Python Verification: Simulating Independent Events

The following runnable Python script simulates 100,000 independent trials of rolling two fair six-sided dice. It calculates the empirical probabilities of:

- Event $A$: The first die lands on $6$.
- Event $B$: The second die lands on $6$.
- Event $A \cap B$: Both dice land on $6$.

The script verifies that the joint probability converges to $P(A) \cdot P(B)$ as the number of simulations grows (Law of Large Numbers).

```python
import numpy as np

# 1. Simulate 100,000 independent rolls of two fair dice
np.random.seed(42)
n_simulations = 100000

# Generate random integers between 1 and 6 inclusive
die1_rolls = np.random.randint(1, 7, size=n_simulations)
die2_rolls = np.random.randint(1, 7, size=n_simulations)

# 2. Define Events
# Event A: First die is 6
event_A = (die1_rolls == 6)
# Event B: Second die is 6
event_B = (die2_rolls == 6)
# Event A and B: Both dice are 6
event_joint = event_A & event_B

# 3. Calculate Empirical Probabilities
p_A = np.mean(event_A)
p_B = np.mean(event_B)
p_joint_empirical = np.mean(event_joint)

# Calculate Theoretical Probabilities
p_A_theoretical = 1.0 / 6.0
p_B_theoretical = 1.0 / 6.0
p_joint_theoretical = p_A_theoretical * p_B_theoretical

print("=== Independent Events Simulation (Dice Rolls) ===")
print(f"Empirical P(A) (Die 1 = 6):            {p_A:.5f} | Theoretical: {p_A_theoretical:.5f}")
print(f"Empirical P(B) (Die 2 = 6):            {p_B:.5f} | Theoretical: {p_B_theoretical:.5f}")
print(f"Empirical P(A ∩ B) (Both = 6):          {p_joint_empirical:.5f} | Theoretical: {p_joint_theoretical:.5f}")

# Calculate product of individual empirical probabilities
product_p_A_p_B = p_A * p_B
print(f"Product P(A) * P(B):                   {product_p_A_p_B:.5f}")

# 4. Programmatic Verification
# Check if P(A ∩ B) equals P(A) * P(B) within a small tolerance
difference = np.abs(p_joint_empirical - product_p_A_p_B)
print(f"Absolute Difference:                    {difference:.6e}")

# Check if conditional probability equals marginal probability
# P(A | B) = P(A ∩ B) / P(B)
p_A_given_B = p_joint_empirical / p_B
print(f"Empirical P(A | B):                    {p_A_given_B:.5f}")

assert np.isclose(p_joint_empirical, product_p_A_p_B, atol=1e-3), "Product rule does not hold for independent events"
assert np.isclose(p_A_given_B, p_A, atol=1e-2), "Conditional probability P(A|B) does not equal marginal P(A)"
print("\n[SUCCESS] Statistical independence verified! The product rule holds and P(A|B) == P(A).")
```

---

- **Next Topic**: [084_naive_bayes_classifier.md](file:///Users/prime/Developer/ml/084_naive_bayes_classifier.md) - Naive Bayes Classifier Part 3: Mutually Exclusive Events.
