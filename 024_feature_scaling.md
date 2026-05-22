# Feature Scaling - Standardization

Feature Scaling is one of the final preprocessing steps in a Machine Learning pipeline, performed right before feeding data into a model. This guide covers **Standardization** (also known as **Z-Score Normalization**), which transforms data so that it has a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1.

---

## 1. What is Feature Scaling?

**Feature Scaling** is a method used to normalize the range of independent variables or features of data. In raw datasets, features often have vastly different ranges. For example:

- `Age`: $18 \text{ to } 80$ years
- `Estimated Salary`: $\$15,000 \text{ to } \$1,000,000$ per year

When features have different scales, algorithms that rely on distance or gradient descent optimization will perform poorly because the feature with the larger scale will dominate the computations.

### Standardization vs. Normalization

- **Standardization**: Centers the data around $0$ and scales it by its standard deviation. It does not bound the features to a specific range (like $0$ to $1$).
- **Normalization**: Scales the features to a fixed range (typically $[0, 1]$ or $[-1, 1]$).

---

## 2. Why Do We Need Feature Scaling?

### A. Distance-Based Algorithms

Algorithms like **K-Nearest Neighbors (KNN)**, **K-Means Clustering**, and **Support Vector Machines (SVM)** compute the Euclidean distance between data points:
$$d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}$$

If one feature (e.g., `Salary`) varies from $\$10,000$ to $\$100,000$, and another (e.g., `Age`) varies from $20$ to $60$, the difference in `Salary` will dominate the distance metric, making `Age` practically irrelevant to the model.

### B. Gradient Descent Optimization

In models like **Linear Regression**, **Logistic Regression**, and **Neural Networks**, weights are updated via Gradient Descent.

- If features are on different scales, the contour lines of the loss function will be highly elongated ellipses.
- This causes the gradient steps to oscillate heavily, slowing down convergence or preventing the model from reaching the global minimum.
- Scaling makes the contour lines more spherical, allowing gradient descent to converge directly and quickly to the minimum.

```
Unscaled Features (Elongated contours)        Scaled Features (Spherical contours)
             /       \                                     /   \
            /   / \   \                                   / / \ \
           |   | x |   |                                 | | x | |
            \   \ /   /                                   \ \ / /
             \       /                                     \   /
      (Oscillating Path)                              (Direct Path)
```

---

## 3. Mathematical Formulation of Standardization

To standardize a feature, we calculate the Z-Score for each individual data point:

$$z = \frac{x_i - \mu}{\sigma}$$

Where:

- $x_i$ is the original feature value.
- $\mu$ is the mean of the feature column: $\mu = \frac{1}{N} \sum_{i=1}^N x_i$
- $\sigma$ is the standard deviation of the feature column: $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}$
- $z$ is the standardized value.

### Mathematical Properties of Standardized Data

Once standardisation is applied, the transformed feature distribution will always have:

1. **Mean ($\mu_z$) = 0**
2. **Standard Deviation ($\sigma_z$) = 1**
3. **Variance ($\sigma_z^2$) = 1**

---

## 4. Geometric Intuition

Geometrically, standardization consists of two sequential operations:

1. **Mean Centering**:
    - Subtracting the mean ($\mu$) from each data point shifts the entire distribution along the coordinate axis so that the center (mean) of the distribution aligns exactly with the origin ($0$).
2. **Scaling**:
    - Dividing by the standard deviation ($\sigma$) compresses or stretches the distribution.
    - If the original standard deviation is greater than 1, the distribution is squashed. If it is less than 1, it is expanded. The final spread along the axis has a standard deviation of 1.

```
       Original Data                Mean Centered (Shifted)       Standardized (Scaled)
             |                                 |                             |
      *  *   *   *  *                      *   *   *   *                     * * * *
    -------------------              -------------------           ---------------------
         ^ (Mean = 35)                         ^ (Mean = 0)                  ^ (Mean = 0, Std = 1)
```

---

## 5. Algorithm Sensitivity to Scaling

Not all machine learning models require feature scaling. The table below lists model sensitivity:

| Algorithm Class                      | Needs Scaling? | Rationale                                                                                                 |
| :----------------------------------- | :------------- | :-------------------------------------------------------------------------------------------------------- |
| **K-Nearest Neighbors (KNN)**        | **Yes**        | Uses Euclidean distance; unscaled columns dominate distance calculations.                                 |
| **K-Means Clustering**               | **Yes**        | Relies on Euclidean distances from centroids.                                                             |
| **Support Vector Machines (SVM)**    | **Yes**        | Maximizes margins between support vectors; distance-dependent.                                            |
| **Principal Component Analysis**     | **Yes**        | Projects directions of maximum variance; unscaled columns look like they have artificially high variance. |
| **Logistic & Linear Regression**     | **Yes**        | Standardizing speeds up gradient descent convergence.                                                     |
| **Neural Networks / Deep Learning**  | **Yes**        | Speeds up convergence and prevents vanishing/exploding gradients.                                         |
| **Decision Trees / Random Forest**   | **No**         | Nodes split based on thresholds (e.g., $x_i > 5.5$); splits are scale-invariant.                          |
| **Gradient Boosted Trees (XGBoost)** | **No**         | Scale-invariant ordering splits.                                                                          |
| **Naive Bayes**                      | **No**         | Calculates class probabilities independently per feature.                                                 |

---

## 6. The Train-Test Split Rule (Data Leakage)

> [!IMPORTANT]
> **Crucial Rule**: Always perform the train-test split **before** applying feature scaling.

When standardizing your data:

1. Fit the scaler **only** on the training set:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
scaler.fit(X_train)
```

    This calculates and stores the training mean ($\mu_{train}$) and standard deviation ($\sigma_{train}$).

2.  Transform **both** the training set and the test set using those training parameters:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
X_test = np.array([[5.0, 6.0]])
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Why?

If you fit the scaler on the entire dataset (`X`), the test set's values will influence the calculated mean ($\mu$) and standard deviation ($\sigma$). This is a form of **Data Leakage** (specifically, distribution leakage). It leaks information about the test set into the training phase, which violates the assumption of evaluating the model on completely unseen data.

---

## 7. Hands-on Python Implementation

This code demonstration evaluates how standardization affects the performance of:

1. **Logistic Regression** (sensitive to scaling).
2. **Decision Tree Classifier** (insensitive to scaling).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Simulate a dataset (Age and Salary predicting purchase behavior)
np.random.seed(42)
n_samples = 400

age = np.random.normal(loc=37.0, scale=10.0, size=n_samples) # mean=37, std=10
salary = np.random.normal(loc=65000.0, scale=20000.0, size=n_samples) # mean=65k, std=20k

# Define a decision boundary with some noise
noise = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
purchase_prob = 0.05 * (age - 35) + 0.00005 * (salary - 60000) + noise
y = (purchase_prob > 0).astype(int)

df = pd.DataFrame({
    'Age': age,
    'EstimatedSalary': salary,
    'Purchased': y
})

print("--- Raw Dataset Stats ---")
print(df.describe().loc[['mean', 'std', 'min', 'max']], "\n")

# 2. Train-Test Split
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Apply Standardization
scaler = StandardScaler()
scaler.fit(X_train) # Fit ONLY on training data

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("--- Verifying Scaled Features ---")
print(f"X_train_scaled Mean: \n{X_train_scaled.mean().round(4)}")
print(f"X_train_scaled Std: \n{X_train_scaled.std().round(4)}\n")

# 4. Compare Logistic Regression (Sensitive Algorithm)
lr_raw = LogisticRegression(random_state=42)
lr_raw.fit(X_train, y_train)
acc_lr_raw = accuracy_score(y_test, lr_raw.predict(X_test))

lr_scaled = LogisticRegression(random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
acc_lr_scaled = accuracy_score(y_test, lr_scaled.predict(X_test_scaled))

# 5. Compare Decision Tree (Insensitive Algorithm)
dt_raw = DecisionTreeClassifier(random_state=42)
dt_raw.fit(X_train, y_train)
acc_dt_raw = accuracy_score(y_test, dt_raw.predict(X_test))

dt_scaled = DecisionTreeClassifier(random_state=42)
dt_scaled.fit(X_train_scaled, y_train)
acc_dt_scaled = accuracy_score(y_test, dt_scaled.predict(X_test_scaled))

# 6. Print Results
results = pd.DataFrame({
    'Model': ['Logistic Regression (Raw)', 'Logistic Regression (Scaled)',
              'Decision Tree (Raw)', 'Decision Tree (Scaled)'],
    'Accuracy': [acc_lr_raw, acc_lr_scaled, acc_dt_raw, acc_dt_scaled]
})
print("--- Performance Comparison ---")
print(results)
```
