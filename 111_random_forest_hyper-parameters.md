# Random Forest Hyperparameters

Random Forest is a highly flexible algorithm with many parameters that can be tuned. To make parameter tuning intuitive, we can categorize the hyperparameters into three groups: **Forest-Level Parameters** (controlling the ensemble), **Tree-Level Parameters** (controlling individual decision trees), and **System/Execution Parameters**.

---

## 1. Categorized Hyperparameters

### Forest-Level Hyperparameters

These parameters control the ensemble construction, sampling rates, and voting behaviors:

- **`n_estimators`** (default = 100): The number of decision trees in the forest. Generally, more trees improve stability and reduce variance, but increase computational cost.
- **`max_features`** (default = `"sqrt"`): The size of the random feature subset $m$ selected at each split node.
  - `"sqrt"` or `"auto"`: $m = \sqrt{p}$
  - `"log2"`: $m = \log_2(p)$
  - Float (e.g., `0.5`): $m = \lfloor 0.5 \times p \rfloor$
  - Integer (e.g., `3`): $m = 3$
  - `None`: $m = p$ (no feature subsampling, like bagging).
- **`bootstrap`** (default = `True`): Whether to build trees using bootstrap samples of the training data. If `False`, the entire dataset is used to train each tree.
- **`max_samples`** (default = `None`): If `bootstrap=True`, the number of samples to draw from $X$ to train each tree. Can be an integer or a float representing a percentage of the total dataset.

### Tree-Level Hyperparameters

These parameters are passed down to each constituent `DecisionTreeClassifier` or `DecisionTreeRegressor` to control tree growth and prevent overfitting:

- **`criterion`**: The mathematical function used to measure split quality.
  - _Classification_: `"gini"` or `"entropy"` (including `"log_loss"`).
  - _Regression_: `"squared_error"`, `"absolute_error"`, `"friedman_mse"`, or `"poisson"`.
- **`max_depth`** (default = `None`): The maximum depth of the trees. If `None`, trees are grown until leaves are pure or contain fewer than `min_samples_split` samples.
- **`min_samples_split`** (default = 2): The minimum number of samples required to split an internal node.
- **`min_samples_leaf`** (default = 1): The minimum number of samples required to be at a leaf node.
- **`max_leaf_nodes`** (default = `None`): The maximum number of leaf nodes a tree can have.
- **`min_impurity_decrease`** (default = 0.0): A node will be split if the split induces an impurity decrease greater than or equal to this value.
- **`ccp_alpha`** (default = 0.0): Complexity parameter used for Minimal Cost-Complexity Pruning.

### System & Execution Hyperparameters

- **`n_jobs`** (default = `None`): The number of CPU cores to run in parallel. Setting `n_jobs=-1` utilizes all available cores, speeding up training.
- **`oob_score`** (default = `False`): Whether to use out-of-bag samples to estimate generalization accuracy.
- **`random_state`**: Controls the random seed for bootstrapping and feature subspace selection, ensuring reproducibility.
- **`class_weight`** (default = `None`): Weights associated with classes for imbalanced classification tasks.

---

## 2. Mathematical Splitting Criteria Formulations

### Classification Criteria

- **Gini Impurity**:
  $$Gini(t) = 1 - \sum_{c=1}^C p_c^2$$
- **Entropy**:
  $$Entropy(t) = -\sum_{c=1}^C p_c \log_2(p_c)$$

### Regression Criteria

- **Mean Squared Error (MSE)**:
  $$MSE(t) = \frac{1}{N_t} \sum_{i \in t} (y_i - \bar{y}_t)^2$$
- **Mean Absolute Error (MAE)**:
  $$MAE(t) = \frac{1}{N_t} \sum_{i \in t} |y_i - \text{median}(y_t)|$$

---

## 3. Python Sweep & Performance Evaluation

Below is a self-contained Python script to train and sweep hyperparameters of a Random Forest Classifier and evaluate their impacts.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Create a synthetic classification dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=8, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Sweep over n_estimators
estimators_list = [5, 20, 100]
est_scores = []

for n in estimators_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    est_scores.append(score)
    print(f"Trees: {n:3d} | Test Accuracy: {score:.4f}")

# Verify that ensemble performance scales or remains high
assert est_scores[-1] > 0.8, "Ensemble accuracy is surprisingly low!"

# 3. Sweep over max_depth (Pruning effect)
depth_list = [1, 3, 10]
depth_scores = []

print("\nDepth Sweep:")
for d in depth_list:
    rf = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    depth_scores.append(score)
    print(f"Max Depth: {d:2d} | Test Accuracy: {score:.4f}")

# Assert that deeper trees perform better on complex datasets than very shallow (depth=1) trees
assert depth_scores[-1] > depth_scores[0], "Fully grown trees should outperform depth=1 stumps on this informative dataset!"

print("\nHyperparameter sweeping completed successfully!")
```

---

_Previous Study Guide: [Day 110: Random Forest vs Bagging & Gini MDI](file:///Users/prime/Developer/ml/110_bagging_vs_random_forest.md)_

_Next Study Guide: [Day 112: Hyperparameter Tuning (GridSearchCV)](file:///Users/prime/Developer/ml/112_hyperparameter_tuning_random_forest_using_gridsearchcv.md)_
