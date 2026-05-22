# Feature Scaling - Normalization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiazML/machine-learning-notes/blob/main/notebooks/025_feature_scaling.ipynb)

Normalization is another fundamental feature scaling technique. While standardization scales data based on the mean and standard deviation, **Normalization** transforms features to a common, bounded scale without distorting differences in their ranges of values.

This guide covers the four primary types of normalization: **Min-Max Scaling**, **Mean Normalization**, **MaxAbs Scaling**, and **Robust Scaling**.

---

## 1. What is Normalization?

**Normalization** is the process of adjusting values measured on different scales to a common scale.
In machine learning, it is often used to eliminate unit differences. For instance, whether weights are measured in grams, kilograms, or pounds, normalising them yields a unitless representation on a standardized scale.

---

## 2. The Four Normalization Techniques

```mermaid
graph TD
    Norm["Normalization Techniques"] --> MMS["Min-Max Scaling<br>"Range: [0, 1"]"]
    Norm --> MN["Mean Normalization<br>"Range: [-1, 1"]"]
    Norm --> MAS["MaxAbs Scaling<br>"Range: [-1, 1"]"]
    Norm --> RS["Robust Scaling<br>Outlier Resistant"]
```

### A. Min-Max Scaling

This is the most common form of normalization. It shifts and rescales the data so that all values fall strictly within the range $[0, 1]$ (or occasionally $[-1, 1]$).

#### Mathematical Formula

$$x_{new} = \frac{x_i - x_{min}}{x_{max} - x_{min}}$$

Where:

- $x_i$ is the original feature value.
- $x_{min}$ is the minimum value in the feature column.
- $x_{max}$ is the maximum value in the feature column.
- $x_{new}$ is the scaled value ($0 \le x_{new} \le 1$).

#### Geometric Intuition

Min-Max scaling squashes the dataset into a **unit bounding box**:

- In **2D space**, the data points are compressed into a unit square of size $1 \times 1$ at the origin.
- In **3D space**, the data points are compressed into a unit cube.
- In **N-dimensional space**, they are compressed into a unit hypercube.

```
       Original Data                Min-Max Scaled [0, 1]
             |                                 |
         *   *   *                             *   *   *
    -------------------              -------------------
    min=20          max=120          min=0.0          max=1.0
```

#### Limitation (Outlier Sensitivity)

Min-Max scaling is highly sensitive to outliers. If a dataset has a single extreme outlier (e.g., $x_{max} = 1000$ while most values are around $10$), the formula will use $1000$ as $x_{max}$. This crushes all the normal data points into a tiny range (e.g., $[0.0, 0.01]$), erasing the resolution and variance of the inliers.

---

### B. Mean Normalization

Mean Normalization centers the data around $0$ by subtracting the mean, and scales it by the total range ($x_{max} - x_{min}$).

#### Mathematical Formula

$$x_{new} = \frac{x_i - \mu}{x_{max} - x_{min}}$$

Where $\mu$ is the mean of the feature. The output values fall within the range $[-1, 1]$.

> [!NOTE]
> Scikit-learn does not provide a native estimator class for Mean Normalization. If needed, you must implement it manually or via a custom transformer (`FunctionTransformer`).

---

### C. MaxAbs Scaling

MaxAbs Scaling scales each feature by its maximum absolute value.

#### Mathematical Formula

$$x_{new} = \frac{x_i}{|x|_{max}}$$

The output values fall within the range $[-1, 1]$. If the data contains only positive numbers, it is equivalent to Min-Max Scaling (scaled to $[0, 1]$).

#### Primary Use Case

MaxAbs scaling is specifically designed for **sparse datasets** (datasets containing a high percentage of zeros, such as text data vectorized via TF-IDF).

- Unlike standardization or Min-Max scaling—which subtract a mean/minimum and convert zeros into non-zero values—MaxAbs scaling only divides by the maximum absolute value.
- This ensures that **zeros remain exactly zero**, preserving the sparse matrix structure and saving significant memory and computational time.

---

### D. Robust Scaling

Robust Scaling is designed to handle datasets with significant outliers. Instead of using the mean, minimum, or maximum, it uses the **median** and the **Interquartile Range (IQR)**, which are statistically robust to extreme values.

#### Mathematical Formula

$$x_{new} = \frac{x_i - \text{median}}{IQR} = \frac{x_i - Q_2(x)}{Q_3(x) - Q_1(x)}$$

Where:

- $Q_1(x)$ is the 25th percentile (1st quartile).
- $Q_2(x)$ is the 50th percentile (median).
- $Q_3(x)$ is the 75th percentile (3rd quartile).
- $IQR = Q_3(x) - Q_1(x)$ (the range containing the middle 50% of the data).

#### Why is it Robust?

Since percentiles are calculated based on the rank/ordering of values rather than their magnitude, extreme values at the tail ends do not affect $Q_1$, $Q_2$, or $Q_3$.

- The majority of the data (the inliers) is scaled to a normal range.
- The outliers themselves will still be scaled, but they will fall outside the typical $[-1, 1]$ range (e.g., $15.4$ or $-23.1$), preventing them from crushing the variance of the inliers.

---

## 3. Scaler Decision Matrix

Use this matrix to select the appropriate feature scaling technique:

| Scaler             | Scikit-Learn Class | Bounded Range?  | Outlier Sensitive? | Best Use Cases                                                                                                                           |
| :----------------- | :----------------- | :-------------- | :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **StandardScaler** | `StandardScaler`   | No              | Yes                | Default choice. Best for Gaussian/normal-like distributions. Ideal for SVM, Logistic Regression, Linear Regression, and Neural Networks. |
| **MinMaxScaler**   | `MinMaxScaler`     | Yes ($[0, 1]$)  | Yes                | Best when features are not Gaussian, or when you have known boundaries (e.g., Image Pixels: $[0, 255]$ scaled to $[0, 1]$).              |
| **MaxAbsScaler**   | `MaxAbsScaler`     | Yes ($[-1, 1]$) | Yes                | Specifically for sparse data matrices (preserves zero-values).                                                                           |
| **RobustScaler**   | `RobustScaler`     | No              | **No**             | Use when the dataset contains significant outliers that you cannot remove.                                                               |

---

## 4. Hands-on Python Implementation

This code block demonstrates how `RobustScaler` protects the representation of inliers compared to `MinMaxScaler` when extreme outliers are present in the dataset.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# 1. Simulate a dataset with extreme outliers
np.random.seed(42)
n_samples = 200

# Normal clean data (mostly clustered between 10 and 20)
clean_data = np.random.normal(loc=15.0, scale=2.0, size=n_samples)

# Add a few extreme outliers
outliers = np.array([150.0, 300.0, 500.0])
data = np.concatenate([clean_data, outliers])

df = pd.DataFrame({'Feature': data})
print("--- Raw Feature Statistics (Including Outliers) ---")
print(df.describe().loc[['mean', 'std', 'min', 'max']], "\n")

# 2. Train-Test Split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# 3. Fit and Transform using MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X_train)
X_train_minmax = pd.DataFrame(minmax_scaler.transform(X_train), columns=['Feature'])

# 4. Fit and Transform using RobustScaler
robust_scaler = RobustScaler()
robust_scaler.fit(X_train)
X_train_robust = pd.DataFrame(robust_scaler.transform(X_train), columns=['Feature'])

# 5. Compare the standard deviation and spread of the clean inlier range
# Inliers are data points originally < 30
inlier_indices = X_train['Feature'] < 30.0

comparison = pd.DataFrame({
    'Metric': ['Min Value of Inliers', 'Max Value of Inliers', 'Std Dev of Inliers'],
    'Original (Unscaled)': [
        X_train.loc[inlier_indices, 'Feature'].min(),
        X_train.loc[inlier_indices, 'Feature'].max(),
        X_train.loc[inlier_indices, 'Feature'].std()
    ],
    'MinMaxScaler': [
        X_train_minmax.loc[inlier_indices.values, 'Feature'].min(),
        X_train_minmax.loc[inlier_indices.values, 'Feature'].max(),
        X_train_minmax.loc[inlier_indices.values, 'Feature'].std()
    ],
    'RobustScaler': [
        X_train_robust.loc[inlier_indices.values, 'Feature'].min(),
        X_train_robust.loc[inlier_indices.values, 'Feature'].max(),
        X_train_robust.loc[inlier_indices.values, 'Feature'].std()
    ]
})

print("--- Scaling Comparison for Inliers ---")
print(comparison.round(4))
print("\nNotice that MinMaxScaler compresses all inliers into a tiny range (0.01 to 0.03) with a tiny standard deviation.")
print("RobustScaler maintains a healthy variance and readable range for the inliers.")
```
