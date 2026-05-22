# Mathematics of SVM: Non-linear Mapping & The Kernel Trick

Linear Support Vector Machines work well when classes are linearly separable by a hyperplane. However, in real-world scenarios, datasets are often non-linearly separable. In such cases, we can project the input data into a higher-dimensional space where a linear decision boundary exists. The **Kernel Trick** is a mathematical method that allows us to perform this higher-dimensional classification efficiently without ever explicitly calculating coordinates in the higher-dimensional space.

---

## 1. Non-linear Mapping: Concentric Circles Example

Consider a 2D dataset consisting of concentric circles. A central ring of class $0$ (red points) is surrounded by an outer ring of class $1$ (green points). No straight line can separate these two classes.

```mermaid
graph TD
    subgraph 2D Space Non-separable
        A["Data points: (x1, x2)"]
    end
    subgraph 3D Space Separable
        B["Mapped points: (x1^2, sqrt("2")*x1*x2, x2^2)"]
    end
    A -->|Feature Mapping phi| B
    B -->|Linear Hyperplane| C["Separated Classes"]
```

If we define an explicit feature mapping $\phi: \mathbb{R}^2 \to \mathbb{R}^3$ as:

$$\phi(x) = \begin{pmatrix} x_1^2 \\ \sqrt{2} x_1 x_2 \\ x_2^2 \end{pmatrix}$$

The points are projected into a 3D paraboloid space. The inner ring (small values of $x_1^2 + x_2^2$) remains low near the bottom, while the outer ring is pushed higher up. In this 3D space, a flat plane (hyperplane) can slice between the inner and outer groups, perfectly separating them.

---

## 2. The Computational Bottleneck and the Kernel Trick

While explicit projection works in theory, mapping features into high-dimensional spaces becomes computationally prohibitive. For instance:

- A polynomial mapping of degree $d$ with $p$ features results in a feature space of size $\mathcal{O}(p^d)$.
- For RBF (Radial Basis Function) kernels, the mapping is infinite-dimensional!

The **Kernel Trick** resolves this. Recall the SVM dual optimization objective function:

$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \left( x_i^T x_j \right)$$

In this dual objective, the input features $x_i$ and $x_j$ enter the optimization _only_ via their dot product (inner product) $x_i^T x_j$.

If we apply a feature mapping $\phi(x)$, the inner product in the higher-dimensional space becomes $\phi(x_i)^T \phi(x_j)$. The Kernel Trick replaces this inner product with a **Kernel Function** $K(x_i, x_j)$ that computes the inner product directly from the low-dimensional vectors:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

### Mathematical Proof for Polynomial Kernel of Degree 2

Let $x = [x_1, x_2]^T$ and $z = [z_1, z_2]^T$. The quadratic polynomial kernel function is:

$$K(x, z) = (x^T z)^2$$

Let's expand this:

$$K(x, z) = (x_1 z_1 + x_2 z_2)^2 = x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2$$

Now let's compute the inner product of their explicit 3D mappings $\phi(x)^T \phi(z)$:

$$\phi(x)^T \phi(z) = \begin{pmatrix} x_1^2 \\ \sqrt{2} x_1 x_2 \\ x_2^2 \end{pmatrix}^T \begin{pmatrix} z_1^2 \\ \sqrt{2} z_1 z_2 \\ z_2^2 \end{pmatrix} = x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2$$

They are mathematically identical! Thus, evaluating $(x^T z)^2$ (an $\mathcal{O}(p)$ operation in the low-dimensional space) gives the exact same result as explicitly transforming both vectors into 3D and computing their dot product.

---

## 3. Python Verification: Explicit 3D Mapping vs. Implicit Kernel

The following script verifies that a linear SVM trained on explicitly transformed 3D features produces identical predictions and decision boundary values to a degree-2 polynomial kernel SVM trained on the original 2D features.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# 1. Generate circular, non-linearly separable dataset
X, y = make_circles(n_samples=50, factor=0.5, noise=0.05, random_state=42)

# 2. Fit linear SVM on explicitly mapped 3D features
# phi(x) = (x1^2, sqrt(2)*x1*x2, x2^2)
phi_X = np.column_stack((X[:, 0]**2, np.sqrt(2) * X[:, 0] * X[:, 1], X[:, 1]**2))
clf_explicit = SVC(kernel='linear', C=1.0)
clf_explicit.fit(phi_X, y)
pred_explicit = clf_explicit.predict(phi_X)

# 3. Fit implicit polynomial kernel SVM of degree 2
# coef0=0.0, gamma=1.0 ensures K(u, v) = (1.0 * u^T v + 0.0)^2 = (u^T v)^2
clf_implicit = SVC(kernel='poly', degree=2, gamma=1.0, coef0=0.0, C=1.0)
clf_implicit.fit(X, y)
pred_implicit = clf_implicit.predict(X)

# 4. Compare decision boundaries and assert exact equivalence
dec_explicit = clf_explicit.decision_function(phi_X)
dec_implicit = clf_implicit.decision_function(X)

print("Explicit predictions:", pred_explicit)
print("Implicit predictions:", pred_implicit)
print("Max absolute difference in decision functions:", np.max(np.abs(dec_explicit - dec_implicit)))

# Assert identical predictions and decision scores
assert np.all(pred_explicit == pred_implicit), "Predictions do not match!"
assert np.allclose(dec_explicit, dec_implicit, atol=1e-3), "Decision functions do not match!"
print("Assertion Passed: Explicit mapping and implicit kernel yield identical results!")
```

---

## 4. Next Steps

- To understand different kernel formulations (RBF, Polynomial, Sigmoid) and how to search for their optimal hyperparameter combinations, proceed to [Day 96: SVM Kernel Hyperparameter Sweeps](file:///Users/prime/Developer/ml/096_kernel_trick_in_svm.md).
- To review SVM dual optimization derivations, refer back to [Day 94: SVM Dual Lagrangian & KKT](file:///Users/prime/Developer/ml/094_mathematics_of_support_vector_machine.md).
