# What are Tensors?

In machine learning and deep learning, **Tensors** are the primary data structures used to store and manipulate numbers. Almost all data (images, text, audio, video) is converted into tensors before being passed to an ML algorithm.

---

## 1. What is a Tensor?

Mathematically, a tensor is a container for numerical data. It is a generalization of scalars, vectors, and matrices to an arbitrary number of dimensions.
In computer science, a tensor is simply a **multi-dimensional array** (represented in Python by NumPy's `ndarray`, PyTorch's `Tensor`, or TensorFlow's `Tensor`).

```
  0D Tensor       1D Tensor          2D Tensor                 3D Tensor
  (Scalar)        (Vector)           (Matrix)                   (Cube)
    ┌──┐        ┌──┬──┬──┐        ┌──┬──┬──┐            ┌──┬──┬──┐
    │ 5 │        │ 1│ 2│ 3│        │ 1│ 2│ 3│           / 1/ 2/ 3/│
    └──┘        └──┴──┴──┘        ├──┼──┼──┤          ┌──┬──┬──┐ │
                                  │ 4│ 5│ 6│          │ 4│ 5│ 6│─┘
                                  └──┴──┴──┘          └──┴──┴──┘
```

---

## 2. The Tensor Hierarchy (Ranks 0 to 5)

A tensor is defined by its **Rank** (number of axes or dimensions) and its **Shape**.

### Rank 0 Tensor: Scalar

A scalar contains a single, isolated number. It has $0$ axes (dimensions).

- **NumPy Representation**:

  ```python
  import numpy as np
  scalar = np.array(5)
  print(scalar.ndim)  # Output: 0 (Rank)
  print(scalar.shape) # Output: () (Shape)
  ```

### Rank 1 Tensor: Vector

A vector is an array of numbers. It has exactly $1$ axis.

- **NumPy Representation**:

  ```python
  vector = np.array([1.2, 3.4, 5.6])
  print(vector.ndim)  # Output: 1 (Rank)
  print(vector.shape) # Output: (3,) (Shape)
  ```

### Rank 2 Tensor: Matrix

A matrix is a 2D grid/table of numbers, containing rows and columns. It has $2$ axes.

- **NumPy Representation**:

  ```python
  matrix = np.array([[1, 2, 3],
                     [4, 5, 6]])
  print(matrix.ndim)  # Output: 2 (Rank)
  print(matrix.shape) # Output: (2, 3) (Shape)
  ```

### Rank 3 Tensor: 3D Tensor

A 3D tensor is a cube of numbers, formed by stacking multiple 2D matrices. It has $3$ axes.

- **Real-World Example**: A single color image (RGB). The three dimensions are height, width, and color channels (Red, Green, Blue).
- **NumPy Representation**:

  ```python
  # A 3x3 pixel color image
  image_3d = np.random.rand(3, 3, 3)
  print(image_3d.ndim)  # Output: 3 (Rank)
  print(image_3d.shape) # Output: (3, 3, 3) (Shape)
  ```

### Rank 4 Tensor: 4D Tensor

A 4D tensor is a vector/batch of 3D tensors. It has $4$ axes.

- **Real-World Example**: A batch of RGB color images passed to a neural network for training.
- **Shape representation**: `(batch_size, height, width, channels)`

  ```python
  # A batch of 32 images, each 256x256 pixels with 3 color channels
  image_batch = np.random.rand(32, 256, 256, 3)
  print(image_batch.ndim)  # Output: 4 (Rank)
  print(image_batch.shape) # Output: (32, 256, 256, 3) (Shape)
  ```

### Rank 5 Tensor: 5D Tensor

A 5D tensor is a collection of 4D tensors. It has $5$ axes.

- **Real-World Example**: Video data. A video consists of a sequence of frames over time.
- **Shape representation**: `(batch_size, frames, height, width, channels)`

  ```python
  # A batch of 4 video clips, each with 150 frames, 1080x1920 pixels, 3 channels
  video_batch = np.random.rand(4, 150, 1080, 1920, 3)
  print(video_batch.ndim)  # Output: 5 (Rank)
  print(video_batch.shape) # Output: (4, 150, 1080, 1920, 3) (Shape)
  ```

---

## 3. Mathematical Definitions & Terminologies

1. **Axes (Singular: Axis)**: The individual directions/dimensions of the tensor. A 2D matrix has axis 0 (rows) and axis 1 (columns).
2. **Rank (ndim)**: The total number of axes. For instance, a 3D tensor has rank 3.
3. **Shape**: A tuple of integers representing the size of the tensor along each axis. For example, a shape of `(3, 2)` represents 3 rows and 2 columns.
4. **Data Type (dtype)**: The type of numbers stored in the tensor (e.g., float32, int64, uint8). All elements in a tensor must share the same data type to support optimized parallel hardware operations (CPU/GPU vectorization).

---

## 4. Case Study: Memory Footprint of Raw Video Data

Let's calculate the memory required to load and process a small video dataset in RAM.

### Dataset Specifications

- **Number of video clips (Batch Size)**: $4$
- **Video Duration**: $60\text{ seconds}$
- **Frame Rate (FPS)**: $30\text{ frames per second}$
- **Resolution**: $480 \times 720\text{ pixels}$ (Height $\times$ Width)
- **Color Format**: RGB ($3\text{ channels}$)

### Step 1: Calculate Total Frames per Video

$$\text{Total Frames} = 60\text{ seconds} \times 30\text{ frames/second} = 1,800\text{ frames}$$

### Step 2: Formulate the 5D Tensor Shape

$$\text{Shape} = (\text{Batch Size}, \text{Frames}, \text{Height}, \text{Width}, \text{Channels}) = (4, 1800, 480, 720, 3)$$

### Step 3: Compute the Total Number of Elements

$$\text{Elements} = 4 \times 1800 \times 480 \times 720 \times 3 = 7,464,960,000\text{ float values}$$

### Step 4: Calculate Memory Size in Bytes (assuming 32-bit Float / Float32)

Each Float32 value occupies **$4\text{ bytes}$** of memory.
$$\text{Total Bytes} = 7,464,960,000 \times 4\text{ bytes} = 29,859,840,000\text{ bytes}$$

### Step 5: Convert Bytes to Gigabytes (GB)

$$\text{Kilobytes (KB)} = \frac{29,859,840,000}{1024} \approx 29,159,999.6\text{ KB}$$
$$\text{Megabytes (MB)} = \frac{29,159,999.6}{1024} \approx 28,476.5\text{ MB}$$
$$\text{Gigabytes (GB)} = \frac{28,476.5}{1024} \approx \mathbf{27.81\text{ GB}}$$

> [!WARNING]
> Processing this small 4-video batch in its raw uncompressed state requires **$27.8\text{ GB}$ of RAM**! This is why raw video arrays are rarely loaded directly in deep learning models. Instead, we use compressed video formats (like H.264/MP4) and stream frames incrementally into RAM using data generators.
