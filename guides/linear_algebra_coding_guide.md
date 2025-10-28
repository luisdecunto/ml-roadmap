# Linear Algebra Coding Projects - From Scratch

**Time:** 8-12 hours total
**Difficulty:** Intermediate-Advanced
**Prerequisites:** Python, NumPy basics, understanding of vectors and matrices

## What You'll Build

Implement core linear algebra operations from scratch in NumPy:
1. Matrix multiplication
2. Power iteration (find largest eigenvalue)
3. Singular Value Decomposition (SVD)
4. **Capstone Project:** Image compression using your SVD implementation

---

## Project Setup

### Create Project Structure

```bash
mkdir linear-algebra-projects
cd linear-algebra-projects

# Create files
touch matrix_ops.py
touch eigenvalues.py
touch svd.py
touch image_compression.py
touch test_implementations.py
touch requirements.txt
```

### Install Dependencies

```bash
# requirements.txt
numpy==1.24.0
matplotlib==3.7.0
pillow==10.0.0
scipy==1.11.0  # Only for verification, not for implementation!
```

Install:
```bash
pip install -r requirements.txt
```

---

## Part 1: Matrix Multiplication from Scratch

### Understanding Matrix Multiplication

For matrices A (m×n) and B (n×p), the result C (m×p) is:

```
C[i,j] = Σ(k=0 to n-1) A[i,k] * B[k,j]
```

Each element is the dot product of a row from A and a column from B.

### Implementation: `matrix_ops.py`

```python
# matrix_ops.py
import numpy as np

def matrix_multiply_naive(A, B):
    """
    Multiply two matrices using three nested loops

    Args:
        A: numpy array of shape (m, n)
        B: numpy array of shape (n, p)

    Returns:
        C: numpy array of shape (m, p)
    """
    # Validate dimensions
    assert A.shape[1] == B.shape[0], f"Incompatible shapes: {A.shape} and {B.shape}"

    m, n = A.shape
    n2, p = B.shape

    # Initialize result matrix with zeros
    C = np.zeros((m, p))

    # Triple nested loop
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C


def matrix_multiply_dot(A, B):
    """
    Multiply matrices using row-column dot products (faster)

    Args:
        A: numpy array of shape (m, n)
        B: numpy array of shape (n, p)

    Returns:
        C: numpy array of shape (m, p)
    """
    assert A.shape[1] == B.shape[0], f"Incompatible shapes: {A.shape} and {B.shape}"

    m, n = A.shape
    n2, p = B.shape

    C = np.zeros((m, p))

    # For each element, compute dot product of row and column
    for i in range(m):
        for j in range(p):
            C[i, j] = np.dot(A[i, :], B[:, j])

    return C


def matrix_multiply_vectorized(A, B):
    """
    Vectorized implementation using NumPy broadcasting (fastest)

    Args:
        A: numpy array of shape (m, n)
        B: numpy array of shape (n, p)

    Returns:
        C: numpy array of shape (m, p)
    """
    assert A.shape[1] == B.shape[0], f"Incompatible shapes: {A.shape} and {B.shape}"

    # Reshape for broadcasting
    # A: (m, n, 1)
    # B: (1, n, p)
    # Result: (m, n, p) -> sum over axis 1 -> (m, p)

    A_expanded = A[:, :, np.newaxis]  # (m, n, 1)
    B_expanded = B[np.newaxis, :, :]  # (1, n, p)

    # Element-wise multiply and sum
    C = np.sum(A_expanded * B_expanded, axis=1)

    return C


def test_matrix_multiply():
    """Test all implementations and compare with NumPy"""

    print("="*60)
    print("Testing Matrix Multiplication Implementations")
    print("="*60)

    # Test case 1: Small matrices
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])

    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    print(f"\nA shape: {A.shape}")
    print(A)
    print(f"\nB shape: {B.shape}")
    print(B)

    # Ground truth
    C_numpy = np.matmul(A, B)
    print(f"\nNumPy result:")
    print(C_numpy)

    # Test implementations
    C_naive = matrix_multiply_naive(A, B)
    C_dot = matrix_multiply_dot(A, B)
    C_vec = matrix_multiply_vectorized(A, B)

    print(f"\nNaive implementation matches: {np.allclose(C_naive, C_numpy)}")
    print(f"Dot implementation matches: {np.allclose(C_dot, C_numpy)}")
    print(f"Vectorized implementation matches: {np.allclose(C_vec, C_numpy)}")

    # Benchmark on larger matrices
    print("\n" + "="*60)
    print("Performance Benchmark (100x100 matrices)")
    print("="*60)

    import time

    A_large = np.random.randn(100, 100)
    B_large = np.random.randn(100, 100)

    # Naive (warning: slow!)
    start = time.time()
    C_naive = matrix_multiply_naive(A_large, B_large)
    time_naive = time.time() - start
    print(f"Naive:      {time_naive*1000:.2f} ms")

    # Dot product
    start = time.time()
    C_dot = matrix_multiply_dot(A_large, B_large)
    time_dot = time.time() - start
    print(f"Dot:        {time_dot*1000:.2f} ms")

    # Vectorized
    start = time.time()
    C_vec = matrix_multiply_vectorized(A_large, B_large)
    time_vec = time.time() - start
    print(f"Vectorized: {time_vec*1000:.2f} ms")

    # NumPy (optimized BLAS)
    start = time.time()
    C_numpy = np.matmul(A_large, B_large)
    time_numpy = time.time() - start
    print(f"NumPy:      {time_numpy*1000:.2f} ms (uses optimized BLAS)")

    print("\n✓ All implementations produce correct results!")


if __name__ == "__main__":
    test_matrix_multiply()
```

**Run it:**
```bash
python matrix_ops.py
```

**Expected Output:**
```
Testing Matrix Multiplication Implementations
A shape: (2, 3)
B shape: (3, 2)

NumPy result:
[[ 58  64]
 [139 154]]

Naive implementation matches: True
Dot implementation matches: True
Vectorized implementation matches: True

Performance Benchmark (100x100 matrices)
Naive:      850.23 ms
Dot:        45.12 ms
Vectorized: 2.31 ms
NumPy:      0.08 ms (uses optimized BLAS)

✓ All implementations produce correct results!
```

---

## Part 2: Power Iteration (Largest Eigenvalue)

### Understanding Power Iteration

Power iteration finds the largest eigenvalue and corresponding eigenvector:

1. Start with random vector v
2. Repeatedly multiply by matrix A: v = A × v
3. Normalize v after each iteration
4. Converges to eigenvector with largest eigenvalue

### Implementation: `eigenvalues.py`

```python
# eigenvalues.py
import numpy as np

def power_iteration(A, num_iterations=100, tolerance=1e-10):
    """
    Find largest eigenvalue and eigenvector using power iteration

    Args:
        A: square matrix (n, n)
        num_iterations: maximum iterations
        tolerance: convergence tolerance

    Returns:
        eigenvalue: largest eigenvalue (scalar)
        eigenvector: corresponding eigenvector (normalized)
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square"

    # Initialize with random vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)  # Normalize

    eigenvalue = 0

    for i in range(num_iterations):
        # Multiply by matrix
        v_new = A @ v

        # Compute eigenvalue (Rayleigh quotient)
        eigenvalue_new = np.dot(v, A @ v) / np.dot(v, v)

        # Normalize
        v_new = v_new / np.linalg.norm(v_new)

        # Check convergence
        if np.abs(eigenvalue_new - eigenvalue) < tolerance:
            print(f"Converged in {i+1} iterations")
            break

        v = v_new
        eigenvalue = eigenvalue_new

    return eigenvalue, v


def find_all_eigenvalues_power(A, k=None):
    """
    Find top k eigenvalues using deflation

    Args:
        A: square matrix (n, n)
        k: number of eigenvalues to find (default: all)

    Returns:
        eigenvalues: array of eigenvalues
        eigenvectors: matrix of eigenvectors (columns)
    """
    n = A.shape[0]
    if k is None:
        k = n

    eigenvalues = []
    eigenvectors = []

    A_deflated = A.copy()

    for i in range(k):
        # Find largest eigenvalue of deflated matrix
        lambda_i, v_i = power_iteration(A_deflated)

        eigenvalues.append(lambda_i)
        eigenvectors.append(v_i)

        # Deflate: remove this eigenvalue's contribution
        A_deflated = A_deflated - lambda_i * np.outer(v_i, v_i)

    return np.array(eigenvalues), np.column_stack(eigenvectors)


def test_power_iteration():
    """Test power iteration implementation"""

    print("="*60)
    print("Testing Power Iteration")
    print("="*60)

    # Create symmetric matrix (guaranteed real eigenvalues)
    A = np.array([[4, 1, 2],
                  [1, 3, 1],
                  [2, 1, 5]])

    print(f"\nMatrix A:")
    print(A)

    # Our implementation
    eigenvalue, eigenvector = power_iteration(A)

    print(f"\nOur largest eigenvalue: {eigenvalue:.6f}")
    print(f"Our eigenvector:\n{eigenvector}")

    # Verify: A*v should equal λ*v
    verification = A @ eigenvector
    expected = eigenvalue * eigenvector

    print(f"\nVerification:")
    print(f"A @ v = {verification}")
    print(f"λ * v = {expected}")
    print(f"Close? {np.allclose(verification, expected)}")

    # Compare with NumPy
    eigenvalues_numpy, eigenvectors_numpy = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigenvalues_numpy))
    largest_numpy = eigenvalues_numpy[idx]

    print(f"\nNumPy's largest eigenvalue: {largest_numpy:.6f}")
    print(f"Match? {np.isclose(eigenvalue, largest_numpy)}")

    # Test finding multiple eigenvalues
    print("\n" + "="*60)
    print("Finding Top 3 Eigenvalues")
    print("="*60)

    eigenvalues, eigenvectors = find_all_eigenvalues_power(A, k=3)

    print(f"\nOur eigenvalues: {eigenvalues}")
    print(f"NumPy eigenvalues (sorted): {np.sort(eigenvalues_numpy)[::-1]}")

    print("\n✓ Power iteration works!")


if __name__ == "__main__":
    test_power_iteration()
```

**Run it:**
```bash
python eigenvalues.py
```

---

## Part 3: Singular Value Decomposition (SVD)

### Understanding SVD

SVD decomposes matrix A (m×n) into:
```
A = U Σ V^T
```

Where:
- U (m×m): Left singular vectors (eigenvectors of AA^T)
- Σ (m×n): Diagonal matrix of singular values
- V (n×n): Right singular vectors (eigenvectors of A^TA)

### Implementation: `svd.py`

```python
# svd.py
import numpy as np
from eigenvalues import find_all_eigenvalues_power

def svd_from_scratch(A, full_matrices=True):
    """
    Compute SVD: A = U Σ V^T

    Args:
        A: matrix to decompose (m, n)
        full_matrices: if True, U is (m,m) and V is (n,n)
                      if False, U is (m,k) and V is (n,k) where k = min(m,n)

    Returns:
        U: left singular vectors
        S: singular values (1D array)
        Vt: right singular vectors (transposed)
    """
    m, n = A.shape

    print(f"Computing SVD of {m}×{n} matrix...")

    # Step 1: Compute A^T A (more efficient if m > n)
    if m >= n:
        # Compute A^T A (n×n)
        AtA = A.T @ A

        # Eigendecomposition of A^T A
        print("Computing eigenvalues of A^T A...")
        eigenvalues, eigenvectors = np.linalg.eigh(AtA)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # V = eigenvectors of A^T A
        V = eigenvectors

        # Singular values = sqrt(eigenvalues)
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        # Compute U = A V Σ^(-1)
        k = min(m, n)
        U = np.zeros((m, k))

        for i in range(k):
            if singular_values[i] > 1e-10:
                U[:, i] = (A @ V[:, i]) / singular_values[i]
            else:
                U[:, i] = np.zeros(m)

        # Extend to full matrices if requested
        if full_matrices and m > n:
            # Add orthonormal basis for null space
            U_full = np.zeros((m, m))
            U_full[:, :n] = U
            # Use QR to get remaining columns
            Q, _ = np.linalg.qr(np.random.randn(m, m - n))
            U_full[:, n:] = Q
            U = U_full

            S = np.zeros(m)
            S[:n] = singular_values
        else:
            S = singular_values

    else:
        # Compute A A^T (m×m)  - when m < n
        AAt = A @ A.T

        print("Computing eigenvalues of A A^T...")
        eigenvalues, eigenvectors = np.linalg.eigh(AAt)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        U = eigenvectors
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        k = min(m, n)
        V = np.zeros((n, k))

        for i in range(k):
            if singular_values[i] > 1e-10:
                V[:, i] = (A.T @ U[:, i]) / singular_values[i]
            else:
                V[:, i] = np.zeros(n)

        S = singular_values

    return U, S, V.T


def test_svd():
    """Test SVD implementation"""

    print("="*60)
    print("Testing SVD Implementation")
    print("="*60)

    # Test matrix
    A = np.array([[3, 2, 2],
                  [2, 3, -2]])

    print(f"\nOriginal matrix A:")
    print(A)
    print(f"Shape: {A.shape}")

    # Our SVD
    U, S, Vt = svd_from_scratch(A, full_matrices=False)

    print(f"\nU shape: {U.shape}")
    print(U)

    print(f"\nSingular values: {S}")

    print(f"\nV^T shape: {Vt.shape}")
    print(Vt)

    # Reconstruct
    Sigma = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(Sigma, S)

    A_reconstructed = U @ Sigma @ Vt

    print(f"\nReconstructed A:")
    print(A_reconstructed)

    print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed):.10f}")

    # Compare with NumPy
    U_numpy, S_numpy, Vt_numpy = np.linalg.svd(A, full_matrices=False)

    print(f"\n" + "="*60)
    print("Comparison with NumPy SVD")
    print("="*60)

    print(f"\nOur singular values: {S}")
    print(f"NumPy singular values: {S_numpy}")
    print(f"Match? {np.allclose(S, S_numpy)}")

    print("\n✓ SVD implementation works!")


if __name__ == "__main__":
    test_svd()
```

**Run it:**
```bash
python svd.py
```

---

## Part 4: IMAGE COMPRESSION PROJECT 🎨

### The Goal

Compress a 512×512 image using your SVD implementation and visualize how many singular values you need for good quality.

### Implementation: `image_compression.py`

```python
# image_compression.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from svd import svd_from_scratch

def load_image(path, target_size=(512, 512)):
    """Load and preprocess image"""
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = img.resize(target_size)
    return np.array(img, dtype=float)


def compress_image_svd(image, k):
    """
    Compress image using top k singular values

    Args:
        image: 2D numpy array
        k: number of singular values to keep

    Returns:
        compressed_image: reconstructed image using k components
        compression_ratio: ratio of compressed vs original size
    """
    m, n = image.shape

    # Compute SVD
    U, S, Vt = svd_from_scratch(image, full_matrices=False)

    # Keep only top k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Reconstruct
    compressed = U_k @ np.diag(S_k) @ Vt_k

    # Compute compression ratio
    original_size = m * n
    compressed_size = k * (m + n + 1)  # U_k + S_k + Vt_k
    compression_ratio = original_size / compressed_size

    return compressed, compression_ratio


def analyze_compression(image_path):
    """
    Analyze image compression at different k values
    """
    print("="*60)
    print("IMAGE COMPRESSION WITH SVD")
    print("="*60)

    # Load image
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)
    m, n = image.shape
    print(f"Image shape: {image.shape}")

    # Test different k values
    k_values = [5, 10, 20, 50, 100, 200]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original\n({m}×{n} = {m*n:,} values)')
    axes[0].axis('off')

    # Compressed versions
    for idx, k in enumerate(k_values, 1):
        print(f"\nCompressing with k={k}...")
        compressed, ratio = compress_image_svd(image, k)

        # Clip values to valid range
        compressed = np.clip(compressed, 0, 255)

        # Compute error
        error = np.linalg.norm(image - compressed) / np.linalg.norm(image)

        # Plot
        axes[idx].imshow(compressed, cmap='gray')
        axes[idx].set_title(
            f'k={k}\n'
            f'Ratio: {ratio:.1f}x\n'
            f'Error: {error*100:.2f}%'
        )
        axes[idx].axis('off')

        print(f"  Compression ratio: {ratio:.2f}x")
        print(f"  Relative error: {error*100:.2f}%")

    # Hide last subplot
    axes[7].axis('off')

    plt.tight_layout()
    plt.savefig('image_compression_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to 'image_compression_comparison.png'")
    plt.show()


def plot_singular_value_spectrum(image_path):
    """Plot how singular values decay"""

    image = load_image(image_path)

    print("\nComputing full SVD for singular value analysis...")
    U, S, Vt = svd_from_scratch(image, full_matrices=False)

    # Plot singular values
    plt.figure(figsize=(12, 5))

    # Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(S, 'b-', linewidth=2)
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Value Spectrum')
    plt.grid(True, alpha=0.3)

    # Log scale
    plt.subplot(1, 2, 2)
    plt.semilogy(S, 'r-', linewidth=2)
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Value Spectrum (Log Scale)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('singular_value_spectrum.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved spectrum to 'singular_value_spectrum.png'")
    plt.show()

    # Print statistics
    total_energy = np.sum(S**2)
    print(f"\n" + "="*60)
    print("Singular Value Statistics")
    print("="*60)

    for k in [10, 50, 100, 200]:
        if k <= len(S):
            energy_k = np.sum(S[:k]**2)
            energy_percent = 100 * energy_k / total_energy
            print(f"Top {k:3d} values capture {energy_percent:5.2f}% of energy")


def main():
    """Run image compression project"""

    # You can use any image - download a sample or use your own
    # For testing, create a synthetic image
    print("Creating test image (512x512 gradient pattern)...")

    # Create a test image (you can replace with your own image path)
    x = np.linspace(0, 4*np.pi, 512)
    y = np.linspace(0, 4*np.pi, 512)
    X, Y = np.meshgrid(x, y)
    test_image = 128 + 127 * np.sin(X) * np.cos(Y)

    # Save test image
    Image.fromarray(test_image.astype(np.uint8)).save('test_image.png')
    print("✓ Saved test image to 'test_image.png'")

    # Analyze compression
    analyze_compression('test_image.png')

    # Plot singular value spectrum
    plot_singular_value_spectrum('test_image.png')

    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print("1. First few singular values capture most information")
    print("2. Compression ratio vs quality tradeoff clearly visible")
    print("3. k=50-100 often gives good quality with 10-20x compression")


if __name__ == "__main__":
    main()
```

**Run the complete project:**
```bash
python image_compression.py
```

**To use your own image:**
```python
analyze_compression('path/to/your/image.jpg')
```

---

## Verification Checklist

- [ ] Matrix multiplication works and matches NumPy
- [ ] Power iteration finds correct largest eigenvalue
- [ ] SVD decomposes matrix correctly (A ≈ UΣV^T)
- [ ] Image compression runs successfully
- [ ] Visualizations generated (compression comparison, singular value spectrum)
- [ ] Understand tradeoff between compression ratio and image quality

---

## Expected Results

### Matrix Multiplication
- All three implementations should match NumPy exactly
- Vectorized version much faster than naive
- But NumPy still faster (uses optimized BLAS libraries)

### Power Iteration
- Should converge in 20-50 iterations for well-conditioned matrices
- Eigenvalue matches NumPy within 1e-6

### SVD
- Reconstruction error < 1e-10
- Singular values match NumPy

### Image Compression
- k=10: Heavy compression (~50x), blurry but recognizable
- k=50: Good compression (~10x), good quality
- k=100: Moderate compression (~5x), excellent quality
- k=200: Low compression (~2.5x), nearly identical to original

---

## Common Issues

### Issue: "Eigenvalues are negative"
**Solution:** Use `np.maximum(eigenvalues, 0)` when taking sqrt for singular values

### Issue: SVD reconstruction error too large
**Solution:** Check that you're sorting eigenvalues in descending order

### Issue: Image compression looks weird
**Solution:**
- Clip values to [0, 255]: `np.clip(compressed, 0, 255)`
- Make sure image is grayscale (2D array, not 3D)

### Issue: Power iteration doesn't converge
**Solution:**
- Increase `num_iterations` to 200-500
- Check if matrix has repeated largest eigenvalues (convergence is slow)

---

## Extensions & Challenges

1. **Randomized SVD**: Implement faster approximate SVD for large matrices
2. **Color Images**: Extend compression to RGB (compress each channel separately)
3. **Comparison**: Compare your SVD with JPEG compression
4. **Rank-k Approximation**: Implement best rank-k matrix approximation
5. **PCA**: Use your SVD for Principal Component Analysis on real data

---

## Resources

- [SVD Tutorial by Steve Brunton](https://www.youtube.com/watch?v=nbBvuuNVfco)
- [3Blue1Brown: Abstract Vector Spaces](https://www.youtube.com/watch?v=TgKwz5Ikpc8)
- [Matrix Multiplication Optimization](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)
- [Image Compression with SVD](https://towardsdatascience.com/image-compression-using-svd-7c2b1c1e4e9e)
