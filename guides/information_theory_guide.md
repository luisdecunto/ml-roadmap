# Information Theory - Coding Guide

**Time:** 6-8 hours
**Difficulty:** Intermediate
**Prerequisites:** Python, NumPy, basic probability theory (Module 4)

## What You'll Build

Implement information theory concepts from scratch:
1. Entropy (Shannon entropy, binary entropy)
2. Cross-entropy (classification loss)
3. KL divergence (distribution distance)
4. Mutual information (shared information between variables)
5. Joint and conditional entropy
6. **Applications:** Information-theoretic analysis of data and models

---

## Project Setup

```bash
mkdir information-theory-from-scratch
cd information-theory-from-scratch

# Create files
touch information_theory.py
touch test_information_theory.py
touch requirements.txt
```

### requirements.txt
```
numpy==1.24.0
matplotlib==3.7.0
scipy==1.11.0
pandas==2.0.0
scikit-learn==1.3.0  # For comparison only
```

---

## Part 1: Information Theory Fundamentals

### Theory

**Entropy** (uncertainty):
```
H(X) = -Σ p(x) log p(x)
```

**Cross-Entropy** (cost of encoding):
```
H(p, q) = -Σ p(x) log q(x)
```

**KL Divergence** (distance between distributions):
```
D_KL(p||q) = Σ p(x) log(p(x)/q(x))
```

**Mutual Information** (shared information):
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

### Implementation

```python
# information_theory.py
import numpy as np
import matplotlib.pyplot as plt

def entropy(p, base=2):
    """
    Calculate Shannon entropy

    Args:
        p: probability distribution (must sum to 1)
        base: logarithm base (2 for bits, e for nats)
    Returns:
        H(p): entropy in bits/nats
    """
    p = np.array(p)
    # Remove zero probabilities (0 log 0 = 0)
    p = p[p > 0]

    if base == 2:
        return -np.sum(p * np.log2(p))
    elif base == np.e:
        return -np.sum(p * np.log(p))
    else:
        return -np.sum(p * np.log(p)) / np.log(base)


def cross_entropy(p, q, base=np.e):
    """
    Calculate cross-entropy H(p, q)

    Args:
        p: true distribution
        q: predicted distribution
        base: logarithm base
    Returns:
        H(p,q): cross-entropy
    """
    p = np.array(p)
    q = np.array(q)

    # Numerical stability: avoid log(0)
    q = np.clip(q, 1e-15, 1.0)

    if base == 2:
        return -np.sum(p * np.log2(q))
    elif base == np.e:
        return -np.sum(p * np.log(q))
    else:
        return -np.sum(p * np.log(q)) / np.log(base)


def kl_divergence(p, q, base=2):
    """
    Calculate KL divergence D_KL(p||q)

    Args:
        p: true distribution
        q: approximate distribution
        base: logarithm base
    Returns:
        D_KL(p||q): KL divergence
    """
    p = np.array(p)
    q = np.array(q)

    # Numerical stability
    p = np.clip(p, 1e-15, 1.0)
    q = np.clip(q, 1e-15, 1.0)

    if base == 2:
        return np.sum(p * np.log2(p / q))
    elif base == np.e:
        return np.sum(p * np.log(p / q))
    else:
        return np.sum(p * np.log(p / q)) / np.log(base)


def mutual_information(px, py, pxy):
    """
    Calculate mutual information I(X;Y)

    Args:
        px: marginal distribution of X
        py: marginal distribution of Y
        pxy: joint distribution P(X,Y)
    Returns:
        I(X;Y): mutual information
    """
    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(pxy.flatten())

    return hx + hy - hxy


def joint_entropy(pxy):
    """Calculate joint entropy H(X,Y)"""
    return entropy(pxy.flatten())


def conditional_entropy(pxy, px):
    """
    Calculate conditional entropy H(Y|X)
    H(Y|X) = H(X,Y) - H(X)
    """
    return joint_entropy(pxy) - entropy(px)


# Visualization and examples
if __name__ == "__main__":
    print("="*60)
    print("Information Theory Metrics")
    print("="*60)

    # Example 1: Entropy of coin flips
    print("\n1. Entropy of coin flips:")
    for p in [0.5, 0.7, 0.9, 0.99]:
        coin_dist = np.array([p, 1-p])
        h = entropy(coin_dist)
        print(f"   P(heads)={p:.2f}: H={h:.4f} bits")

    # Example 2: Cross-entropy
    print("\n2. Cross-entropy (classification loss):")
    p_true = np.array([1, 0, 0])  # True class is 0
    p_pred = np.array([0.7, 0.2, 0.1])  # Predicted probabilities
    ce = cross_entropy(p_true, p_pred, base=np.e)
    print(f"   True: {p_true}")
    print(f"   Pred: {p_pred}")
    print(f"   Cross-entropy: {ce:.4f} nats")

    # Example 3: KL divergence
    print("\n3. KL divergence:")
    p = np.array([0.5, 0.5])
    q1 = np.array([0.5, 0.5])  # Same as p
    q2 = np.array([0.9, 0.1])  # Different from p
    print(f"   D_KL(p||p)  = {kl_divergence(p, q1):.4f} bits (should be 0)")
    print(f"   D_KL(p||q2) = {kl_divergence(p, q2):.4f} bits")

    # Example 4: Mutual information
    print("\n4. Mutual information:")
    # Create joint distribution for two correlated variables
    pxy = np.array([[0.4, 0.1],
                    [0.1, 0.4]])
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = mutual_information(px, py, pxy)
    print(f"   I(X;Y) = {mi:.4f} bits")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Entropy vs probability
    ax1 = axes[0, 0]
    p_vals = np.linspace(0.01, 0.99, 100)
    h_vals = [entropy(np.array([p, 1-p])) for p in p_vals]
    ax1.plot(p_vals, h_vals, 'b-', linewidth=2)
    ax1.set_xlabel('P(X=1)')
    ax1.set_ylabel('H(X) [bits]')
    ax1.set_title('Binary Entropy Function')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Max entropy')
    ax1.legend()

    # Plot 2: KL divergence
    ax2 = axes[0, 1]
    p_true = 0.5
    q_vals = np.linspace(0.01, 0.99, 100)
    kl_vals = [kl_divergence(np.array([p_true, 1-p_true]),
                              np.array([q, 1-q]))
               for q in q_vals]
    ax2.plot(q_vals, kl_vals, 'g-', linewidth=2)
    ax2.set_xlabel('q (predicted probability)')
    ax2.set_ylabel('D_KL(p||q) [bits]')
    ax2.set_title(f'KL Divergence (true p={p_true})')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(p_true, color='r', linestyle='--', alpha=0.5, label='p=q (min)')
    ax2.legend()

    # Plot 3: Cross-entropy
    ax3 = axes[1, 0]
    ce_vals = [cross_entropy(np.array([p_true, 1-p_true]),
                             np.array([q, 1-q]), base=np.e)
               for q in q_vals]
    ax3.plot(q_vals, ce_vals, 'purple', linewidth=2)
    ax3.set_xlabel('q (predicted probability)')
    ax3.set_ylabel('H(p,q) [nats]')
    ax3.set_title('Cross-Entropy')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Joint distribution heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(pxy, cmap='Blues', aspect='auto')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('X')
    ax4.set_title('Joint Distribution P(X,Y)')
    plt.colorbar(im, ax=ax4)

    # Add text annotations
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            ax4.text(j, i, f'{pxy[i,j]:.2f}',
                    ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig('information_theory.png', dpi=150)
    plt.show()
```

---

## Part 2: Testing and Validation

```python
# test_information_theory.py
import numpy as np
from information_theory import (
    entropy, cross_entropy, kl_divergence,
    mutual_information, joint_entropy, conditional_entropy
)

def test_entropy():
    """Test entropy calculation"""
    print("Testing Entropy...")

    # Test 1: Uniform distribution has maximum entropy
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    h_uniform = entropy(uniform, base=2)
    print(f"✓ Uniform distribution (4 outcomes): {h_uniform:.4f} bits (should be 2.0)")
    assert np.isclose(h_uniform, 2.0), "Uniform entropy incorrect"

    # Test 2: Deterministic distribution has zero entropy
    deterministic = np.array([1.0, 0.0, 0.0, 0.0])
    h_det = entropy(deterministic, base=2)
    print(f"✓ Deterministic distribution: {h_det:.4f} bits (should be 0.0)")
    assert np.isclose(h_det, 0.0), "Deterministic entropy should be 0"

    # Test 3: Fair coin
    fair_coin = np.array([0.5, 0.5])
    h_coin = entropy(fair_coin, base=2)
    print(f"✓ Fair coin: {h_coin:.4f} bits (should be 1.0)")
    assert np.isclose(h_coin, 1.0), "Fair coin entropy should be 1"

    print("All entropy tests passed!\n")


def test_cross_entropy():
    """Test cross-entropy calculation"""
    print("Testing Cross-Entropy...")

    # Test: H(p,p) = H(p) (cross-entropy with itself equals entropy)
    p = np.array([0.6, 0.4])
    h_p = entropy(p, base=2)
    ce_pp = cross_entropy(p, p, base=2)

    print(f"✓ H(p): {h_p:.4f} bits")
    print(f"✓ H(p,p): {ce_pp:.4f} bits")
    print("✓ H(p,p) = H(p) (cross-entropy with self equals entropy)")
    assert np.isclose(h_p, ce_pp, atol=1e-6), "H(p,p) should equal H(p)"

    # Test: H(p,q) >= H(p) (cross-entropy is always >= entropy)
    q = np.array([0.8, 0.2])
    ce_pq = cross_entropy(p, q, base=2)
    print(f"✓ H(p,q): {ce_pq:.4f} bits")
    print(f"✓ H(p,q) >= H(p): {ce_pq:.4f} >= {h_p:.4f}")
    assert ce_pq >= h_p - 1e-6, "Cross-entropy should be >= entropy"

    print("All cross-entropy tests passed!\n")


def test_kl_divergence():
    """Test KL divergence calculation"""
    print("Testing KL Divergence...")

    # Test 1: D_KL(p||p) = 0
    p = np.array([0.3, 0.7])
    kl_pp = kl_divergence(p, p, base=2)
    print(f"✓ D_KL(p||p): {kl_pp:.6f} bits (should be 0)")
    assert np.isclose(kl_pp, 0.0, atol=1e-6), "D_KL(p||p) should be 0"

    # Test 2: D_KL(p||q) >= 0 (Gibbs' inequality)
    q = np.array([0.5, 0.5])
    kl_pq = kl_divergence(p, q, base=2)
    print(f"✓ D_KL(p||q): {kl_pq:.4f} bits (should be >= 0)")
    assert kl_pq >= -1e-6, "KL divergence should be non-negative"

    # Test 3: D_KL(p||q) != D_KL(q||p) (asymmetric)
    kl_qp = kl_divergence(q, p, base=2)
    print(f"✓ D_KL(q||p): {kl_qp:.4f} bits")
    print(f"✓ KL divergence is asymmetric: D_KL(p||q) ≠ D_KL(q||p)")

    # Test 4: Relationship with cross-entropy
    # D_KL(p||q) = H(p,q) - H(p)
    h_p = entropy(p, base=2)
    ce_pq = cross_entropy(p, q, base=2)
    kl_from_ce = ce_pq - h_p
    print(f"✓ D_KL from cross-entropy: {kl_from_ce:.4f} bits")
    assert np.isclose(kl_pq, kl_from_ce, atol=1e-6), "KL = H(p,q) - H(p)"

    print("All KL divergence tests passed!\n")


def test_mutual_information():
    """Test mutual information calculation"""
    print("Testing Mutual Information...")

    # Test 1: Independent variables have zero mutual information
    # P(X,Y) = P(X)P(Y)
    px = np.array([0.5, 0.5])
    py = np.array([0.6, 0.4])
    pxy_indep = np.outer(px, py)  # Independence: P(X,Y) = P(X)P(Y)

    mi_indep = mutual_information(px, py, pxy_indep)
    print(f"✓ I(X;Y) for independent variables: {mi_indep:.6f} bits (should be 0)")
    assert np.isclose(mi_indep, 0.0, atol=1e-6), "MI should be 0 for independent vars"

    # Test 2: Perfectly correlated variables
    pxy_perfect = np.array([[0.5, 0.0],
                           [0.0, 0.5]])
    px2 = pxy_perfect.sum(axis=1)
    py2 = pxy_perfect.sum(axis=0)

    mi_perfect = mutual_information(px2, py2, pxy_perfect)
    h_x = entropy(px2, base=2)
    print(f"✓ I(X;Y) for perfect correlation: {mi_perfect:.4f} bits")
    print(f"✓ H(X): {h_x:.4f} bits")
    print(f"✓ For perfect correlation: I(X;Y) = H(X) = H(Y)")
    assert np.isclose(mi_perfect, h_x, atol=1e-6), "MI should equal H(X) for perfect correlation"

    print("All mutual information tests passed!\n")


def test_conditional_entropy():
    """Test conditional entropy"""
    print("Testing Conditional Entropy...")

    # H(Y|X) = H(X,Y) - H(X)
    pxy = np.array([[0.4, 0.1],
                    [0.1, 0.4]])
    px = pxy.sum(axis=1)

    h_y_given_x = conditional_entropy(pxy, px)
    h_xy = joint_entropy(pxy)
    h_x = entropy(px, base=2)

    print(f"✓ H(Y|X): {h_y_given_x:.4f} bits")
    print(f"✓ H(X,Y) - H(X): {h_xy - h_x:.4f} bits")
    assert np.isclose(h_y_given_x, h_xy - h_x, atol=1e-6), "H(Y|X) = H(X,Y) - H(X)"

    # Chain rule: H(X,Y) = H(X) + H(Y|X)
    print(f"✓ Chain rule: H(X,Y) = H(X) + H(Y|X)")
    print(f"  {h_xy:.4f} = {h_x:.4f} + {h_y_given_x:.4f}")

    print("All conditional entropy tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Information Theory - Test Suite")
    print("="*60)
    print()

    test_entropy()
    test_cross_entropy()
    test_kl_divergence()
    test_mutual_information()
    test_conditional_entropy()

    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
```

---

## Checklist

Use this to track your progress:

### Core Implementations
- [ ] Implement `entropy()` with support for different bases
- [ ] Implement `cross_entropy()` with numerical stability
- [ ] Implement `kl_divergence()`
- [ ] Implement `mutual_information()`
- [ ] Implement `joint_entropy()` and `conditional_entropy()`

### Testing & Validation
- [ ] Test entropy: uniform distribution has maximum entropy
- [ ] Test entropy: deterministic distribution has zero entropy
- [ ] Test cross-entropy: H(p,p) = H(p)
- [ ] Test cross-entropy: H(p,q) >= H(p)
- [ ] Test KL divergence: D_KL(p||p) = 0
- [ ] Test KL divergence: D_KL(p||q) >= 0 (Gibbs' inequality)
- [ ] Test KL divergence: asymmetry D_KL(p||q) ≠ D_KL(q||p)
- [ ] Test mutual information: I(X;Y) = 0 for independent variables
- [ ] Test mutual information: I(X;Y) = H(X) for perfect correlation
- [ ] Test conditional entropy: H(Y|X) = H(X,Y) - H(X)

### Visualizations
- [ ] Plot binary entropy function
- [ ] Plot KL divergence vs predicted probability
- [ ] Plot cross-entropy vs predicted probability
- [ ] Visualize joint distribution as heatmap

### Understanding
- [ ] Explain why entropy is maximized for uniform distributions
- [ ] Explain relationship: D_KL(p||q) = H(p,q) - H(p)
- [ ] Explain why KL divergence is asymmetric
- [ ] Explain chain rule: H(X,Y) = H(X) + H(Y|X)
- [ ] Connect cross-entropy to classification loss in neural networks

---

## Resources

**Core Reading:**
- MacKay's "Information Theory, Inference, and Learning Algorithms" Ch 2
- MML Book Ch 6.6 (Exponential Family connection)
- [Visual Information Theory by colah](https://colah.github.io/posts/2015-09-Visual-Information/)

**Additional:**
- [Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- [KL Divergence Tutorial](https://eli.thegreenplace.net/2025/cross-entropy-and-kl-divergence/)
- Understanding Deep Learning Ch 5.4-5.7 (classification losses)

**Next Steps:**
- Module 6: Feedforward Neural Networks (apply cross-entropy loss)
- Module 7: Backpropagation (optimize cross-entropy)
- Advanced: Information Bottleneck theory (Tishby's work)
