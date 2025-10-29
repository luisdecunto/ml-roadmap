# Information Theory Solutions - Module 5

Comprehensive solutions with step-by-step mathematical work and code implementations.

---

## Part 1: Entropy Basics

### Exercise 1.1 Solution: Basic Entropy Calculation

A fair coin (H or T), P(H) = P(T) = 0.5

**Entropy formula:** H(X) = -∑ p(x) log₂ p(x)

**Step 1: Calculate entropy**
```
H(X) = -[P(H)·log₂ P(H) + P(T)·log₂ P(T)]
     = -[0.5·log₂(0.5) + 0.5·log₂(0.5)]
     = -[0.5·(-1) + 0.5·(-1)]
     = -[-0.5 - 0.5]
     = 1 bit
```

**Interpretation:** A fair coin has maximum entropy of 1 bit. This means we need exactly 1 bit to encode the outcome on average, and there's maximum uncertainty before the flip.

**NumPy verification:**
```python
import numpy as np

def entropy(probs):
    """Calculate entropy H(X) = -∑ p(x) log₂ p(x)"""
    # Filter out zero probabilities to avoid log(0)
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# Fair coin
p_fair = [0.5, 0.5]
H_fair = entropy(p_fair)
print(f"Fair coin entropy: {H_fair:.4f} bits")
# Output: 1.0000 bits
```

---

### Exercise 1.2 Solution: Entropy of Biased Coin

Biased coin with P(H) = 0.8, P(T) = 0.2

**Step 1: Calculate entropy**
```
H(X) = -[0.8·log₂(0.8) + 0.2·log₂(0.2)]
     = -[0.8·(-0.322) + 0.2·(-2.322)]
     = -[-0.258 - 0.464]
     = 0.722 bits
```

**Step 2: Compare with fair coin**
```
Fair coin:   H = 1.000 bits (maximum)
Biased coin: H = 0.722 bits (lower)

The biased coin has less uncertainty/entropy!
```

**Interpretation:** When outcomes are unequal, entropy decreases. The more predictable the system, the lower the entropy. Maximum entropy occurs when all outcomes are equally likely.

**NumPy verification:**
```python
# Biased coin
p_biased = [0.8, 0.2]
H_biased = entropy(p_biased)
print(f"Biased coin entropy: {H_biased:.4f} bits")
print(f"Reduction from fair: {H_fair - H_biased:.4f} bits")

# Visualize entropy for different bias levels
import matplotlib.pyplot as plt

p_values = np.linspace(0.01, 0.99, 100)
H_values = [entropy([p, 1-p]) for p in p_values]

plt.figure(figsize=(8, 5))
plt.plot(p_values, H_values)
plt.axvline(0.5, color='r', linestyle='--', label='Fair coin (max entropy)')
plt.axvline(0.8, color='g', linestyle='--', label='Biased coin (p=0.8)')
plt.xlabel('P(Heads)')
plt.ylabel('Entropy (bits)')
plt.title('Entropy of Biased Coin')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Exercise 1.3 Solution: Entropy of Die Roll

Fair 6-sided die, each outcome has probability 1/6

**Step 1: Calculate entropy**
```
H(X) = -∑ᵢ₌₁⁶ (1/6)·log₂(1/6)
     = -6 × (1/6)·log₂(1/6)
     = -log₂(1/6)
     = log₂(6)
     ≈ 2.585 bits
```

**Step 2: Why more than coin?**
```
Coin: 2 outcomes → H = log₂(2) = 1 bit
Die:  6 outcomes → H = log₂(6) ≈ 2.585 bits

More outcomes = more uncertainty = higher entropy
```

**General formula:** For n equally likely outcomes: H(X) = log₂(n)

**NumPy verification:**
```python
# Fair 6-sided die
p_die = [1/6] * 6
H_die = entropy(p_die)
print(f"Fair die entropy: {H_die:.4f} bits")
print(f"Theoretical: {np.log2(6):.4f} bits")

# Compare different dice
for n_sides in [2, 4, 6, 8, 10, 20]:
    p = [1/n_sides] * n_sides
    H = entropy(p)
    print(f"{n_sides:2d}-sided die: H = {H:.4f} bits = log₂({n_sides})")
```

---

## Part 2: Joint and Conditional Entropy

### Exercise 2.1 Solution: Joint Entropy

Joint distribution P(X, Y):
```
      Y=0   Y=1
X=0   0.2   0.3
X=1   0.3   0.2
```

**Step 1: Calculate H(X, Y)**
```
H(X,Y) = -∑ₓ ∑ᵧ P(x,y)·log₂ P(x,y)
       = -[0.2·log₂(0.2) + 0.3·log₂(0.3) + 0.3·log₂(0.3) + 0.2·log₂(0.2)]
       = -[0.2·(-2.322) + 0.3·(-1.737) + 0.3·(-1.737) + 0.2·(-2.322)]
       = -[-0.464 - 0.521 - 0.521 - 0.464]
       = 1.970 bits
```

**Step 2: Calculate marginal H(X)**
```
P(X=0) = 0.2 + 0.3 = 0.5
P(X=1) = 0.3 + 0.2 = 0.5

H(X) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)] = 1 bit
```

**Step 3: Calculate marginal H(Y)**
```
P(Y=0) = 0.2 + 0.3 = 0.5
P(Y=1) = 0.3 + 0.2 = 0.5

H(Y) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)] = 1 bit
```

**Step 4: Compare**
```
H(X,Y) = 1.970 bits
H(X) + H(Y) = 1 + 1 = 2 bits

H(X,Y) < H(X) + H(Y) because X and Y are not independent!
The difference (0.03 bits) is the mutual information I(X;Y)
```

**NumPy implementation:**
```python
# Joint distribution
joint_probs = np.array([
    [0.2, 0.3],  # X=0
    [0.3, 0.2]   # X=1
])

# Joint entropy
H_XY = entropy(joint_probs.flatten())
print(f"H(X,Y) = {H_XY:.4f} bits")

# Marginal distributions
p_X = joint_probs.sum(axis=1)  # Sum over Y
p_Y = joint_probs.sum(axis=0)  # Sum over X

H_X = entropy(p_X)
H_Y = entropy(p_Y)

print(f"H(X) = {H_X:.4f} bits")
print(f"H(Y) = {H_Y:.4f} bits")
print(f"H(X) + H(Y) = {H_X + H_Y:.4f} bits")
print(f"Difference: {(H_X + H_Y) - H_XY:.4f} bits (mutual information)")
```

---

### Exercise 2.2 Solution: Conditional Entropy

Using the same joint distribution from Exercise 2.1

**Formula:** H(Y|X) = H(X,Y) - H(X)

**Step 1: Use chain rule**
```
H(Y|X) = H(X,Y) - H(X)
       = 1.970 - 1.000
       = 0.970 bits
```

**Step 2: Verify by direct calculation**
```
H(Y|X) = ∑ₓ P(x)·H(Y|X=x)

For X=0:
P(Y=0|X=0) = 0.2/0.5 = 0.4
P(Y=1|X=0) = 0.3/0.5 = 0.6
H(Y|X=0) = -[0.4·log₂(0.4) + 0.6·log₂(0.6)]
         = -[-0.529 - 0.442]
         = 0.971 bits

For X=1:
P(Y=0|X=1) = 0.3/0.5 = 0.6
P(Y=1|X=1) = 0.2/0.5 = 0.4
H(Y|X=1) = 0.971 bits (symmetric)

H(Y|X) = 0.5·0.971 + 0.5·0.971 = 0.971 bits ✓
```

**Interpretation:** Knowing X reduces uncertainty about Y from H(Y) = 1 bit to H(Y|X) = 0.97 bits. The reduction is small because X and Y are weakly dependent.

**NumPy implementation:**
```python
# Method 1: Chain rule
H_Y_given_X = H_XY - H_X
print(f"H(Y|X) via chain rule: {H_Y_given_X:.4f} bits")

# Method 2: Direct calculation
def conditional_entropy(joint_probs):
    """Calculate H(Y|X) = ∑ₓ P(x)·H(Y|X=x)"""
    p_X = joint_probs.sum(axis=1)
    H_cond = 0

    for i, p_x in enumerate(p_X):
        if p_x > 0:
            # Conditional distribution P(Y|X=i)
            p_Y_given_X = joint_probs[i, :] / p_x
            H_Y_given_Xi = entropy(p_Y_given_X)
            H_cond += p_x * H_Y_given_Xi

    return H_cond

H_Y_given_X_direct = conditional_entropy(joint_probs)
print(f"H(Y|X) direct: {H_Y_given_X_direct:.4f} bits")

# Information gain
print(f"Information gain: H(Y) - H(Y|X) = {H_Y - H_Y_given_X:.4f} bits")
```

---

### Exercise 2.3 Solution: Chain Rule Verification

**Chain Rule:** H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)

Using our joint distribution:

**Step 1: Calculate H(X) + H(Y|X)**
```
H(X) = 1.000 bits (from Exercise 2.1)
H(Y|X) = 0.970 bits (from Exercise 2.2)
H(X) + H(Y|X) = 1.970 bits
```

**Step 2: Calculate H(Y) + H(X|Y)**
```
H(Y) = 1.000 bits (from Exercise 2.1)
H(X|Y) = H(X,Y) - H(Y) = 1.970 - 1.000 = 0.970 bits
H(Y) + H(X|Y) = 1.970 bits
```

**Step 3: Verify all equal**
```
H(X,Y) = 1.970 bits
H(X) + H(Y|X) = 1.970 bits ✓
H(Y) + H(X|Y) = 1.970 bits ✓

Chain rule verified!
```

**NumPy implementation:**
```python
# Calculate all quantities
H_X_given_Y = H_XY - H_Y

print("Chain Rule Verification:")
print(f"H(X,Y) = {H_XY:.4f} bits")
print(f"H(X) + H(Y|X) = {H_X + H_Y_given_X:.4f} bits")
print(f"H(Y) + H(X|Y) = {H_Y + H_X_given_Y:.4f} bits")
print(f"All equal? {np.allclose([H_XY, H_X + H_Y_given_X, H_Y + H_X_given_Y])}")
```

---

## Part 3: Mutual Information

### Exercise 3.1 Solution: Mutual Information Calculation

**Formula:** I(X;Y) = H(X) + H(Y) - H(X,Y)

Using our joint distribution:

**Step 1: Calculate I(X;Y)**
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
       = 1.000 + 1.000 - 1.970
       = 0.030 bits
```

**Step 2: Alternative formula**
```
I(X;Y) = H(X) - H(X|Y)
       = 1.000 - 0.970
       = 0.030 bits ✓

I(X;Y) = H(Y) - H(Y|X)
       = 1.000 - 0.970
       = 0.030 bits ✓
```

**Interpretation:** X and Y share 0.03 bits of information. This is small, indicating weak dependence. If they were independent, I(X;Y) = 0. If perfectly correlated, I(X;Y) = 1 bit.

**NumPy implementation:**
```python
def mutual_information(joint_probs):
    """Calculate I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    # Marginal distributions
    p_X = joint_probs.sum(axis=1)
    p_Y = joint_probs.sum(axis=0)

    # Entropies
    H_X = entropy(p_X)
    H_Y = entropy(p_Y)
    H_XY = entropy(joint_probs.flatten())

    return H_X + H_Y - H_XY

I_XY = mutual_information(joint_probs)
print(f"I(X;Y) = {I_XY:.4f} bits")

# Verify alternative formulas
I_XY_alt1 = H_X - H_X_given_Y
I_XY_alt2 = H_Y - H_Y_given_X

print(f"I(X;Y) = H(X) - H(X|Y) = {I_XY_alt1:.4f} bits")
print(f"I(X;Y) = H(Y) - H(Y|X) = {I_XY_alt2:.4f} bits")
```

---

### Exercise 3.2 Solution: Independent Variables

For independent X and Y: P(X,Y) = P(X)·P(Y)

**Example:** X ~ {0, 1} with P(X=0) = 0.3, Y ~ {0, 1} with P(Y=0) = 0.6

**Step 1: Joint distribution**
```
P(0,0) = 0.3 × 0.6 = 0.18
P(0,1) = 0.3 × 0.4 = 0.12
P(1,0) = 0.7 × 0.6 = 0.42
P(1,1) = 0.7 × 0.4 = 0.28
```

**Step 2: Calculate entropies**
```
H(X) = -[0.3·log₂(0.3) + 0.7·log₂(0.7)]
     = -[-0.521 - 0.360]
     = 0.881 bits

H(Y) = -[0.6·log₂(0.6) + 0.4·log₂(0.4)]
     = -[-0.442 - 0.529]
     = 0.971 bits

H(X,Y) = -[0.18·log₂(0.18) + 0.12·log₂(0.12) +
           0.42·log₂(0.42) + 0.28·log₂(0.28)]
       = 1.852 bits
```

**Step 3: Calculate I(X;Y)**
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
       = 0.881 + 0.971 - 1.852
       = 0.000 bits ✓

For independent variables, I(X;Y) = 0!
```

**NumPy verification:**
```python
# Independent variables
p_X_indep = np.array([0.3, 0.7])
p_Y_indep = np.array([0.6, 0.4])

# Joint distribution (outer product)
joint_indep = np.outer(p_X_indep, p_Y_indep)

print("Joint distribution (independent):")
print(joint_indep)

I_indep = mutual_information(joint_indep)
print(f"\nI(X;Y) for independent variables: {I_indep:.6f} bits")
print(f"Essentially zero? {np.isclose(I_indep, 0)}")
```

---

### Exercise 3.3 Solution: Perfectly Correlated Variables

Y = X (perfect correlation)

**Step 1: Joint distribution**
```
      Y=0   Y=1
X=0   0.5   0.0
X=1   0.0   0.5
```

**Step 2: Calculate entropies**
```
H(X) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)] = 1 bit
H(Y) = 1 bit (same as X)

H(X,Y) = -[0.5·log₂(0.5) + 0·log₂(0) + 0·log₂(0) + 0.5·log₂(0.5)]
       = -[0.5·(-1) + 0.5·(-1)]
       = 1 bit

(Note: 0·log₂(0) = 0 by convention)
```

**Step 3: Mutual information**
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
       = 1 + 1 - 1
       = 1 bit

I(X;Y) = H(X) = H(Y) for perfect correlation!
```

**Interpretation:** When Y = X, knowing one variable completely determines the other. The mutual information equals the entropy of either variable.

**NumPy implementation:**
```python
# Perfect correlation Y = X
joint_perfect = np.array([
    [0.5, 0.0],
    [0.0, 0.5]
])

I_perfect = mutual_information(joint_perfect)
p_X_perfect = joint_perfect.sum(axis=1)
H_X_perfect = entropy(p_X_perfect)

print(f"I(X;Y) for Y=X: {I_perfect:.4f} bits")
print(f"H(X): {H_X_perfect:.4f} bits")
print(f"I(X;Y) = H(X)? {np.isclose(I_perfect, H_X_perfect)}")

# Conditional entropy should be 0
H_Y_given_X_perfect = entropy(joint_perfect.flatten()) - H_X_perfect
print(f"H(Y|X) for Y=X: {H_Y_given_X_perfect:.6f} bits (should be 0)")
```

---

## Part 4: Cross-Entropy and KL Divergence

### Exercise 4.1 Solution: Cross-Entropy

True distribution: p = [0.5, 0.5]
Model distribution: q = [0.7, 0.3]

**Cross-entropy formula:** H(p, q) = -∑ₓ p(x)·log₂ q(x)

**Step 1: Calculate H(p, q)**
```
H(p,q) = -[p(0)·log₂ q(0) + p(1)·log₂ q(1)]
       = -[0.5·log₂(0.7) + 0.5·log₂(0.3)]
       = -[0.5·(-0.515) + 0.5·(-1.737)]
       = -[-0.258 - 0.868]
       = 1.126 bits
```

**Step 2: Compare with H(p)**
```
H(p) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)] = 1.000 bit

H(p,q) = 1.126 bits > H(p) = 1.000 bit

Cross-entropy is always ≥ entropy, with equality when p = q
```

**Step 3: Interpretation**
The cross-entropy measures the expected number of bits needed to encode events from p using code optimized for q. Since q ≠ p, we need more bits (1.126) than optimal (1.000).

**NumPy implementation:**
```python
def cross_entropy(p, q):
    """Calculate H(p,q) = -∑ p(x)·log₂ q(x)"""
    p = np.array(p)
    q = np.array(q)
    # Avoid log(0)
    return -np.sum(p * np.log2(q + 1e-10))

p_true = np.array([0.5, 0.5])
q_model = np.array([0.7, 0.3])

H_p = entropy(p_true)
H_pq = cross_entropy(p_true, q_model)

print(f"H(p) = {H_p:.4f} bits")
print(f"H(p,q) = {H_pq:.4f} bits")
print(f"Extra bits due to model mismatch: {H_pq - H_p:.4f} bits")
```

---

### Exercise 4.2 Solution: KL Divergence

**KL Divergence formula:** D_KL(p||q) = ∑ₓ p(x)·log₂(p(x)/q(x))

Using p = [0.5, 0.5] and q = [0.7, 0.3]:

**Step 1: Calculate D_KL(p||q)**
```
D_KL(p||q) = p(0)·log₂(p(0)/q(0)) + p(1)·log₂(p(1)/q(1))
           = 0.5·log₂(0.5/0.7) + 0.5·log₂(0.5/0.3)
           = 0.5·log₂(0.714) + 0.5·log₂(1.667)
           = 0.5·(-0.485) + 0.5·(0.737)
           = -0.243 + 0.368
           = 0.126 bits
```

**Step 2: Relationship to cross-entropy**
```
D_KL(p||q) = H(p,q) - H(p)
           = 1.126 - 1.000
           = 0.126 bits ✓
```

**Step 3: Calculate D_KL(q||p) (reverse)**
```
D_KL(q||p) = 0.7·log₂(0.7/0.5) + 0.3·log₂(0.3/0.5)
           = 0.7·log₂(1.4) + 0.3·log₂(0.6)
           = 0.7·(0.485) + 0.3·(-0.737)
           = 0.340 - 0.221
           = 0.119 bits

D_KL(p||q) ≠ D_KL(q||p)
KL divergence is NOT symmetric!
```

**NumPy implementation:**
```python
def kl_divergence(p, q):
    """Calculate D_KL(p||q) = ∑ p(x)·log₂(p(x)/q(x))"""
    p = np.array(p)
    q = np.array(q)
    # Avoid division by zero
    return np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))

D_pq = kl_divergence(p_true, q_model)
D_qp = kl_divergence(q_model, p_true)

print(f"D_KL(p||q) = {D_pq:.4f} bits")
print(f"D_KL(q||p) = {D_qp:.4f} bits")
print(f"Symmetric? {np.isclose(D_pq, D_qp)}")

# Verify relationship
print(f"\nVerification: D_KL(p||q) = H(p,q) - H(p)")
print(f"{D_pq:.4f} = {H_pq:.4f} - {H_p:.4f} ✓")
```

---

### Exercise 4.3 Solution: Properties of KL Divergence

**Property 1: Non-negativity** D_KL(p||q) ≥ 0, with equality iff p = q

**Proof sketch:**
By Gibbs' inequality: -∑ p(x)·log q(x) ≥ -∑ p(x)·log p(x)
Therefore: H(p,q) ≥ H(p)
So: D_KL(p||q) = H(p,q) - H(p) ≥ 0

**Test 1: Different distributions**
```
p = [0.5, 0.5]
q = [0.7, 0.3]
D_KL(p||q) = 0.126 > 0 ✓
```

**Test 2: Identical distributions**
```
p = [0.5, 0.5]
q = [0.5, 0.5]
D_KL(p||q) = 0.5·log₂(1) + 0.5·log₂(1) = 0 ✓
```

**NumPy verification:**
```python
# Test 1: Different distributions
p1 = np.array([0.5, 0.5])
q1 = np.array([0.7, 0.3])
D1 = kl_divergence(p1, q1)
print(f"Different dists: D_KL = {D1:.6f} (should be > 0)")
assert D1 > 0, "KL divergence should be positive for different distributions"

# Test 2: Identical distributions
p2 = np.array([0.5, 0.5])
q2 = np.array([0.5, 0.5])
D2 = kl_divergence(p2, q2)
print(f"Identical dists: D_KL = {D2:.6f} (should be 0)")
assert np.isclose(D2, 0), "KL divergence should be 0 for identical distributions"

# Test 3: Many random distributions
for _ in range(10):
    p = np.random.dirichlet([1, 1, 1])
    q = np.random.dirichlet([1, 1, 1])
    D = kl_divergence(p, q)
    assert D >= -1e-10, f"KL divergence should be non-negative, got {D}"

print("\nAll non-negativity tests passed! ✓")
```

---

### Exercise 4.4 Solution: JS Divergence

**Jensen-Shannon Divergence:** JS(p||q) = ½D_KL(p||m) + ½D_KL(q||m), where m = ½(p + q)

Using p = [0.5, 0.5] and q = [0.7, 0.3]:

**Step 1: Calculate midpoint**
```
m = ½(p + q) = ½([0.5, 0.5] + [0.7, 0.3])
              = ½[1.2, 0.8]
              = [0.6, 0.4]
```

**Step 2: Calculate D_KL(p||m)**
```
D_KL(p||m) = 0.5·log₂(0.5/0.6) + 0.5·log₂(0.5/0.4)
           = 0.5·log₂(0.833) + 0.5·log₂(1.25)
           = 0.5·(-0.263) + 0.5·(0.322)
           = 0.030 bits
```

**Step 3: Calculate D_KL(q||m)**
```
D_KL(q||m) = 0.7·log₂(0.7/0.6) + 0.3·log₂(0.3/0.4)
           = 0.7·log₂(1.167) + 0.3·log₂(0.75)
           = 0.7·(0.222) + 0.3·(-0.415)
           = 0.030 bits
```

**Step 4: Calculate JS divergence**
```
JS(p||q) = ½·0.030 + ½·0.030
         = 0.030 bits
```

**Property: Symmetry**
```
JS(p||q) = JS(q||p) by construction

Unlike KL divergence, JS divergence is symmetric!
Also: 0 ≤ JS(p||q) ≤ 1 (bounded)
```

**NumPy implementation:**
```python
def js_divergence(p, q):
    """Calculate Jensen-Shannon divergence"""
    p = np.array(p)
    q = np.array(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

JS_pq = js_divergence(p_true, q_model)
JS_qp = js_divergence(q_model, p_true)

print(f"JS(p||q) = {JS_pq:.4f} bits")
print(f"JS(q||p) = {JS_qp:.4f} bits")
print(f"Symmetric? {np.isclose(JS_pq, JS_qp)}")

# Compare with KL divergence
print(f"\nD_KL(p||q) = {D_pq:.4f} bits (not symmetric)")
print(f"D_KL(q||p) = {D_qp:.4f} bits")
print(f"JS(p||q) = {JS_pq:.4f} bits (symmetric)")
```

---

## Part 5: Applications to ML

### Exercise 5.1 Solution: Cross-Entropy Loss

True labels: y = [1, 0, 1] (one-hot encoded)
Predictions: ŷ = [0.8, 0.1, 0.9]

**Binary cross-entropy formula:** L = -∑ᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

**Step 1: Calculate loss for each sample**
```
Sample 1: y=1, ŷ=0.8
L₁ = -[1·log(0.8) + 0·log(0.2)]
   = -log(0.8)
   = 0.223

Sample 2: y=0, ŷ=0.1
L₂ = -[0·log(0.1) + 1·log(0.9)]
   = -log(0.9)
   = 0.105

Sample 3: y=1, ŷ=0.9
L₃ = -[1·log(0.9) + 0·log(0.1)]
   = -log(0.9)
   = 0.105
```

**Step 2: Average loss**
```
L_avg = (L₁ + L₂ + L₃) / 3
      = (0.223 + 0.105 + 0.105) / 3
      = 0.144
```

**NumPy implementation:**
```python
def binary_cross_entropy(y_true, y_pred):
    """Calculate binary cross-entropy loss"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1])
y_pred = np.array([0.8, 0.1, 0.9])

loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary cross-entropy loss: {loss:.4f}")

# Individual losses
individual_losses = -(y_true * np.log(y_pred + 1e-10) +
                      (1 - y_true) * np.log(1 - y_pred + 1e-10))
print(f"Individual losses: {individual_losses}")
print(f"Mean: {np.mean(individual_losses):.4f}")
```

---

### Exercise 5.2 Solution: Categorical Cross-Entropy

True label: y = [0, 1, 0] (class 1)
Predictions: ŷ = [0.1, 0.7, 0.2]

**Categorical cross-entropy:** L = -∑ₖ yₖ·log(ŷₖ)

**Step 1: Calculate loss**
```
L = -[0·log(0.1) + 1·log(0.7) + 0·log(0.2)]
  = -log(0.7)
  = 0.357
```

**Interpretation:** The loss is the negative log probability of the correct class. Lower probability → higher loss.

**Compare different predictions:**
```
Prediction [0.1, 0.7, 0.2]: L = -log(0.7) = 0.357
Prediction [0.1, 0.9, 0.0]: L = -log(0.9) = 0.105 (better!)
Prediction [0.4, 0.4, 0.2]: L = -log(0.4) = 0.916 (worse!)
```

**NumPy implementation:**
```python
def categorical_cross_entropy(y_true, y_pred):
    """Calculate categorical cross-entropy loss"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1)
    return -np.sum(y_true * np.log(y_pred))

y_true = np.array([0, 1, 0])
y_pred = np.array([0.1, 0.7, 0.2])

loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical cross-entropy: {loss:.4f}")

# Test with different predictions
predictions = [
    [0.1, 0.7, 0.2],
    [0.1, 0.9, 0.0],
    [0.4, 0.4, 0.2],
    [0.0, 1.0, 0.0],  # Perfect prediction
]

for pred in predictions:
    L = categorical_cross_entropy(y_true, pred)
    print(f"Prediction {pred}: Loss = {L:.4f}")
```

---

### Exercise 5.3 Solution: Softmax and Cross-Entropy

Logits: z = [2.0, 1.0, 0.1]
True label: y = [0, 1, 0]

**Step 1: Apply softmax**
```
softmax(zᵢ) = exp(zᵢ) / ∑ⱼ exp(zⱼ)

exp(z) = [exp(2.0), exp(1.0), exp(0.1)]
       = [7.389, 2.718, 1.105]

∑ exp(z) = 7.389 + 2.718 + 1.105 = 11.212

ŷ = [7.389/11.212, 2.718/11.212, 1.105/11.212]
  = [0.659, 0.242, 0.099]
```

**Step 2: Calculate cross-entropy**
```
L = -[0·log(0.659) + 1·log(0.242) + 0·log(0.099)]
  = -log(0.242)
  = 1.419
```

**Step 3: Derivative for backprop**
```
∂L/∂zᵢ = ŷᵢ - yᵢ

∂L/∂z = [0.659-0, 0.242-1, 0.099-0]
       = [0.659, -0.758, 0.099]
```

**NumPy implementation:**
```python
def softmax(z):
    """Numerically stable softmax"""
    exp_z = np.exp(z - np.max(z))  # Subtract max for stability
    return exp_z / np.sum(exp_z)

def softmax_cross_entropy_loss(z, y_true):
    """Combined softmax + cross-entropy"""
    y_pred = softmax(z)
    loss = -np.sum(y_true * np.log(y_pred + 1e-10))
    # Gradient
    grad = y_pred - y_true
    return loss, grad, y_pred

z = np.array([2.0, 1.0, 0.1])
y_true = np.array([0, 1, 0])

loss, grad, y_pred = softmax_cross_entropy_loss(z, y_true)

print(f"Logits: {z}")
print(f"Softmax probabilities: {y_pred}")
print(f"Cross-entropy loss: {loss:.4f}")
print(f"Gradient ∂L/∂z: {grad}")
```

---

## Challenge Problems

### Challenge 1 Solution: Entropy of English Text

Estimate the entropy of English text by character frequencies

**Method:** Sample English text and calculate character-level entropy

**Step 1: Count character frequencies**
```python
sample_text = """
The quick brown fox jumps over the lazy dog.
Machine learning is transforming artificial intelligence.
Information theory provides the mathematical foundation.
""".lower()

# Count frequencies
from collections import Counter
char_counts = Counter(sample_text)
total_chars = sum(char_counts.values())

# Calculate probabilities
char_probs = {char: count/total_chars for char, count in char_counts.items()}

# Calculate entropy
H_english = entropy(list(char_probs.values()))
print(f"Character-level entropy of English: {H_english:.4f} bits/char")
```

**Step 2: Compare with uniform distribution**
```python
# If all 27 characters (26 letters + space) were equally likely
H_uniform = np.log2(27)
print(f"Uniform distribution entropy: {H_uniform:.4f} bits/char")
print(f"Compression ratio: {H_english / H_uniform:.2%}")
```

**Interpretation:**
- English has ~4.5 bits/char entropy (varies by analysis)
- Uniform would be ~4.75 bits/char (27 characters)
- Letters are NOT equally likely (e, t, a more common than z, q)
- This redundancy allows compression!

**Full implementation:**
```python
def estimate_text_entropy(text):
    """Estimate entropy of text"""
    text = text.lower()
    # Count character frequencies
    counts = Counter(text)
    total = sum(counts.values())
    probs = np.array([count/total for count in counts.values()])
    return entropy(probs)

# Test on longer text
with open('sample_english.txt', 'r') as f:
    long_text = f.read()

H_est = estimate_text_entropy(long_text)
print(f"Estimated entropy: {H_est:.4f} bits/char")

# Claude Shannon's estimate: ~1 bit/char for word-level English
# Character-level: ~4-5 bits/char
# This difference shows context reduces uncertainty!
```

---

### Challenge 2 Solution: Information Gain for Decision Trees

Dataset: 8 samples, 4 positive, 4 negative
Feature A splits: Left (3+, 1-), Right (1+, 3-)

**Step 1: Calculate initial entropy**
```
H(Y) = -[4/8·log₂(4/8) + 4/8·log₂(4/8)]
     = -[0.5·(-1) + 0.5·(-1)]
     = 1 bit
```

**Step 2: Calculate entropy after split**

Left node (4 samples: 3+, 1-):
```
H(Y|Left) = -[3/4·log₂(3/4) + 1/4·log₂(1/4)]
          = -[0.75·(-0.415) + 0.25·(-2)]
          = -[-0.311 - 0.5]
          = 0.811 bits
```

Right node (4 samples: 1+, 3-):
```
H(Y|Right) = -[1/4·log₂(1/4) + 3/4·log₂(3/4)]
           = 0.811 bits (symmetric)
```

**Step 3: Weighted average entropy after split**
```
H(Y|A) = P(Left)·H(Y|Left) + P(Right)·H(Y|Right)
       = 4/8·0.811 + 4/8·0.811
       = 0.811 bits
```

**Step 4: Information gain**
```
IG(Y, A) = H(Y) - H(Y|A)
         = 1.000 - 0.811
         = 0.189 bits
```

**Interpretation:** Feature A reduces uncertainty by 0.189 bits. This is the same as the mutual information I(Y;A).

**NumPy implementation:**
```python
def calculate_entropy_split(n_pos, n_neg):
    """Calculate entropy for a node"""
    total = n_pos + n_neg
    if total == 0:
        return 0
    p_pos = n_pos / total
    p_neg = n_neg / total
    return entropy([p_pos, p_neg])

def information_gain(n_pos_parent, n_neg_parent,
                    n_pos_left, n_neg_left,
                    n_pos_right, n_neg_right):
    """Calculate information gain for a split"""
    # Parent entropy
    H_parent = calculate_entropy_split(n_pos_parent, n_neg_parent)

    # Child entropies
    H_left = calculate_entropy_split(n_pos_left, n_neg_left)
    H_right = calculate_entropy_split(n_pos_right, n_neg_right)

    # Weighted average
    total = n_pos_parent + n_neg_parent
    n_left = n_pos_left + n_neg_left
    n_right = n_pos_right + n_neg_right

    H_children = (n_left/total * H_left +
                  n_right/total * H_right)

    # Information gain
    return H_parent - H_children

# Example from problem
IG = information_gain(
    n_pos_parent=4, n_neg_parent=4,
    n_pos_left=3, n_neg_left=1,
    n_pos_right=1, n_neg_right=3
)

print(f"Information Gain: {IG:.4f} bits")

# Compare with different splits
print("\nComparing different possible splits:")

# Perfect split
IG_perfect = information_gain(4, 4, 4, 0, 0, 4)
print(f"Perfect split (4+,0- | 0+,4-): IG = {IG_perfect:.4f} bits")

# No information
IG_none = information_gain(4, 4, 2, 2, 2, 2)
print(f"No information (2+,2- | 2+,2-): IG = {IG_none:.4f} bits")

# Our split
print(f"Our split (3+,1- | 1+,3-): IG = {IG:.4f} bits")
```

---

## Summary

**Key Concepts:**

1. **Entropy H(X):** Measure of uncertainty/information content
   - H(X) = -∑ p(x)·log₂ p(x)
   - Maximum when distribution is uniform
   - Measured in bits

2. **Joint and Conditional Entropy:**
   - H(X,Y): Uncertainty about both variables
   - H(Y|X): Uncertainty about Y given X
   - Chain rule: H(X,Y) = H(X) + H(Y|X)

3. **Mutual Information I(X;Y):**
   - Shared information between variables
   - I(X;Y) = H(X) + H(Y) - H(X,Y)
   - I(X;Y) = 0 for independent variables
   - I(X;Y) = H(X) for perfectly correlated variables

4. **Cross-Entropy and KL Divergence:**
   - H(p,q): Expected bits using wrong code
   - D_KL(p||q): Extra bits due to model mismatch
   - D_KL(p||q) = H(p,q) - H(p)
   - KL divergence is non-negative and asymmetric

5. **Applications to ML:**
   - Cross-entropy loss for classification
   - Information gain for decision trees
   - KL divergence for model comparison
   - Entropy for regularization

**Practical Tips:**
- Use cross-entropy loss for classification problems
- Higher entropy = more uncertainty = harder to predict
- Mutual information measures feature relevance
- KL divergence measures distribution mismatch
- Information theory provides theoretical limits for compression and communication
