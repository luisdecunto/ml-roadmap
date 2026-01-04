# Information Theory Solutions - Module 5

Comprehensive solutions with step-by-step mathematical work and code implementations.

---

## Part 1: Entropy Basics

### Exercise 1.1 Solution: Binary Entropy

A biased coin has P(H) = 0.7, P(T) = 0.3

**1. Calculate entropy H(X) = -Σ p(x) log₂ p(x)**

```
H(X) = -[P(H)·log₂ P(H) + P(T)·log₂ P(T)]
     = -[0.7·log₂(0.7) + 0.3·log₂(0.3)]
     = -[0.7·(-0.515) + 0.3·(-1.737)]
     = -[-0.3605 - 0.5211]
     = 0.8816 bits
```

**2. Units: bits** (because we use log₂)

**3. Compare with fair coin (P(H) = 0.5)**

Fair coin entropy:
```
H(fair) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)]
        = -[0.5·(-1) + 0.5·(-1)]
        = 1.0 bit
```

**4. Which has more entropy?**

The **fair coin** (1.0 bit) has more entropy than the biased coin (0.8816 bits).

**Why?** Entropy measures uncertainty/surprise. A fair coin is maximally uncertain (50-50 chance). The biased coin is more predictable (70% likely heads), so lower entropy.

**NumPy verification:**
```python
import numpy as np

def entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # Filter zeros
    return -np.sum(probs * np.log2(probs))

# Biased coin
p_biased = [0.7, 0.3]
H_biased = entropy(p_biased)
print(f"Biased coin: {H_biased:.4f} bits")  # 0.8816

# Fair coin
p_fair = [0.5, 0.5]
H_fair = entropy(p_fair)
print(f"Fair coin: {H_fair:.4f} bits")  # 1.0000
```

---

### Exercise 1.2 Solution: Discrete Distribution Entropy

Distribution: P(X=1) = 0.5, P(X=2) = 0.25, P(X=3) = 0.125, P(X=4) = 0.125

**1. Calculate H(X)**

```
H(X) = -Σ p(xᵢ) log₂ p(xᵢ)
     = -[0.5·log₂(0.5) + 0.25·log₂(0.25) + 0.125·log₂(0.125) + 0.125·log₂(0.125)]
     = -[0.5·(-1) + 0.25·(-2) + 0.125·(-3) + 0.125·(-3)]
     = -[-0.5 - 0.5 - 0.375 - 0.375]
     = 1.75 bits
```

**2. Maximum possible entropy for 4 outcomes**

Maximum entropy occurs with uniform distribution:
```
H_max = log₂(4) = 2 bits
```

**3. How close to maximum?**

```
Ratio = H(X) / H_max = 1.75 / 2 = 0.875 = 87.5%
```

This distribution is 87.5% as entropic as the maximum. It's fairly high entropy but not uniform (outcome 1 is more likely).

**NumPy verification:**
```python
p = np.array([0.5, 0.25, 0.125, 0.125])
H = -np.sum(p * np.log2(p))
H_max = np.log2(len(p))
print(f"H(X) = {H:.4f} bits")  # 1.7500
print(f"H_max = {H_max:.4f} bits")  # 2.0000
print(f"Ratio = {H/H_max:.2%}")  # 87.50%
```

---

### Exercise 1.3 Solution: Uniform vs Non-uniform

A: P(a) = 0.25 for all 4 outcomes (uniform)
B: P(b) = [0.7, 0.1, 0.1, 0.1] (non-uniform)

**1. Calculate H(A)**

```
H(A) = -4 × (0.25 × log₂(0.25))
     = -4 × (0.25 × -2)
     = -4 × (-0.5)
     = 2 bits
```

**2. Calculate H(B)**

```
H(B) = -[0.7·log₂(0.7) + 3×(0.1·log₂(0.1))]
     = -[0.7·(-0.515) + 0.3·(-3.322)]
     = -[-0.3605 - 0.9966]
     = 1.3571 bits
```

**3. Which is more "surprising"?**

**Distribution A** (2 bits) is more surprising. All outcomes are equally likely, so maximum uncertainty.

**4. Entropy and predictability relationship**

**High entropy = Low predictability = High surprise**
- Uniform distribution A: can't predict outcome, high surprise
- Skewed distribution B: outcome 1 is 70% likely, more predictable

**General principle:** Entropy quantifies average surprise/information content. Predictable systems have low entropy, unpredictable systems have high entropy.

**NumPy verification:**
```python
A = np.array([0.25, 0.25, 0.25, 0.25])
B = np.array([0.7, 0.1, 0.1, 0.1])

H_A = -np.sum(A * np.log2(A))
H_B = -np.sum(B * np.log2(B))

print(f"H(A) = {H_A:.4f} bits")  # 2.0000
print(f"H(B) = {H_B:.4f} bits")  # 1.3571
```

---

## Part 2: Joint and Conditional Entropy

### Exercise 2.1 Solution: Joint Entropy

Joint distribution table:

| X\Y | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.3 | 0.2 |
| X=1 | 0.1 | 0.4 |

**1. Calculate H(X, Y)**

```
H(X,Y) = -ΣΣ p(x,y) log₂ p(x,y)
       = -[0.3·log₂(0.3) + 0.2·log₂(0.2) + 0.1·log₂(0.1) + 0.4·log₂(0.4)]
       = -[0.3·(-1.737) + 0.2·(-2.322) + 0.1·(-3.322) + 0.4·(-1.322)]
       = -[-0.5211 - 0.4644 - 0.3322 - 0.5288]
       = 1.8465 bits
```

**2. Calculate marginal H(X)**

Marginal P(X):
- P(X=0) = 0.3 + 0.2 = 0.5
- P(X=1) = 0.1 + 0.4 = 0.5

```
H(X) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)]
     = 1 bit
```

**3. Calculate marginal H(Y)**

Marginal P(Y):
- P(Y=0) = 0.3 + 0.1 = 0.4
- P(Y=1) = 0.2 + 0.4 = 0.6

```
H(Y) = -[0.4·log₂(0.4) + 0.6·log₂(0.6)]
     = -[0.4·(-1.322) + 0.6·(-0.737)]
     = -[-0.5288 - 0.4422]
     = 0.9710 bits
```

**4. Verify: H(X, Y) ≤ H(X) + H(Y)**

```
H(X,Y) = 1.8465 bits
H(X) + H(Y) = 1 + 0.9710 = 1.9710 bits

1.8465 ≤ 1.9710 ✓
```

**Why?** Joint entropy is maximized when variables are independent. If dependent (as here), H(X,Y) < H(X) + H(Y).

**NumPy verification:**
```python
joint = np.array([[0.3, 0.2],
                   [0.1, 0.4]])

H_joint = -np.sum(joint * np.log2(joint))
p_x = joint.sum(axis=1)
p_y = joint.sum(axis=0)
H_x = -np.sum(p_x * np.log2(p_x))
H_y = -np.sum(p_y * np.log2(p_y))

print(f"H(X,Y) = {H_joint:.4f} bits")  # 1.8465
print(f"H(X) = {H_x:.4f} bits")  # 1.0000
print(f"H(Y) = {H_y:.4f} bits")  # 0.9710
print(f"H(X) + H(Y) = {H_x + H_y:.4f} bits")  # 1.9710
```

---

### Exercise 2.2 Solution: Conditional Entropy

Using same joint distribution from 2.1.

**1. Calculate P(Y|X=0) and P(Y|X=1)**

P(Y|X=0):
```
P(Y=0|X=0) = P(X=0,Y=0) / P(X=0) = 0.3 / 0.5 = 0.6
P(Y=1|X=0) = P(X=0,Y=1) / P(X=0) = 0.2 / 0.5 = 0.4
```

P(Y|X=1):
```
P(Y=0|X=1) = P(X=1,Y=0) / P(X=1) = 0.1 / 0.5 = 0.2
P(Y=1|X=1) = P(X=1,Y=1) / P(X=1) = 0.4 / 0.5 = 0.8
```

**2. Calculate H(Y|X=0) and H(Y|X=1)**

```
H(Y|X=0) = -[0.6·log₂(0.6) + 0.4·log₂(0.4)]
         = -[0.6·(-0.737) + 0.4·(-1.322)]
         = 0.9710 bits

H(Y|X=1) = -[0.2·log₂(0.2) + 0.8·log₂(0.8)]
         = -[0.2·(-2.322) + 0.8·(-0.322)]
         = 0.7219 bits
```

**3. Calculate H(Y|X)**

```
H(Y|X) = Σ P(x) H(Y|X=x)
       = P(X=0)·H(Y|X=0) + P(X=1)·H(Y|X=1)
       = 0.5·(0.9710) + 0.5·(0.7219)
       = 0.8465 bits
```

**4. Verify: H(X, Y) = H(X) + H(Y|X)**

```
H(X) + H(Y|X) = 1.0 + 0.8465 = 1.8465 bits
H(X,Y) = 1.8465 bits ✓
```

This is the **chain rule for entropy**!

**NumPy verification:**
```python
# Conditional distributions
P_Y_given_X0 = joint[0, :] / p_x[0]
P_Y_given_X1 = joint[1, :] / p_x[1]

H_Y_given_X0 = -np.sum(P_Y_given_X0 * np.log2(P_Y_given_X0))
H_Y_given_X1 = -np.sum(P_Y_given_X1 * np.log2(P_Y_given_X1))

H_Y_given_X = p_x[0] * H_Y_given_X0 + p_x[1] * H_Y_given_X1

print(f"H(Y|X=0) = {H_Y_given_X0:.4f}")  # 0.9710
print(f"H(Y|X=1) = {H_Y_given_X1:.4f}")  # 0.7219
print(f"H(Y|X) = {H_Y_given_X:.4f}")  # 0.8465
print(f"H(X) + H(Y|X) = {H_x + H_Y_given_X:.4f}")  # 1.8465
```

---

### Exercise 2.3 Solution: Chain Rule

**1. Chain rule for 3 variables**

```
H(X,Y,Z) = H(X) + H(Y|X) + H(Z|X,Y)
```

**Intuition:** The joint entropy equals the entropy of the first variable, plus the additional entropy from the second given the first, plus the additional entropy from the third given both previous.

**2. Given values:**
- H(X) = 2 bits
- H(Y|X) = 1.5 bits
- H(Z|X,Y) = 1 bit

**3. Calculate H(X, Y, Z)**

```
H(X,Y,Z) = H(X) + H(Y|X) + H(Z|X,Y)
         = 2 + 1.5 + 1
         = 4.5 bits
```

**4. Why useful for sequential data?**

The chain rule is crucial for modeling sequences (text, speech, time series):
- Language models predict next word given context
- In "The cat sat on the ___", we predict given previous words
- Each word adds information conditioned on what came before
- H(sentence) = H(w₁) + H(w₂|w₁) + H(w₃|w₁,w₂) + ...

This decomposition helps us:
- Understand information flow in sequences
- Build autoregressive models (GPT, etc.)
- Measure compression efficiency
- Calculate perplexity (exp of entropy per symbol)

---

## Part 3: Mutual Information

### Exercise 3.1 Solution: Computing Mutual Information

Using joint distribution from Exercise 2.1:

**1. Calculate I(X; Y) = H(X) + H(Y) - H(X, Y)**

From Exercise 2.1:
- H(X) = 1.0 bit
- H(Y) = 0.9710 bits
- H(X, Y) = 1.8465 bits

```
I(X; Y) = 1.0 + 0.9710 - 1.8465
        = 0.1245 bits
```

**2. Alternative: I(X; Y) = H(X) - H(X|Y)**

First calculate H(X|Y):
```
H(X|Y) = H(X,Y) - H(Y)
       = 1.8465 - 0.9710
       = 0.8755 bits
```

Then:
```
I(X; Y) = H(X) - H(X|Y)
        = 1.0 - 0.8755
        = 0.1245 bits ✓
```

**3. Both give same result** ✓

**4. Are X and Y independent?**

**No.** If independent, I(X; Y) = 0. Here I(X; Y) = 0.1245 bits > 0, so X and Y share information.

**Interpretation:** Knowing X reduces uncertainty about Y by 0.1245 bits (and vice versa).

**NumPy verification:**
```python
I_xy = H_x + H_y - H_joint
print(f"I(X;Y) = {I_xy:.4f} bits")  # 0.1245

# Alternative
H_x_given_y = H_joint - H_y
I_xy_alt = H_x - H_x_given_y
print(f"I(X;Y) alternative = {I_xy_alt:.4f} bits")  # 0.1245
```

---

### Exercise 3.2 Solution: Independence Test

Joint distribution:

| X\Y | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.3 | 0.3 |
| X=1 | 0.2 | 0.2 |

**1. Calculate I(X; Y)**

Marginals:
- P(X=0) = 0.6, P(X=1) = 0.4
- P(Y=0) = 0.5, P(Y=1) = 0.5

```
H(X) = -[0.6·log₂(0.6) + 0.4·log₂(0.4)]
     = 0.9710 bits

H(Y) = -[0.5·log₂(0.5) + 0.5·log₂(0.5)]
     = 1.0 bit

H(X,Y) = -[0.3·log₂(0.3) + 0.3·log₂(0.3) + 0.2·log₂(0.2) + 0.2·log₂(0.2)]
       = -[2×(0.3·(-1.737)) + 2×(0.2·(-2.322))]
       = 1.9710 bits

I(X;Y) = 0.9710 + 1.0 - 1.9710 = 0 bits
```

**2. Are X and Y independent?**

**Yes!** I(X; Y) = 0 means X and Y share no information.

**3. Verify: P(X,Y) = P(X)P(Y) for all x,y**

```
P(X=0,Y=0) = 0.3,  P(X=0)·P(Y=0) = 0.6·0.5 = 0.3 ✓
P(X=0,Y=1) = 0.3,  P(X=0)·P(Y=1) = 0.6·0.5 = 0.3 ✓
P(X=1,Y=0) = 0.2,  P(X=1)·P(Y=0) = 0.4·0.5 = 0.2 ✓
P(X=1,Y=1) = 0.2,  P(X=1)·P(Y=1) = 0.4·0.5 = 0.2 ✓
```

All equal! **Independence confirmed.**

**NumPy verification:**
```python
joint_indep = np.array([[0.3, 0.3],
                         [0.2, 0.2]])

H_joint_indep = -np.sum(joint_indep * np.log2(joint_indep))
p_x_indep = joint_indep.sum(axis=1)
p_y_indep = joint_indep.sum(axis=0)
H_x_indep = -np.sum(p_x_indep * np.log2(p_x_indep))
H_y_indep = -np.sum(p_y_indep * np.log2(p_y_indep))

I_indep = H_x_indep + H_y_indep - H_joint_indep
print(f"I(X;Y) = {I_indep:.10f} bits")  # ~0.0000000000
```

---

### Exercise 3.3 Solution: Mutual Information Properties

Given: I(X; Y) = 0.5 bits, H(X) = 2 bits, H(Y) = 1.5 bits

**1. Calculate H(X|Y)**

From I(X; Y) = H(X) - H(X|Y):
```
H(X|Y) = H(X) - I(X; Y)
       = 2 - 0.5
       = 1.5 bits
```

**2. Calculate H(Y|X)**

From I(X; Y) = H(Y) - H(Y|X):
```
H(Y|X) = H(Y) - I(X; Y)
       = 1.5 - 0.5
       = 1.0 bit
```

**3. Calculate H(X, Y)**

From I(X; Y) = H(X) + H(Y) - H(X, Y):
```
H(X,Y) = H(X) + H(Y) - I(X; Y)
       = 2 + 1.5 - 0.5
       = 3.0 bits
```

**4. Entropy Venn Diagram**

```
         ┌─────────────────┐
         │                 │
    ┌────┼────┐            │
    │    │    │            │
    │ 1.5│0.5 │    1.0     │
    │    │    │            │
    └────┼────┘            │
         │                 │
         └─────────────────┘

    H(X|Y)  I(X;Y)   H(Y|X)

    Total H(X) = 1.5 + 0.5 = 2.0
    Total H(Y) = 0.5 + 1.0 = 1.5
    Total H(X,Y) = 1.5 + 0.5 + 1.0 = 3.0
```

**Key relationships:**
- I(X; Y) is the overlap (shared information)
- H(X|Y) is X's unique information (given Y)
- H(Y|X) is Y's unique information (given X)
- H(X, Y) = H(X|Y) + I(X; Y) + H(Y|X)

---

## Part 4: Cross-Entropy and KL Divergence

### Exercise 4.1 Solution: Cross-Entropy

True distribution P: [0.5, 0.3, 0.2]
Model distribution Q: [0.4, 0.4, 0.2]

**1. Calculate H(P, Q) = -Σ p(x) log q(x)**

```
H(P,Q) = -[0.5·log₂(0.4) + 0.3·log₂(0.4) + 0.2·log₂(0.2)]
       = -[0.5·(-1.322) + 0.3·(-1.322) + 0.2·(-2.322)]
       = -[-0.6610 - 0.3966 - 0.4644]
       = 1.5220 bits
```

**2. Calculate H(P)**

```
H(P) = -[0.5·log₂(0.5) + 0.3·log₂(0.3) + 0.2·log₂(0.2)]
     = -[0.5·(-1) + 0.3·(-1.737) + 0.2·(-2.322)]
     = 1.4855 bits
```

**3. Which is larger?**

**H(P, Q) = 1.5220 > H(P) = 1.4855**

Cross-entropy is always ≥ entropy of true distribution.

**Why?** H(P, Q) ≥ H(P) with equality only when P = Q. The difference is the KL divergence.

**4. What does this tell us?**

Using Q to encode data from P requires **more bits** (1.5220) than the optimal code for P (1.4855).

The extra cost is:
```
D_KL(P||Q) = H(P,Q) - H(P) = 1.5220 - 1.4855 = 0.0365 bits
```

This is the **penalty** for using the wrong distribution.

**NumPy verification:**
```python
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.4, 0.4, 0.2])

H_p = -np.sum(P * np.log2(P))
H_pq = -np.sum(P * np.log2(Q))

print(f"H(P) = {H_p:.4f} bits")  # 1.4855
print(f"H(P,Q) = {H_pq:.4f} bits")  # 1.5220
print(f"Difference = {H_pq - H_p:.4f} bits")  # 0.0365
```

---

### Exercise 4.2 Solution: KL Divergence

Same distributions: P: [0.5, 0.3, 0.2], Q: [0.4, 0.4, 0.2]

**1. Calculate D_KL(P || Q) = Σ p(x) log(p(x)/q(x))**

```
D_KL(P||Q) = 0.5·log₂(0.5/0.4) + 0.3·log₂(0.3/0.4) + 0.2·log₂(0.2/0.2)
           = 0.5·log₂(1.25) + 0.3·log₂(0.75) + 0.2·log₂(1.0)
           = 0.5·(0.3219) + 0.3·(-0.4150) + 0.2·(0)
           = 0.1610 - 0.1245 + 0
           = 0.0365 bits
```

**2. Verify: D_KL(P || Q) = H(P, Q) - H(P)**

```
H(P,Q) - H(P) = 1.5220 - 1.4855 = 0.0365 bits ✓
```

This confirms the relationship!

**3. Calculate D_KL(Q || P)**

```
D_KL(Q||P) = 0.4·log₂(0.4/0.5) + 0.4·log₂(0.4/0.3) + 0.2·log₂(0.2/0.2)
           = 0.4·log₂(0.8) + 0.4·log₂(1.333) + 0
           = 0.4·(-0.3219) + 0.4·(0.4150)
           = -0.1288 + 0.1660
           = 0.0372 bits
```

**4. Is KL divergence symmetric?**

**No!**
```
D_KL(P||Q) = 0.0365 bits
D_KL(Q||P) = 0.0372 bits

D_KL(P||Q) ≠ D_KL(Q||P)
```

**Interpretation:**
- D_KL(P||Q): cost of using Q to approximate P
- D_KL(Q||P): cost of using P to approximate Q
- Different perspectives, different costs!

**NumPy verification:**
```python
D_kl_PQ = np.sum(P * np.log2(P / Q))
D_kl_QP = np.sum(Q * np.log2(Q / P))

print(f"D_KL(P||Q) = {D_kl_PQ:.4f} bits")  # 0.0365
print(f"D_KL(Q||P) = {D_kl_QP:.4f} bits")  # 0.0372
print(f"Are they equal? {np.isclose(D_kl_PQ, D_kl_QP)}")  # False
```

---

### Exercise 4.3 Solution: KL Divergence Properties

P = [0.8, 0.2], Q = [0.6, 0.4], R = [0.5, 0.5]

**1. Calculate D_KL(P || Q)**

```
D_KL(P||Q) = 0.8·log₂(0.8/0.6) + 0.2·log₂(0.2/0.4)
           = 0.8·log₂(1.333) + 0.2·log₂(0.5)
           = 0.8·(0.4150) + 0.2·(-1.0)
           = 0.3320 - 0.2
           = 0.1320 bits
```

**2. Calculate D_KL(P || R)**

```
D_KL(P||R) = 0.8·log₂(0.8/0.5) + 0.2·log₂(0.2/0.5)
           = 0.8·log₂(1.6) + 0.2·log₂(0.4)
           = 0.8·(0.6781) + 0.2·(-1.3219)
           = 0.5425 - 0.2644
           = 0.2781 bits
```

**3. Calculate D_KL(Q || R)**

```
D_KL(Q||R) = 0.6·log₂(0.6/0.5) + 0.4·log₂(0.4/0.5)
           = 0.6·log₂(1.2) + 0.4·log₂(0.8)
           = 0.6·(0.2630) + 0.4·(-0.3219)
           = 0.1578 - 0.1288
           = 0.0290 bits
```

**4. Which model is "closer" to P?**

**Model Q** is closer to P:
```
D_KL(P||Q) = 0.1320 bits < D_KL(P||R) = 0.2781 bits
```

**Interpretation:**
- Q = [0.6, 0.4] is closer to P = [0.8, 0.2]
- R = [0.5, 0.5] (uniform) is farther from P
- Lower KL divergence = better approximation

**NumPy verification:**
```python
P = np.array([0.8, 0.2])
Q = np.array([0.6, 0.4])
R = np.array([0.5, 0.5])

D_PQ = np.sum(P * np.log2(P / Q))
D_PR = np.sum(P * np.log2(P / R))
D_QR = np.sum(Q * np.log2(Q / R))

print(f"D_KL(P||Q) = {D_PQ:.4f} bits")  # 0.1320
print(f"D_KL(P||R) = {D_PR:.4f} bits")  # 0.2781
print(f"D_KL(Q||R) = {D_QR:.4f} bits")  # 0.0290
print(f"Closest to P: {'Q' if D_PQ < D_PR else 'R'}")  # Q
```

---

### Exercise 4.4 Solution: Cross-Entropy in Classification

Binary classification:
- True labels: [1, 0, 1, 1]
- Predictions: [0.9, 0.2, 0.8, 0.7]

**1. Cross-entropy loss formula**

```
L = -(1/n)Σ[yᵢ log(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```

**2. Calculate for each sample**

Sample 1: y=1, ŷ=0.9
```
L₁ = -[1·log(0.9) + 0·log(0.1)]
   = -log(0.9)
   = 0.1054
```

Sample 2: y=0, ŷ=0.2
```
L₂ = -[0·log(0.2) + 1·log(0.8)]
   = -log(0.8)
   = 0.2231
```

Sample 3: y=1, ŷ=0.8
```
L₃ = -log(0.8)
   = 0.2231
```

Sample 4: y=1, ŷ=0.7
```
L₄ = -log(0.7)
   = 0.3567
```

**3. Take average**

```
L = (0.1054 + 0.2231 + 0.2231 + 0.3567) / 4
  = 0.9083 / 4
  = 0.2271
```

**4. What happens if prediction is wrong but confident?**

Example: y=1, ŷ=0.1 (confident but wrong)
```
L = -log(0.1) = 2.3026 (very high!)
```

Example: y=1, ŷ=0.9 (confident and right)
```
L = -log(0.9) = 0.1054 (low)
```

**Key insight:** Cross-entropy **heavily penalizes** confident wrong predictions. The loss grows without bound as confidence in wrong answer increases.

**NumPy verification:**
```python
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.2, 0.8, 0.7])

# Per-sample losses
losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
print(f"Per-sample losses: {losses}")
# [0.1054, 0.2231, 0.2231, 0.3567]

# Average
bce = np.mean(losses)
print(f"Binary Cross-Entropy = {bce:.4f}")  # 0.2271

# Confident wrong prediction
y_wrong = np.array([1])
y_pred_wrong = np.array([0.1])
loss_wrong = -np.log(y_pred_wrong[0])
print(f"Loss for confident wrong = {loss_wrong:.4f}")  # 2.3026
```

---

## Part 5: Applications to ML

### Exercise 5.1 Solution: Optimal Code Length

Message probabilities: P(A)=0.5, P(B)=0.25, P(C)=0.125, P(D)=0.125

**1. Calculate entropy H (minimum average bits)**

```
H = -[0.5·log₂(0.5) + 0.25·log₂(0.25) + 0.125·log₂(0.125) + 0.125·log₂(0.125)]
  = -[0.5·(-1) + 0.25·(-2) + 0.125·(-3) + 0.125·(-3)]
  = 1.75 bits
```

**2. Design Huffman code**

Build tree by combining lowest probabilities:

```
Step 1: Combine C(0.125) + D(0.125) = CD(0.25)
Step 2: Combine B(0.25) + CD(0.25) = BCD(0.5)
Step 3: Combine A(0.5) + BCD(0.5) = Root(1.0)

Tree:
         Root
        /    \
       A      BCD
       |      / \
       |     B  CD
       |     |  / \
       |     |  C  D

Codes (left=0, right=1):
A:  0      (1 bit)
B:  10     (2 bits)
C:  110    (3 bits)
D:  111    (3 bits)
```

**3. Calculate average code length**

```
Avg = 0.5·(1) + 0.25·(2) + 0.125·(3) + 0.125·(3)
    = 0.5 + 0.5 + 0.375 + 0.375
    = 1.75 bits
```

**4. How close to entropy?**

```
Entropy = 1.75 bits
Huffman avg = 1.75 bits

Perfect match! ✓
```

**Why perfect?** This distribution has powers-of-2 probabilities, so Huffman achieves Shannon's entropy bound exactly.

**Python verification:**
```python
# Calculate entropy
probs = np.array([0.5, 0.25, 0.125, 0.125])
H = -np.sum(probs * np.log2(probs))

# Huffman code lengths
code_lengths = np.array([1, 2, 3, 3])
avg_length = np.sum(probs * code_lengths)

print(f"Entropy H = {H:.4f} bits")  # 1.7500
print(f"Huffman avg = {avg_length:.4f} bits")  # 1.7500
print(f"Efficiency = {H/avg_length:.2%}")  # 100.00%
```

---

### Exercise 5.2 Solution: Decision Tree Splitting

Dataset: [6Y, 4N] total

Split A:
- Left: [4Y, 1N] (5 samples)
- Right: [2Y, 3N] (5 samples)

**1. Entropy of parent node**

```
P(Y) = 6/10 = 0.6
P(N) = 4/10 = 0.4

H(parent) = -[0.6·log₂(0.6) + 0.4·log₂(0.4)]
          = -[0.6·(-0.737) + 0.4·(-1.322)]
          = 0.9710 bits
```

**2. Entropy of left child [4Y, 1N]**

```
P(Y) = 4/5 = 0.8
P(N) = 1/5 = 0.2

H(left) = -[0.8·log₂(0.8) + 0.2·log₂(0.2)]
        = -[0.8·(-0.322) + 0.2·(-2.322)]
        = 0.7219 bits
```

**3. Entropy of right child [2Y, 3N]**

```
P(Y) = 2/5 = 0.4
P(N) = 3/5 = 0.6

H(right) = -[0.4·log₂(0.4) + 0.6·log₂(0.6)]
         = -[0.4·(-1.322) + 0.6·(-0.737)]
         = 0.9710 bits
```

**4. Calculate information gain**

```
IG = H(parent) - Σ(nᵢ/n)·H(childᵢ)
   = H(parent) - [(5/10)·H(left) + (5/10)·H(right)]
   = 0.9710 - [0.5·(0.7219) + 0.5·(0.9710)]
   = 0.9710 - [0.3610 + 0.4855]
   = 0.9710 - 0.8465
   = 0.1245 bits
```

**5. Is this a good split?**

**Yes, but moderate.** We gain 0.1245 bits of information (reduce uncertainty by 12.8%).

Good split characteristics:
- IG > 0 ✓ (we learn something)
- Left child is purer (0.7219 < 0.9710) ✓
- Right child is same entropy (0.9710 = 0.9710)

**Could be better:** Right child didn't improve. Ideal split would maximize purity in both children.

**NumPy verification:**
```python
def calc_entropy(counts):
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))

parent = np.array([6, 4])
left = np.array([4, 1])
right = np.array([2, 3])

H_parent = calc_entropy(parent)
H_left = calc_entropy(left)
H_right = calc_entropy(right)

n = parent.sum()
IG = H_parent - (left.sum()/n * H_left + right.sum()/n * H_right)

print(f"H(parent) = {H_parent:.4f} bits")  # 0.9710
print(f"H(left) = {H_left:.4f} bits")  # 0.7219
print(f"H(right) = {H_right:.4f} bits")  # 0.9710
print(f"Information Gain = {IG:.4f} bits")  # 0.1245
print(f"Reduction = {IG/H_parent:.2%}")  # 12.82%
```

---

### Exercise 5.3 Solution: Softmax and Cross-Entropy

Logits: z = [2.0, 1.0, 0.1]
True label: class 0 (one-hot: [1, 0, 0])

**1. Calculate softmax: p(i) = exp(zᵢ) / Σ exp(zⱼ)**

```
exp(z) = [exp(2.0), exp(1.0), exp(0.1)]
       = [7.389, 2.718, 1.105]

Sum = 7.389 + 2.718 + 1.105 = 11.212

p(0) = 7.389 / 11.212 = 0.6590
p(1) = 2.718 / 11.212 = 0.2424
p(2) = 1.105 / 11.212 = 0.0986
```

Check: 0.6590 + 0.2424 + 0.0986 = 1.0000 ✓

**2. Calculate cross-entropy: -Σ yᵢ log(p(i))**

True label: y = [1, 0, 0]

```
CE = -[1·log(0.6590) + 0·log(0.2424) + 0·log(0.0986)]
   = -log(0.6590)
   = 0.4170
```

**Simplification:** For one-hot encoded labels, only the true class matters:
```
CE = -log(p(true_class))
```

**3. What if true label was class 2?**

```
CE = -log(p(2))
   = -log(0.0986)
   = 2.3165
```

Much higher loss! Model predicted 6.59% for class 2 but it was correct.

**4. Why use log in the loss?**

Key reasons:
1. **Numerical stability:** Softmax outputs multiply (p₁·p₂·...), log converts to sum
2. **Gradient simplicity:** ∂CE/∂zᵢ = p(i) - yᵢ (beautiful clean gradient!)
3. **Information theory:** log relates to bits/nats of surprise
4. **Penalizes confidence:** Wrong predictions near 0 get huge loss (→ ∞)
5. **Maximum likelihood:** Log-likelihood is the natural objective

**NumPy verification:**
```python
logits = np.array([2.0, 1.0, 0.1])

# Softmax
probs = np.exp(logits) / np.sum(np.exp(logits))
print(f"Softmax probs: {probs}")
# [0.6590, 0.2424, 0.0986]

# Cross-entropy for class 0
true_label = np.array([1, 0, 0])
ce_loss = -np.sum(true_label * np.log(probs))
print(f"CE loss (class 0): {ce_loss:.4f}")  # 0.4170

# Cross-entropy for class 2
true_label_2 = np.array([0, 0, 1])
ce_loss_2 = -np.sum(true_label_2 * np.log(probs))
print(f"CE loss (class 2): {ce_loss_2:.4f}")  # 2.3165

# Simplified calculation
ce_simple = -np.log(probs[0])
print(f"CE simplified: {ce_simple:.4f}")  # 0.4170
```

---

## Challenge Problems

### Challenge 1 Solution: Entropy Rate of Markov Chain

Transition matrix:
```
     [0.7  0.3]
P =  [0.4  0.6]
```

**1. Find stationary distribution π (solve πP = π)**

Setup: π = [π₀, π₁] such that:
```
π₀·0.7 + π₁·0.4 = π₀
π₀·0.3 + π₁·0.6 = π₁
π₀ + π₁ = 1
```

From first equation:
```
π₀·0.7 + π₁·0.4 = π₀
π₁·0.4 = π₀·0.3
π₁ = (0.3/0.4)·π₀ = 0.75·π₀
```

Substitute into normalization:
```
π₀ + 0.75·π₀ = 1
1.75·π₀ = 1
π₀ = 1/1.75 = 0.5714

π₁ = 0.75·0.5714 = 0.4286
```

**Stationary distribution: π = [0.5714, 0.4286]**

Check:
```
π·P = [0.5714·0.7 + 0.4286·0.4, 0.5714·0.3 + 0.4286·0.6]
    = [0.4000 + 0.1714, 0.1714 + 0.2572]
    = [0.5714, 0.4286] = π ✓
```

**2. Calculate entropy rate: H = -ΣᵢΣⱼ πᵢ pᵢⱼ log pᵢⱼ**

```
H = -[π₀·(p₀₀·log₂ p₀₀ + p₀₁·log₂ p₀₁) + π₁·(p₁₀·log₂ p₁₀ + p₁₁·log₂ p₁₁)]
  = -[0.5714·(0.7·log₂ 0.7 + 0.3·log₂ 0.3) + 0.4286·(0.4·log₂ 0.4 + 0.6·log₂ 0.6)]
  = -[0.5714·(0.7·(-0.515) + 0.3·(-1.737)) + 0.4286·(0.4·(-1.322) + 0.6·(-0.737))]
  = -[0.5714·(-0.8816) + 0.4286·(-0.9710)]
  = -[-0.5038 - 0.4162]
  = 0.9200 bits/step
```

**3. Compare with entropy of stationary distribution**

```
H(π) = -[0.5714·log₂ 0.5714 + 0.4286·log₂ 0.4286]
     = -[0.5714·(-0.807) + 0.4286·(-1.222)]
     = 0.9850 bits
```

**Comparison:**
- Entropy rate: 0.9200 bits/step
- Stationary entropy: 0.9850 bits

**Entropy rate < Stationary entropy** because knowing current state reduces uncertainty about next state (Markov property).

**4. Interpret: what does entropy rate measure?**

**Entropy rate = average uncertainty about the next symbol given the process history**

In this Markov chain:
- H = 0.9200 bits/step: average surprise of each new state
- H(π) = 0.9850 bits: surprise if states were independent
- Difference: 0.0650 bits saved by using Markov structure

**Applications:**
- Text compression (predict next character from context)
- Speech coding (exploit temporal correlation)
- Measuring complexity of sequences

**NumPy verification:**
```python
P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# Find stationary distribution (eigenvector with eigenvalue 1)
eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
stationary = eigenvectors[:, idx].real
stationary = stationary / stationary.sum()

print(f"Stationary distribution: {stationary}")
# [0.5714, 0.4286]

# Entropy rate
H_rate = 0
for i in range(len(stationary)):
    for j in range(len(stationary)):
        if P[i, j] > 0:
            H_rate -= stationary[i] * P[i, j] * np.log2(P[i, j])

print(f"Entropy rate: {H_rate:.4f} bits/step")  # 0.9200

# Stationary entropy
H_stationary = -np.sum(stationary * np.log2(stationary))
print(f"Stationary entropy: {H_stationary:.4f} bits")  # 0.9850
print(f"Reduction: {H_stationary - H_rate:.4f} bits")  # 0.0650
```

---

### Challenge 2 Solution: Differential Entropy

Continuous uniform distribution X ~ Uniform(0, a)

**1. PDF: f(x) = 1/a for x ∈ [0, a]**

This is given. The PDF is constant over [0, a].

**2. Differential entropy: h(X) = -∫ f(x) log f(x) dx**

```
h(X) = -∫₀ᵃ (1/a) log(1/a) dx
```

Note: (1/a) is constant, and log(1/a) = -log(a)

```
h(X) = -∫₀ᵃ (1/a)·(-log a) dx
     = (log a / a) ∫₀ᵃ dx
     = (log a / a) · a
     = log a
```

**3. h(X) in terms of a**

```
h(X) = log a  (in nats if ln, in bits if log₂)
```

For log₂:
```
h(X) = log₂ a bits
```

**Examples:**
- a = 1: h(X) = 0 bits (no uncertainty, point mass at 0)
- a = 2: h(X) = 1 bit
- a = 4: h(X) = 2 bits
- a = 0.5: h(X) = -1 bit (can be negative!)

**4. How does it differ from discrete entropy?**

**Key differences:**

| Property | Discrete Entropy H(X) | Differential Entropy h(X) |
|----------|----------------------|--------------------------|
| **Range** | H(X) ≥ 0 | h(X) can be negative! |
| **Units** | Absolute (bits) | Relative (depends on units) |
| **Meaning** | Average surprise | Relative to uniform |
| **Maximum** | H(X) ≤ log N | No finite maximum |
| **Under scaling** | Unchanged | h(cX) = h(X) + log\|c\| |

**Why can h(X) be negative?**

Discrete: smallest probability is 1/N, so H ≥ 0

Continuous: PDF can be > 1 (concentrated mass), giving negative differential entropy

Example: Uniform(0, 0.5) has h(X) = log₂(0.5) = -1 bit

**Interpretation:** Differential entropy measures bits needed relative to uniform distribution with same support, not absolute information content.

**Python verification:**
```python
import scipy.stats as stats

# Uniform(0, a)
a_values = [0.25, 0.5, 1, 2, 4, 8]

for a in a_values:
    # Differential entropy (analytical)
    h_analytical = np.log2(a)

    # Numerical verification via scipy
    dist = stats.uniform(0, a)
    h_numerical = dist.entropy() / np.log(2)  # Convert nats to bits

    print(f"a = {a:4.2f}: h(X) = {h_analytical:6.2f} bits (analytical) = {h_numerical:6.2f} bits (numerical)")

# Output:
# a = 0.25: h(X) =  -2.00 bits (analytical) =  -2.00 bits (numerical)
# a = 0.50: h(X) =  -1.00 bits (analytical) =  -1.00 bits (numerical)
# a = 1.00: h(X) =   0.00 bits (analytical) =   0.00 bits (numerical)
# a = 2.00: h(X) =   1.00 bits (analytical) =   1.00 bits (numerical)
# a = 4.00: h(X) =   2.00 bits (analytical) =   2.00 bits (numerical)
# a = 8.00: h(X) =   3.00 bits (analytical) =   3.00 bits (numerical)
```

---

## Summary: Key Formulas

### Entropy
```
H(X) = -Σ p(x) log p(x)
```

### Joint Entropy
```
H(X,Y) = -ΣΣ p(x,y) log p(x,y)
```

### Conditional Entropy
```
H(Y|X) = Σ p(x) H(Y|X=x)
H(X,Y) = H(X) + H(Y|X)  [Chain Rule]
```

### Mutual Information
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
I(X;Y) = H(X) - H(X|Y)
I(X;Y) = H(Y) - H(Y|X)
I(X;Y) = 0 ⟺ X and Y independent
```

### Cross-Entropy
```
H(P,Q) = -Σ p(x) log q(x)
H(P,Q) ≥ H(P)  [Gibbs' Inequality]
```

### KL Divergence
```
D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
D_KL(P||Q) = H(P,Q) - H(P)
D_KL(P||Q) ≥ 0
D_KL(P||Q) ≠ D_KL(Q||P)  [Not symmetric!]
```

### Binary Cross-Entropy
```
L = -[y log(ŷ) + (1-y)log(1-ŷ)]
```

### Information Gain
```
IG = H(parent) - Σ (nᵢ/n) H(childᵢ)
```

---

## Tips & Common Mistakes

✅ **DO:**
- Use log₂ for bits, ln for nats
- Add small epsilon (10⁻¹⁰) to avoid log(0)
- Draw Venn diagrams for entropy relationships
- Check: I(X;Y) ≥ 0, H(X,Y) ≤ H(X) + H(Y)
- Remember: cross-entropy ≥ entropy always

❌ **DON'T:**
- Confuse H(X,Y) with H(X) + H(Y) (only equal if independent)
- Think KL divergence is symmetric (it's not!)
- Forget that differential entropy can be negative
- Mix up D_KL(P||Q) vs D_KL(Q||P) (order matters!)
- Use H(X|Y) when you mean H(Y|X) (conditioning direction matters)

---

**Congratulations!** You've completed all Information Theory exercises. These concepts form the mathematical foundation for:
- Loss functions in neural networks
- Compression algorithms
- Feature selection methods
- Probabilistic modeling
- Information bottleneck theory
- Understanding what neural networks learn

Keep these formulas handy - you'll use them constantly in ML!
