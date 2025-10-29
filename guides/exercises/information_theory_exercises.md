# Information Theory Exercises - Module 5

**Time:** 2-3 hours
**Difficulty:** Intermediate-Advanced
**Materials needed:** Paper, pencil, calculator, NumPy

Complete these exercises by hand first, then verify with NumPy. Solutions are in `guides/solutions/information_theory_solutions.md`

---

## Part 1: Entropy Basics (25 min)

### Exercise 1.1: Binary Entropy
A biased coin has P(H) = 0.7, P(T) = 0.3:

1. Calculate entropy H(X) = -Σ p(x) log₂ p(x)
2. What are the units? (bits)
3. Compare with fair coin (P(H) = 0.5)
4. Which has more entropy/uncertainty?

### Exercise 1.2: Discrete Distribution Entropy
Random variable X with distribution:
- P(X=1) = 0.5
- P(X=2) = 0.25
- P(X=3) = 0.125
- P(X=4) = 0.125

1. Calculate H(X)
2. What is the maximum possible entropy for 4 outcomes?
3. How close is this distribution to maximum entropy?

### Exercise 1.3: Uniform vs Non-uniform
Compare entropies:
- A: P(a) = 0.25 for all 4 outcomes
- B: P(b) = [0.7, 0.1, 0.1, 0.1]

1. Calculate H(A)
2. Calculate H(B)
3. Which distribution is more "surprising"?
4. Explain the relationship between entropy and predictability

---

## Part 2: Joint and Conditional Entropy (30 min)

### Exercise 2.1: Joint Entropy
Two binary variables X and Y with joint distribution:

| X\Y | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.3 | 0.2 |
| X=1 | 0.1 | 0.4 |

1. Calculate H(X, Y) = -ΣΣ p(x,y) log p(x,y)
2. Calculate marginal H(X)
3. Calculate marginal H(Y)
4. Verify: H(X, Y) ≤ H(X) + H(Y)

### Exercise 2.2: Conditional Entropy
Using same joint distribution as 2.1:

1. Calculate P(Y|X=0) and P(Y|X=1)
2. Calculate H(Y|X=0) and H(Y|X=1)
3. Calculate H(Y|X) = Σ P(x) H(Y|X=x)
4. Verify: H(X, Y) = H(X) + H(Y|X)

### Exercise 2.3: Chain Rule
For three variables with joint P(X, Y, Z):

1. Write the chain rule: H(X,Y,Z) = H(X) + H(Y|X) + H(Z|X,Y)
2. Given: H(X) = 2, H(Y|X) = 1.5, H(Z|X,Y) = 1
3. Calculate H(X, Y, Z)
4. Why is this useful for modeling sequential data?

---

## Part 3: Mutual Information (30 min)

### Exercise 3.1: Computing Mutual Information
Using joint distribution from Exercise 2.1:

1. Calculate I(X; Y) = H(X) + H(Y) - H(X, Y)
2. Alternative: I(X; Y) = H(X) - H(X|Y)
3. Verify both give same result
4. Are X and Y independent? (Check if I(X; Y) = 0)

### Exercise 3.2: Independence Test
Test independence of X and Y:

| X\Y | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.3 | 0.3 |
| X=1 | 0.2 | 0.2 |

1. Calculate I(X; Y)
2. Are X and Y independent?
3. Verify: P(X,Y) = P(X)P(Y) for all x,y

### Exercise 3.3: Mutual Information Properties
Given: I(X; Y) = 0.5 bits, H(X) = 2 bits, H(Y) = 1.5 bits

1. Calculate H(X|Y)
2. Calculate H(Y|X)
3. Calculate H(X, Y)
4. Draw the entropy Venn diagram

---

## Part 4: Cross-Entropy and KL Divergence (35 min)

### Exercise 4.1: Cross-Entropy
True distribution P: [0.5, 0.3, 0.2]
Model distribution Q: [0.4, 0.4, 0.2]

1. Calculate H(P, Q) = -Σ p(x) log q(x)
2. Calculate H(P) (entropy of true distribution)
3. Which is larger: H(P, Q) or H(P)? Why?
4. What does this tell us about using Q to encode data from P?

### Exercise 4.2: KL Divergence
Same distributions as 4.1:

1. Calculate D_KL(P || Q) = Σ p(x) log(p(x)/q(x))
2. Verify: D_KL(P || Q) = H(P, Q) - H(P)
3. Calculate D_KL(Q || P)
4. Is KL divergence symmetric? (Compare D_KL(P||Q) vs D_KL(Q||P))

### Exercise 4.3: KL Divergence Properties
P = [0.8, 0.2], Q = [0.6, 0.4], R = [0.5, 0.5]

1. Calculate D_KL(P || Q)
2. Calculate D_KL(P || R)
3. Calculate D_KL(Q || R)
4. Which model (Q or R) is "closer" to P?

### Exercise 4.4: Cross-Entropy in Classification
Binary classification with 4 samples:
- True labels: [1, 0, 1, 1]
- Predictions: [0.9, 0.2, 0.8, 0.7]

1. Calculate cross-entropy loss: L = -(1/n)Σ[y log(ŷ) + (1-y)log(1-ŷ)]
2. Calculate for each sample
3. Take average
4. What happens if prediction is wrong but confident?

---

## Part 5: Applications to ML (30 min)

### Exercise 5.1: Optimal Code Length
Message with symbol probabilities: P(A)=0.5, P(B)=0.25, P(C)=0.125, P(D)=0.125

1. Calculate entropy H (minimum average bits needed)
2. Design Huffman code (assign shorter codes to more frequent symbols)
3. Calculate average code length
4. How close is it to entropy?

### Exercise 5.2: Decision Tree Splitting
Dataset for classifying [Y/N]:

Parent: [6Y, 4N]

Split A:
- Left: [4Y, 1N]
- Right: [2Y, 3N]

1. Calculate entropy of parent node
2. Calculate entropy of left child
3. Calculate entropy of right child
4. Calculate information gain: IG = H(parent) - Σ (nᵢ/n)H(childᵢ)
5. Is this a good split?

### Exercise 5.3: Softmax and Cross-Entropy
Logits: z = [2.0, 1.0, 0.1]
True label: class 0 (one-hot: [1, 0, 0])

1. Calculate softmax: p(i) = exp(zᵢ) / Σ exp(zⱼ)
2. Calculate cross-entropy: -Σ yᵢ log(p(i))
3. What if true label was class 2?
4. Why do we use log in the loss?

---

## Challenge Problems (Optional)

### Challenge 1: Entropy Rate of Markov Chain
2-state Markov chain with transition matrix:
```
     [0.7  0.3]
P =  [0.4  0.6]
```

1. Find stationary distribution π (solve πP = π)
2. Calculate entropy rate: H = -ΣᵢΣⱼ πᵢ pᵢⱼ log pᵢⱼ
3. Compare with entropy of stationary distribution
4. Interpret: what does entropy rate measure?

### Challenge 2: Differential Entropy
Continuous uniform distribution X ~ Uniform(0, a):

1. PDF: f(x) = 1/a for x ∈ [0, a]
2. Differential entropy: h(X) = -∫ f(x) log f(x) dx
3. Calculate h(X) in terms of a
4. How does it differ from discrete entropy?

---

## NumPy Verification

```python
import numpy as np
from scipy.stats import entropy

# Exercise 1.1 - Binary Entropy
p = np.array([0.7, 0.3])
H = -np.sum(p * np.log2(p))
print(f"H(X) = {H:.4f} bits")

# Fair coin comparison
p_fair = np.array([0.5, 0.5])
H_fair = -np.sum(p_fair * np.log2(p_fair))
print(f"H(fair coin) = {H_fair:.4f} bits")

# Exercise 1.2 - Discrete Distribution
p = np.array([0.5, 0.25, 0.125, 0.125])
H = -np.sum(p * np.log2(p))
H_max = np.log2(len(p))
print(f"H(X) = {H:.4f} bits, H_max = {H_max:.4f} bits")

# Exercise 2.1 - Joint Entropy
joint = np.array([[0.3, 0.2],
                   [0.1, 0.4]])
H_joint = -np.sum(joint * np.log2(joint + 1e-10))  # Add epsilon for numerical stability
print(f"H(X,Y) = {H_joint:.4f} bits")

# Marginals
p_x = joint.sum(axis=1)
p_y = joint.sum(axis=0)
H_x = -np.sum(p_x * np.log2(p_x + 1e-10))
H_y = -np.sum(p_y * np.log2(p_y + 1e-10))
print(f"H(X) = {H_x:.4f}, H(Y) = {H_y:.4f}")

# Exercise 3.1 - Mutual Information
I_xy = H_x + H_y - H_joint
print(f"I(X;Y) = {I_xy:.4f} bits")

# Exercise 4.1 - Cross-Entropy
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.4, 0.4, 0.2])
H_p = -np.sum(P * np.log2(P))
H_pq = -np.sum(P * np.log2(Q))
print(f"H(P) = {H_p:.4f}, H(P,Q) = {H_pq:.4f}")

# Exercise 4.2 - KL Divergence
D_kl = np.sum(P * np.log2(P / Q))
print(f"D_KL(P||Q) = {D_kl:.4f} bits")
# Verify relationship
print(f"H(P,Q) - H(P) = {H_pq - H_p:.4f}")

# Exercise 4.4 - Binary Cross-Entropy
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.2, 0.8, 0.7])
bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
print(f"Binary Cross-Entropy = {bce:.4f}")

# Exercise 5.2 - Information Gain
def entropy_helper(counts):
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))

parent = np.array([6, 4])
left = np.array([4, 1])
right = np.array([2, 3])
H_parent = entropy_helper(parent)
H_left = entropy_helper(left)
H_right = entropy_helper(right)
n = parent.sum()
IG = H_parent - (left.sum()/n * H_left + right.sum()/n * H_right)
print(f"Information Gain = {IG:.4f} bits")

# Exercise 5.3 - Softmax and Cross-Entropy
logits = np.array([2.0, 1.0, 0.1])
probs = np.exp(logits) / np.sum(np.exp(logits))
true_label = np.array([1, 0, 0])  # One-hot for class 0
ce_loss = -np.sum(true_label * np.log(probs))
print(f"Softmax probs: {probs}")
print(f"Cross-Entropy Loss: {ce_loss:.4f}")

# Challenge 1 - Stationary Distribution
P = np.array([[0.7, 0.3],
              [0.4, 0.6]])
# Find stationary distribution
eigenvalues, eigenvectors = np.linalg.eig(P.T)
stationary = eigenvectors[:, np.argmax(eigenvalues)].real
stationary = stationary / stationary.sum()
print(f"Stationary distribution: {stationary}")

# Entropy rate
H_rate = 0
for i in range(len(stationary)):
    for j in range(len(stationary)):
        if P[i, j] > 0:
            H_rate -= stationary[i] * P[i, j] * np.log2(P[i, j])
print(f"Entropy rate: {H_rate:.4f} bits/step")
```

---

## Tips for Success

1. **Use log₂** - Information theory typically uses bits (log base 2)
2. **Handle zeros** - Add small epsilon (10⁻¹⁰) to avoid log(0)
3. **Units matter** - Bits (log₂), nats (ln), or dits (log₁₀)
4. **Venn diagrams** - Visualize relationships between entropies
5. **Cross-entropy ≥ Entropy** - Always! Model can't beat true distribution
6. **KL is not symmetric** - D_KL(P||Q) ≠ D_KL(Q||P)
7. **Information gain** - Reduction in entropy/uncertainty
8. **Softmax + Cross-Entropy** - Standard combo for classification
