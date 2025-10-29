# Probability & Statistics Exercises - Module 4

**Time:** 3-4 hours
**Difficulty:** Intermediate
**Materials needed:** Paper, pencil, calculator, NumPy

Complete these exercises by hand first, then verify with NumPy. Solutions are in `guides/solutions/probability_statistics_solutions.md`

---

## Part 1: Probability Basics (35 min)

### Exercise 1.1: Basic Probability Rules
A bag contains 5 red, 3 blue, and 2 green marbles:

1. What is P(red)?
2. What is P(not red)?
3. If you draw 2 marbles without replacement, what is P(both red)?
4. With replacement, what is P(both red)?

### Exercise 1.2: Conditional Probability
Given:
- P(A) = 0.6
- P(B) = 0.5
- P(A ∩ B) = 0.3

Calculate:
1. P(A | B)
2. P(B | A)
3. P(A ∪ B)
4. Are A and B independent? Show your reasoning.

### Exercise 1.3: Law of Total Probability
A factory has three machines A, B, C producing 30%, 45%, 25% of items respectively. Defect rates are 2%, 3%, 4%.

1. Draw a probability tree
2. What is P(defective)?
3. If an item is defective, what is P(from machine B)?

### Exercise 1.4: Independence
For events A and B:
- P(A) = 0.4
- P(B) = 0.3
- Assume A and B are independent

Calculate:
1. P(A ∩ B)
2. P(A ∪ B)
3. P(A | B)
4. P(B | A)

---

## Part 2: Distributions (40 min)

### Exercise 2.1: Binomial Distribution
A fair coin is flipped 10 times:

1. What is P(exactly 6 heads)?
   - Use formula: P(X = k) = C(n,k) · p^k · (1-p)^(n-k)
2. What is P(at least 8 heads)?
3. What is E[X] (expected number of heads)?
4. What is Var(X)?

### Exercise 2.2: Normal Distribution
Given X ~ N(μ=100, σ²=225):

1. What is the standard deviation?
2. Calculate P(X < 115) using z-score
3. Calculate P(85 < X < 115)
4. What value of x has P(X < x) = 0.95?

### Exercise 2.3: Uniform Distribution
X ~ Uniform(0, 10):

1. What is the PDF f(x)?
2. Calculate P(3 < X < 7)
3. What is E[X]?
4. What is Var(X)?
5. Plot or sketch the PDF and CDF

### Exercise 2.4: Exponential Distribution
Time between events follows Exponential(λ=0.5):

1. Write the PDF f(x) = λe^(-λx)
2. Calculate P(X > 2)
3. What is E[X] = 1/λ?
4. This models memoryless processes - what does that mean?

---

## Part 3: Bayes' Theorem (30 min)

### Exercise 3.1: Medical Testing
A disease affects 1% of population. A test is 95% accurate (both sensitivity and specificity):

1. If you test positive, what is P(you have disease)?
2. Use Bayes' theorem: P(D|+) = P(+|D)·P(D) / P(+)
3. Calculate P(+) using law of total probability
4. Interpret the result - surprising?

### Exercise 3.2: Spam Filtering
Email dataset:
- P(spam) = 0.3
- P("winner" | spam) = 0.7
- P("winner" | not spam) = 0.05

1. If email contains "winner", what is P(spam)?
2. Use Bayes' theorem
3. Show all steps

### Exercise 3.3: Multi-class Bayesian Classification
Three classes with priors:
- P(C₁) = 0.5, P(C₂) = 0.3, P(C₃) = 0.2

Likelihoods for feature x=5:
- P(x=5 | C₁) = 0.2
- P(x=5 | C₂) = 0.4
- P(x=5 | C₃) = 0.1

1. Calculate P(C₁ | x=5)
2. Calculate P(C₂ | x=5)
3. Calculate P(C₃ | x=5)
4. Which class would you predict?

---

## Part 4: Maximum Likelihood (35 min)

### Exercise 4.1: MLE for Bernoulli
You flip a coin 100 times, get 60 heads:

1. Write likelihood function L(p) = p^60 · (1-p)^40
2. Write log-likelihood: log L(p)
3. Take derivative and set to 0
4. Solve for p̂ (MLE estimate)
5. Verify it equals 60/100

### Exercise 4.2: MLE for Normal Distribution
Dataset: [2, 4, 3, 5, 6]

1. Write likelihood for N(μ, σ²): L(μ, σ²) = ∏ᵢ (1/√(2πσ²)) · exp(-(xᵢ-μ)²/(2σ²))
2. Write log-likelihood
3. Find MLE for μ by taking ∂/∂μ and setting to 0
4. Show μ̂ = (1/n)Σxᵢ (sample mean)
5. Calculate μ̂ for the dataset

### Exercise 4.3: MLE for Exponential
Time data: [1.2, 0.8, 2.1, 1.5, 0.9] (in hours)

1. Exponential PDF: f(x|λ) = λe^(-λx)
2. Write likelihood L(λ)
3. Write log-likelihood
4. Find MLE: λ̂ = n / Σxᵢ = 1/x̄
5. Calculate λ̂ for the data

---

## Part 5: Statistical Inference (35 min)

### Exercise 5.1: Confidence Intervals
Sample: [10, 12, 13, 11, 14, 12, 10, 13] (n=8)

1. Calculate sample mean x̄
2. Calculate sample standard deviation s
3. For 95% CI, use t-distribution with df=7: t* ≈ 2.365
4. Calculate CI: x̄ ± t* · (s/√n)
5. Interpret: what does this interval mean?

### Exercise 5.2: Hypothesis Testing
Test if a coin is fair. You flip 100 times, get 60 heads.
- H₀: p = 0.5 (fair coin)
- H₁: p ≠ 0.5 (not fair)

1. Calculate test statistic: z = (p̂ - p₀) / √(p₀(1-p₀)/n)
2. p̂ = 60/100 = 0.6
3. Calculate z-value
4. Find p-value (two-tailed)
5. At α = 0.05, do you reject H₀?

### Exercise 5.3: Comparing Means
Group A: [23, 25, 27, 24, 26] (n₁=5)
Group B: [20, 22, 19, 21, 23] (n₂=5)

1. Calculate x̄₁ and x̄₂
2. Calculate s₁ and s₂
3. Test H₀: μ₁ = μ₂ vs H₁: μ₁ ≠ μ₂
4. Use pooled variance t-test
5. What do you conclude at α = 0.05?

---

## Challenge Problems (Optional)

### Challenge 1: Monte Carlo Estimation
Estimate π using Monte Carlo:

1. Generate random points (x, y) in [0, 1]×[0, 1]
2. Count how many fall inside quarter circle: x² + y² ≤ 1
3. Estimate π ≈ 4 · (points inside) / (total points)
4. Run with 1000, 10000, 100000 points
5. How does accuracy improve?

### Challenge 2: Central Limit Theorem
Sample from Uniform(0, 1):

1. Take samples of size n=5, compute 1000 sample means
2. Plot histogram of sample means
3. Repeat for n=30
4. Compare distributions - converging to normal?
5. Calculate mean and variance of sampling distribution

---

## NumPy Verification

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Exercise 1.1 - Simulation
marbles = ['R']*5 + ['B']*3 + ['G']*2
trials = 10000
red_count = sum(1 for _ in range(trials) if np.random.choice(marbles) == 'R')
print(f"P(red) ≈ {red_count/trials}")

# Exercise 2.1 - Binomial
from scipy.stats import binom
n, p = 10, 0.5
print(f"P(X=6) = {binom.pmf(6, n, p)}")
print(f"P(X>=8) = {1 - binom.cdf(7, n, p)}")
print(f"E[X] = {binom.mean(n, p)}")
print(f"Var(X) = {binom.var(n, p)}")

# Exercise 2.2 - Normal
mu, sigma = 100, 15
x = stats.norm(mu, sigma)
print(f"P(X < 115) = {x.cdf(115)}")
print(f"P(85 < X < 115) = {x.cdf(115) - x.cdf(85)}")
print(f"95th percentile = {x.ppf(0.95)}")

# Exercise 3.1 - Bayes' Theorem
P_D = 0.01
P_pos_given_D = 0.95
P_pos_given_not_D = 0.05
P_pos = P_pos_given_D * P_D + P_pos_given_not_D * (1 - P_D)
P_D_given_pos = (P_pos_given_D * P_D) / P_pos
print(f"P(Disease | Positive) = {P_D_given_pos:.4f}")

# Exercise 4.2 - MLE for Normal
data = np.array([2, 4, 3, 5, 6])
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # MLE uses n, not n-1
print(f"μ̂ = {mu_mle}, σ̂ = {sigma_mle}")

# Exercise 5.1 - Confidence Interval
data = np.array([10, 12, 13, 11, 14, 12, 10, 13])
mean = np.mean(data)
std = np.std(data, ddof=1)  # Sample std uses n-1
se = std / np.sqrt(len(data))
t_crit = stats.t.ppf(0.975, df=len(data)-1)  # 95% CI, two-tailed
ci = (mean - t_crit*se, mean + t_crit*se)
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")

# Exercise 5.2 - Hypothesis Test
n = 100
p_hat = 0.6
p_0 = 0.5
se = np.sqrt(p_0 * (1-p_0) / n)
z = (p_hat - p_0) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed
print(f"z = {z:.3f}, p-value = {p_value:.4f}")

# Challenge 1 - Monte Carlo
def estimate_pi(n_points):
    points = np.random.uniform(0, 1, (n_points, 2))
    inside = np.sum(points[:, 0]**2 + points[:, 1]**2 <= 1)
    return 4 * inside / n_points

for n in [1000, 10000, 100000]:
    pi_est = estimate_pi(n)
    print(f"n={n}: π ≈ {pi_est:.4f}, error = {abs(pi_est - np.pi):.4f}")

# Challenge 2 - Central Limit Theorem
sample_sizes = [5, 30]
for n in sample_sizes:
    sample_means = [np.mean(np.random.uniform(0, 1, n)) for _ in range(1000)]
    plt.figure()
    plt.hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of Sample Means (n={n})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.show()
    print(f"n={n}: Mean={np.mean(sample_means):.3f}, Var={np.var(sample_means):.4f}")
```

---

## Tips for Success

1. **Draw diagrams** - Probability trees and Venn diagrams help
2. **Check your denominator** - P(A|B) uses P(B), not P(A)
3. **Bayes' intuition** - Update beliefs with evidence
4. **Simulate** - Use NumPy to verify tricky probability problems
5. **Units matter** - Keep track of what your numbers represent
6. **Sanity checks** - Probabilities must be in [0, 1]
7. **Log-likelihood** - Always use logs for products
8. **CI interpretation** - The interval is random, not the parameter
