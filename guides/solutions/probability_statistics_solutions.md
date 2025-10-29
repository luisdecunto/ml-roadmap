# Probability & Statistics Solutions - Module 4

Comprehensive solutions with step-by-step mathematical work and code implementations.

---

## Part 1: Probability Basics

### Exercise 1.1 Solution: Basic Probability Rules

Bag contains: 5 red, 3 blue, 2 green marbles (total = 10)

**1. P(red) = 5/10 = 0.5 or 50%**

**2. P(not red) = 1 - P(red) = 1 - 0.5 = 0.5**

**3. P(both red, without replacement):**
```
First red: 5/10
Second red given first red: 4/9
P(both red) = (5/10) × (4/9) = 20/90 = 2/9 ≈ 0.222
```

**4. P(both red, with replacement):**
```
First red: 5/10
Second red: 5/10 (marble is replaced)
P(both red) = (5/10) × (5/10) = 25/100 = 0.25
```

**NumPy verification:**
```python
import numpy as np

# Simulation for without replacement
marbles = ['R']*5 + ['B']*3 + ['G']*2
trials = 100000
both_red = 0

for _ in range(trials):
    sample = np.random.choice(marbles, size=2, replace=False)
    if sample[0] == 'R' and sample[1] == 'R':
        both_red += 1

print(f"P(both red, no replacement) ≈ {both_red/trials:.4f}")
print(f"Theoretical: {2/9:.4f}")

# With replacement
both_red_replace = 0
for _ in range(trials):
    sample = np.random.choice(marbles, size=2, replace=True)
    if sample[0] == 'R' and sample[1] == 'R':
        both_red_replace += 1

print(f"P(both red, with replacement) ≈ {both_red_replace/trials:.4f}")
print(f"Theoretical: 0.25")
```

---

### Exercise 1.2 Solution: Conditional Probability

Given: P(A) = 0.6, P(B) = 0.5, P(A ∩ B) = 0.3

**1. P(A | B) = P(A ∩ B) / P(B)**
```
P(A | B) = 0.3 / 0.5 = 0.6
```

**2. P(B | A) = P(A ∩ B) / P(A)**
```
P(B | A) = 0.3 / 0.6 = 0.5
```

**3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)**
```
P(A ∪ B) = 0.6 + 0.5 - 0.3 = 0.8
```

**4. Are A and B independent?**
```
For independence: P(A ∩ B) = P(A) × P(B)
Check: 0.3 = 0.6 × 0.5 = 0.3 ✓

YES, A and B are independent!
```

**NumPy verification:**
```python
P_A = 0.6
P_B = 0.5
P_A_and_B = 0.3

P_A_given_B = P_A_and_B / P_B
P_B_given_A = P_A_and_B / P_A
P_A_or_B = P_A + P_B - P_A_and_B

print(f"P(A|B) = {P_A_given_B}")
print(f"P(B|A) = {P_B_given_A}")
print(f"P(A∪B) = {P_A_or_B}")
print(f"Independent? {np.isclose(P_A_and_B, P_A * P_B)}")
```

---

### Exercise 1.3 Solution: Law of Total Probability

Factory machines: A (30%), B (45%), C (25%)
Defect rates: A (2%), B (3%), C (4%)

**1. Probability tree:**
```
        Machine A (0.30) ─── Defective (0.02)
       /                  └── Good (0.98)
      /
Origin ── Machine B (0.45) ─── Defective (0.03)
      \                    └── Good (0.97)
       \
        Machine C (0.25) ─── Defective (0.04)
                         └── Good (0.96)
```

**2. P(defective) using law of total probability:**
```
P(D) = P(D|A)·P(A) + P(D|B)·P(B) + P(D|C)·P(C)
     = 0.02 × 0.30 + 0.03 × 0.45 + 0.04 × 0.25
     = 0.006 + 0.0135 + 0.010
     = 0.0295 or 2.95%
```

**3. P(from B | defective) using Bayes' theorem:**
```
P(B|D) = P(D|B) × P(B) / P(D)
       = (0.03 × 0.45) / 0.0295
       = 0.0135 / 0.0295
       ≈ 0.458 or 45.8%
```

**NumPy verification:**
```python
# Machine probabilities
P_A, P_B, P_C = 0.30, 0.45, 0.25

# Defect rates
P_D_given_A = 0.02
P_D_given_B = 0.03
P_D_given_C = 0.04

# Total probability of defect
P_D = P_D_given_A * P_A + P_D_given_B * P_B + P_D_given_C * P_C
print(f"P(defective) = {P_D:.4f}")

# Bayes' theorem for P(B | D)
P_B_given_D = (P_D_given_B * P_B) / P_D
print(f"P(from B | defective) = {P_B_given_D:.4f}")
```

---

### Exercise 1.4 Solution: Independence

P(A) = 0.4, P(B) = 0.3, A and B are independent

**1. P(A ∩ B) = P(A) × P(B)** (by independence)
```
P(A ∩ B) = 0.4 × 0.3 = 0.12
```

**2. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)**
```
P(A ∪ B) = 0.4 + 0.3 - 0.12 = 0.58
```

**3. P(A | B) = P(A)** (by independence)
```
P(A | B) = 0.4
```

**4. P(B | A) = P(B)** (by independence)
```
P(B | A) = 0.3
```

**Key insight:** For independent events, knowing that B occurred doesn't change the probability of A!

---

## Part 2: Distributions

### Exercise 2.1 Solution: Binomial Distribution

Fair coin flipped 10 times, n = 10, p = 0.5

**Formula:** P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

**1. P(exactly 6 heads):**
```
P(X = 6) = C(10,6) × 0.5^6 × 0.5^4
C(10,6) = 10!/(6!×4!) = 210
P(X = 6) = 210 × (0.5)^10 = 210/1024 ≈ 0.205
```

**2. P(at least 8 heads) = P(X ≥ 8) = P(X=8) + P(X=9) + P(X=10):**
```
P(X = 8) = C(10,8) × (0.5)^10 = 45/1024
P(X = 9) = C(10,9) × (0.5)^10 = 10/1024
P(X = 10) = C(10,10) × (0.5)^10 = 1/1024
P(X ≥ 8) = (45 + 10 + 1)/1024 = 56/1024 ≈ 0.0547
```

**3. E[X] = n × p:**
```
E[X] = 10 × 0.5 = 5
```

**4. Var(X) = n × p × (1-p):**
```
Var(X) = 10 × 0.5 × 0.5 = 2.5
```

**NumPy verification:**
```python
from scipy.stats import binom

n, p = 10, 0.5

# P(X = 6)
prob_6 = binom.pmf(6, n, p)
print(f"P(X=6) = {prob_6:.4f}")

# P(X >= 8)
prob_8_or_more = 1 - binom.cdf(7, n, p)
print(f"P(X≥8) = {prob_8_or_more:.4f}")

# Expected value and variance
print(f"E[X] = {binom.mean(n, p)}")
print(f"Var(X) = {binom.var(n, p)}")
```

---

### Exercise 2.2 Solution: Normal Distribution

X ~ N(μ=100, σ²=225), so σ = 15

**1. Standard deviation:**
```
σ = √225 = 15
```

**2. P(X < 115):**
```
Convert to z-score: z = (115 - 100)/15 = 1.0
P(X < 115) = Φ(1.0) ≈ 0.8413 (from z-table)
```

**3. P(85 < X < 115):**
```
z₁ = (85 - 100)/15 = -1.0
z₂ = (115 - 100)/15 = 1.0
P(85 < X < 115) = Φ(1.0) - Φ(-1.0)
                 = 0.8413 - 0.1587
                 = 0.6826 (≈68%, within 1 std dev)
```

**4. 95th percentile (P(X < x) = 0.95):**
```
z₀.₉₅ ≈ 1.645 (from z-table)
x = μ + z×σ = 100 + 1.645×15 ≈ 124.7
```

**NumPy verification:**
```python
from scipy.stats import norm

mu, sigma = 100, 15
dist = norm(mu, sigma)

print(f"P(X < 115) = {dist.cdf(115):.4f}")
print(f"P(85 < X < 115) = {dist.cdf(115) - dist.cdf(85):.4f}")
print(f"95th percentile = {dist.ppf(0.95):.2f}")
```

---

### Exercise 2.3 Solution: Uniform Distribution

X ~ Uniform(0, 10)

**1. PDF f(x):**
```
f(x) = 1/(b-a) = 1/(10-0) = 1/10 = 0.1 for x ∈ [0, 10]
f(x) = 0 otherwise
```

**2. P(3 < X < 7):**
```
For uniform: P(a < X < b) = (b-a)/(max-min)
P(3 < X < 7) = (7-3)/10 = 4/10 = 0.4
```

**3. E[X] = (a + b)/2:**
```
E[X] = (0 + 10)/2 = 5
```

**4. Var(X) = (b - a)²/12:**
```
Var(X) = (10 - 0)²/12 = 100/12 ≈ 8.33
```

**5. Sketch:**
```
PDF:            CDF:
f(x)            F(x)
0.1|▄▄▄▄▄▄      1.0|    ╱────
   |            0.5|  ╱
0  |______      0  |╱______
   0    10         0    10
```

**NumPy verification:**
```python
from scipy.stats import uniform

dist = uniform(0, 10)

print(f"P(3 < X < 7) = {dist.cdf(7) - dist.cdf(3):.4f}")
print(f"E[X] = {dist.mean():.4f}")
print(f"Var(X) = {dist.var():.4f}")
```

---

### Exercise 2.4 Solution: Exponential Distribution

X ~ Exponential(λ=0.5)

**1. PDF:**
```
f(x) = λe^(-λx) = 0.5e^(-0.5x) for x ≥ 0
```

**2. P(X > 2):**
```
P(X > 2) = e^(-λ×2) = e^(-0.5×2) = e^(-1) ≈ 0.368
```

**3. E[X] = 1/λ:**
```
E[X] = 1/0.5 = 2
```

**4. Memoryless property:**
```
P(X > s+t | X > s) = P(X > t)

This means: "If you've already waited s time units, the probability
of waiting an additional t units is the same as if you just started!"

Example: If a bus hasn't arrived in 10 minutes, the probability it
arrives in the next 5 minutes is the same as if you just got to the
stop. Past doesn't affect future.
```

**NumPy verification:**
```python
from scipy.stats import expon

lam = 0.5
dist = expon(scale=1/lam)  # scipy uses scale = 1/λ

print(f"P(X > 2) = {1 - dist.cdf(2):.4f}")
print(f"E[X] = {dist.mean():.4f}")

# Verify memoryless property
s, t = 3, 2
p_conditional = 1 - dist.cdf(s + t) / (1 - dist.cdf(s))
p_direct = 1 - dist.cdf(t)
print(f"Memoryless: {np.isclose(p_conditional, p_direct)}")
```

---

## Part 3: Bayes' Theorem

### Exercise 3.1 Solution: Medical Testing

Disease prevalence: P(D) = 0.01
Test accuracy: P(+|D) = P(-|¬D) = 0.95

**1. Find P(D | +) using Bayes' theorem:**

**Step 1: Calculate P(+) using law of total probability:**
```
P(+) = P(+|D)×P(D) + P(+|¬D)×P(¬D)
     = 0.95×0.01 + 0.05×0.99
     = 0.0095 + 0.0495
     = 0.059
```

**Step 2: Apply Bayes' theorem:**
```
P(D|+) = P(+|D) × P(D) / P(+)
       = (0.95 × 0.01) / 0.059
       = 0.0095 / 0.059
       ≈ 0.161 or 16.1%
```

**Interpretation:**
Even with a 95% accurate test, if you test positive, there's only 16% chance you actually have the disease! This is because the disease is rare (1% prevalence).

**NumPy verification:**
```python
P_D = 0.01
P_pos_given_D = 0.95
P_pos_given_not_D = 0.05

# Total probability
P_pos = P_pos_given_D * P_D + P_pos_given_not_D * (1 - P_D)

# Bayes' theorem
P_D_given_pos = (P_pos_given_D * P_D) / P_pos

print(f"P(+) = {P_pos:.4f}")
print(f"P(D|+) = {P_D_given_pos:.4f}")
```

---

### Exercise 3.2 Solution: Spam Filtering

P(spam) = 0.3, P("winner"|spam) = 0.7, P("winner"|not spam) = 0.05

**Find P(spam | "winner"):**

**Step 1: Calculate P("winner"):**
```
P(W) = P(W|S)×P(S) + P(W|¬S)×P(¬S)
     = 0.7×0.3 + 0.05×0.7
     = 0.21 + 0.035
     = 0.245
```

**Step 2: Apply Bayes' theorem:**
```
P(S|W) = P(W|S) × P(S) / P(W)
       = (0.7 × 0.3) / 0.245
       = 0.21 / 0.245
       ≈ 0.857 or 85.7%
```

**Conclusion:** If an email contains "winner", there's an 85.7% chance it's spam!

**NumPy verification:**
```python
P_S = 0.3
P_W_given_S = 0.7
P_W_given_not_S = 0.05

P_W = P_W_given_S * P_S + P_W_given_not_S * (1 - P_S)
P_S_given_W = (P_W_given_S * P_S) / P_W

print(f"P(spam | 'winner') = {P_S_given_W:.4f}")
```

---

### Exercise 3.3 Solution: Multi-class Bayesian Classification

Priors: P(C₁) = 0.5, P(C₂) = 0.3, P(C₃) = 0.2
Likelihoods at x=5: P(x=5|C₁) = 0.2, P(x=5|C₂) = 0.4, P(x=5|C₃) = 0.1

**Step 1: Calculate P(x=5):**
```
P(x=5) = P(x=5|C₁)×P(C₁) + P(x=5|C₂)×P(C₂) + P(x=5|C₃)×P(C₃)
       = 0.2×0.5 + 0.4×0.3 + 0.1×0.2
       = 0.1 + 0.12 + 0.02
       = 0.24
```

**Step 2: Calculate posteriors:**

**P(C₁ | x=5):**
```
P(C₁|x=5) = P(x=5|C₁) × P(C₁) / P(x=5)
          = (0.2 × 0.5) / 0.24
          = 0.1 / 0.24
          ≈ 0.417 or 41.7%
```

**P(C₂ | x=5):**
```
P(C₂|x=5) = (0.4 × 0.3) / 0.24
          = 0.12 / 0.24
          = 0.5 or 50%
```

**P(C₃ | x=5):**
```
P(C₃|x=5) = (0.1 × 0.2) / 0.24
          = 0.02 / 0.24
          ≈ 0.083 or 8.3%
```

**Verification:** 0.417 + 0.5 + 0.083 = 1.0 ✓

**Prediction:** Choose C₂ (highest posterior probability of 50%)

**NumPy implementation:**
```python
# Priors
priors = np.array([0.5, 0.3, 0.2])

# Likelihoods
likelihoods = np.array([0.2, 0.4, 0.1])

# Evidence
evidence = np.sum(likelihoods * priors)

# Posteriors
posteriors = (likelihoods * priors) / evidence

print("Posteriors:", posteriors)
print("Predicted class:", np.argmax(posteriors) + 1)
```

---

## Part 4: Maximum Likelihood

### Exercise 4.1 Solution: MLE for Bernoulli

100 flips, 60 heads

**1. Likelihood function:**
```
L(p) = p^60 × (1-p)^40
```

**2. Log-likelihood:**
```
log L(p) = 60 log(p) + 40 log(1-p)
```

**3. Take derivative and set to 0:**
```
d/dp [log L(p)] = 60/p - 40/(1-p)

Set to 0:
60/p = 40/(1-p)
60(1-p) = 40p
60 - 60p = 40p
60 = 100p
p̂ = 0.6
```

**4. Verify:**
```
p̂ = number of heads / total flips = 60/100 = 0.6 ✓
```

**NumPy implementation:**
```python
heads = 60
total = 100

# MLE estimate
p_hat = heads / total
print(f"MLE estimate: p̂ = {p_hat}")

# Verify with numerical optimization
from scipy.optimize import minimize_scalar

def neg_log_likelihood(p):
    return -(heads * np.log(p) + (total - heads) * np.log(1 - p))

result = minimize_scalar(neg_log_likelihood, bounds=(0.001, 0.999), method='bounded')
print(f"Numerical MLE: {result.x:.4f}")
```

---

### Exercise 4.2 Solution: MLE for Normal Distribution

Data: [2, 4, 3, 5, 6]

**1. Likelihood for N(μ, σ²):**
```
L(μ, σ²) = ∏ᵢ (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))
```

**2. Log-likelihood:**
```
log L = -n/2 log(2πσ²) - (1/2σ²)Σ(xᵢ-μ)²
```

**3. Find MLE for μ:**
```
∂/∂μ [log L] = (1/σ²)Σ(xᵢ-μ)

Set to 0:
Σ(xᵢ-μ) = 0
Σxᵢ - nμ = 0
μ̂ = (1/n)Σxᵢ = x̄ (sample mean)
```

**4. For our data:**
```
μ̂ = (2 + 4 + 3 + 5 + 6)/5 = 20/5 = 4
```

**NumPy implementation:**
```python
data = np.array([2, 4, 3, 5, 6])

# MLE for mean
mu_hat = np.mean(data)
print(f"μ̂ = {mu_hat}")

# MLE for variance (uses n, not n-1)
sigma_hat_sq = np.var(data, ddof=0)
print(f"σ̂² = {sigma_hat_sq}")
```

---

### Exercise 4.3 Solution: MLE for Exponential

Time data: [1.2, 0.8, 2.1, 1.5, 0.9] hours

**1. Exponential PDF:** f(x|λ) = λe^(-λx)

**2. Likelihood:**
```
L(λ) = ∏ᵢ λe^(-λxᵢ) = λⁿ × e^(-λΣxᵢ)
```

**3. Log-likelihood:**
```
log L(λ) = n log(λ) - λΣxᵢ
```

**4. Find MLE:**
```
d/dλ [log L] = n/λ - Σxᵢ

Set to 0:
n/λ = Σxᵢ
λ̂ = n / Σxᵢ = 1/x̄
```

**5. Calculate:**
```
Data: [1.2, 0.8, 2.1, 1.5, 0.9]
x̄ = (1.2 + 0.8 + 2.1 + 1.5 + 0.9)/5 = 6.5/5 = 1.3
λ̂ = 1/1.3 ≈ 0.769
```

**NumPy implementation:**
```python
data = np.array([1.2, 0.8, 2.1, 1.5, 0.9])

# MLE for λ
lambda_hat = 1 / np.mean(data)
print(f"λ̂ = {lambda_hat:.4f}")

# Verify with scipy
from scipy.stats import expon
fit_params = expon.fit(data, floc=0)  # Fix location at 0
print(f"Fitted λ (1/scale): {1/fit_params[1]:.4f}")
```

---

## Part 5: Statistical Inference

### Exercise 5.1 Solution: Confidence Intervals

Sample: [10, 12, 13, 11, 14, 12, 10, 13], n = 8

**1. Sample mean:**
```
x̄ = (10+12+13+11+14+12+10+13)/8 = 95/8 = 11.875
```

**2. Sample standard deviation:**
```
s² = (1/n-1)Σ(xᵢ - x̄)²
   = (1/7)[(10-11.875)² + (12-11.875)² + ... + (13-11.875)²]
   = (1/7)[3.516 + 0.016 + 1.266 + 0.766 + 4.516 + 0.016 + 3.516 + 1.266]
   = (1/7) × 14.875
   ≈ 2.125

s = √2.125 ≈ 1.458
```

**3. 95% CI with t-distribution (df = 7):**
```
t* ≈ 2.365 (from t-table)
```

**4. Confidence interval:**
```
SE = s/√n = 1.458/√8 ≈ 0.516
CI = x̄ ± t* × SE
   = 11.875 ± 2.365 × 0.516
   = 11.875 ± 1.22
   = (10.655, 13.095)
```

**5. Interpretation:**
"We are 95% confident that the true population mean lies between 10.655 and 13.095. If we repeated this sampling procedure many times, 95% of the computed intervals would contain the true mean."

**NumPy implementation:**
```python
from scipy import stats

data = np.array([10, 12, 13, 11, 14, 12, 10, 13])

mean = np.mean(data)
std = np.std(data, ddof=1)  # Sample std
se = std / np.sqrt(len(data))

# 95% confidence interval
t_crit = stats.t.ppf(0.975, df=len(data)-1)
ci = (mean - t_crit*se, mean + t_crit*se)

print(f"Mean: {mean}")
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
```

---

### Exercise 5.2 Solution: Hypothesis Testing

Test if coin is fair: 100 flips, 60 heads
H₀: p = 0.5, H₁: p ≠ 0.5

**1. Test statistic:**
```
z = (p̂ - p₀) / √(p₀(1-p₀)/n)
```

**2. Calculate:**
```
p̂ = 60/100 = 0.6
p₀ = 0.5
n = 100

SE = √(0.5 × 0.5 / 100) = √0.0025 = 0.05

z = (0.6 - 0.5) / 0.05 = 0.1 / 0.05 = 2.0
```

**3. Find p-value (two-tailed):**
```
P(|Z| ≥ 2.0) = 2 × P(Z ≥ 2.0)
             = 2 × 0.0228
             ≈ 0.0456
```

**4. Decision at α = 0.05:**
```
p-value (0.0456) < α (0.05)
REJECT H₀

Conclusion: There is sufficient evidence to conclude the coin is not fair.
```

**NumPy implementation:**
```python
n = 100
p_hat = 0.6
p_0 = 0.5

# Calculate z-statistic
se = np.sqrt(p_0 * (1 - p_0) / n)
z = (p_hat - p_0) / se

# Two-tailed p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"z-statistic: {z:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Reject H₀? {p_value < 0.05}")
```

---

### Exercise 5.3 Solution: Comparing Means

Group A: [23, 25, 27, 24, 26]
Group B: [20, 22, 19, 21, 23]

**1. Calculate means:**
```
x̄₁ = (23+25+27+24+26)/5 = 125/5 = 25
x̄₂ = (20+22+19+21+23)/5 = 105/5 = 21
```

**2. Calculate standard deviations:**
```
s₁² = (1/4)[(23-25)² + (25-25)² + (27-25)² + (24-25)² + (26-25)²]
    = (1/4)[4 + 0 + 4 + 1 + 1] = 10/4 = 2.5
s₁ = √2.5 ≈ 1.581

s₂² = (1/4)[(20-21)² + (22-21)² + (19-21)² + (21-21)² + (23-21)²]
    = (1/4)[1 + 1 + 4 + 0 + 4] = 10/4 = 2.5
s₂ = √2.5 ≈ 1.581
```

**3. Pooled variance t-test:**
```
s²ₚ = [(n₁-1)s₁² + (n₂-1)s₂²] / (n₁+n₂-2)
    = [4×2.5 + 4×2.5] / 8
    = 20/8 = 2.5

SE = √(s²ₚ/n₁ + s²ₚ/n₂) = √(2.5/5 + 2.5/5) = √1 = 1

t = (x̄₁ - x̄₂) / SE = (25 - 21) / 1 = 4
```

**4. Find p-value:**
```
df = n₁ + n₂ - 2 = 8
For t = 4 with df = 8, p-value ≈ 0.004 (two-tailed)
```

**5. Conclusion at α = 0.05:**
```
p-value (0.004) < α (0.05)
REJECT H₀

There is strong evidence that the means of the two groups differ.
Group A has a significantly higher mean than Group B.
```

**NumPy implementation:**
```python
groupA = np.array([23, 25, 27, 24, 26])
groupB = np.array([20, 22, 19, 21, 23])

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(groupA, groupB)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Reject H₀? {p_value < 0.05}")
```

---

## Challenge Problems

### Challenge 1 Solution: Monte Carlo Estimation

Estimate π by random sampling

**Method:**
1. Generate random points in [0,1] × [0,1]
2. Count points inside quarter circle (x² + y² ≤ 1)
3. π ≈ 4 × (points inside) / (total points)

**Why it works:**
Area of quarter circle = πr²/4 = π/4 (for r=1)
Area of square = 1
Ratio = π/4, so π = 4 × ratio

**Implementation:**
```python
def estimate_pi(n_points):
    # Generate random points
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)

    # Count points inside quarter circle
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside) / n_points

    return pi_estimate

# Test with different sample sizes
for n in [1000, 10000, 100000]:
    pi_est = estimate_pi(n)
    error = abs(pi_est - np.pi)
    print(f"n={n:6d}: π ≈ {pi_est:.5f}, error = {error:.5f}")

# Output shows error decreases as √n
```

**Results:**
```
n=  1000: π ≈ 3.14400, error = 0.00241
n= 10000: π ≈ 3.14040, error = 0.00119
n=100000: π ≈ 3.14207, error = 0.00048

Error decreases proportionally to 1/√n (Monte Carlo rate)
```

---

### Challenge 2 Solution: Central Limit Theorem

Sample from Uniform(0, 1) and examine sampling distribution

**Theory:**
- Population: Uniform(0,1), μ = 0.5, σ² = 1/12 ≈ 0.0833
- Sampling distribution: Mean = μ, Variance = σ²/n

**Implementation:**
```python
import matplotlib.pyplot as plt

def clt_demo(n, num_samples=1000):
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.uniform(0, 1, n)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of Sample Means (n={n})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')

    # Overlay theoretical normal
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    theoretical_mean = 0.5
    theoretical_std = np.sqrt(1/12 / n)
    y = stats.norm.pdf(x, theoretical_mean, theoretical_std)
    plt.plot(x, y, 'r-', linewidth=2, label='Theoretical Normal')
    plt.legend()
    plt.show()

    # Calculate statistics
    print(f"\nn = {n}:")
    print(f"Sample mean of means: {np.mean(sample_means):.4f} (theoretical: 0.5)")
    print(f"Sample variance: {np.var(sample_means):.6f} (theoretical: {1/12/n:.6f})")

# Test with n=5 and n=30
clt_demo(5)
clt_demo(30)
```

**Observations:**
- With n=5: Distribution already bell-shaped but slightly uneven
- With n=30: Distribution very close to perfect normal
- Mean stays at 0.5 for both
- Variance decreases as 1/n
- CLT: Even non-normal populations → normal sampling distribution!

---

## Summary

**Key Concepts:**

1. **Probability Rules:** Addition, multiplication, conditional probability
2. **Bayes' Theorem:** Update beliefs with evidence
3. **Distributions:** Binomial, Normal, Uniform, Exponential
4. **MLE:** Find parameters that maximize likelihood of observed data
5. **Confidence Intervals:** Range of plausible values for parameter
6. **Hypothesis Testing:** Statistical evidence for/against claims
7. **Central Limit Theorem:** Foundation of statistical inference

**Practical Tips:**
- Always check independence assumptions
- Use appropriate distribution for your data
- Understand difference between population and sample statistics
- Be careful with one-tailed vs two-tailed tests
- Monte Carlo methods are powerful for complex problems
