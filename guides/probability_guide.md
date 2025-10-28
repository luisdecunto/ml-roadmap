# Probability & Information Theory

**Time:** 6-8 hours | **Difficulty:** Intermediate

## Gaussian Distribution

```python
def gaussian_pdf(x, mu=0, sigma=1):
    """Gaussian PDF: (1/σ√2π) * exp(-0.5*((x-μ)/σ)²)"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Plot for different parameters
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 1000)
for sigma in [0.5, 1, 2]:
    plt.plot(x, gaussian_pdf(x, mu=0, sigma=sigma), label=f'σ={sigma}')
plt.legend()
plt.show()
```

## Naive Bayes Classifier

```python
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(X)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c]))
                likelihood -= 0.5 * np.sum((x - self.mean[c])**2 / self.var[c])
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```

## Information Theory

```python
def entropy(p):
    """H(X) = -Σ p(x) log p(x)"""
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

def cross_entropy(p, q):
    """H(p, q) = -Σ p(x) log q(x)"""
    return -np.sum(p * np.log(q + 1e-9))

def kl_divergence(p, q):
    """D_KL(p||q) = Σ p(x) log(p(x)/q(x))"""
    return np.sum(p * np.log(p / (q + 1e-9) + 1e-9))

# Example
p = np.array([0.5, 0.5])
q = np.array([0.4, 0.6])
print(f"H(p) = {entropy(p):.4f}")
print(f"H(p, q) = {cross_entropy(p, q):.4f}")
print(f"D_KL(p||q) = {kl_divergence(p, q):.4f}")
```

## Spam Classifier Project

Compare MLE vs MAP estimation on spam dataset

**Data:** SMS Spam Collection
**Target:** >90% accuracy

## Resources
- [Murphy's PML Book Ch 2-4](https://probml.github.io/pml-book/)
- [MML Book Ch 6](https://mml-book.github.io/)
- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
