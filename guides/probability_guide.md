# Probability & Statistics - Coding Guide

**Time:** 8-10 hours
**Difficulty:** Intermediate
**Prerequisites:** Python, NumPy, basic probability theory

## What You'll Build

Implement probability and statistical methods from scratch:
1. Probability distributions (Gaussian, Bernoulli, Binomial)
2. Maximum Likelihood Estimation (MLE)
3. Naive Bayes classifier from scratch
4. **Project:** Spam classifier with uncertainty quantification

---

## Project Setup

```bash
mkdir probability-from-scratch
cd probability-from-scratch

# Create files
touch distributions.py
touch naive_bayes.py
touch spam_classifier.py
touch test_all.py
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

## Part 1: Probability Distributions

### Theory

**Gaussian (Normal) Distribution:**
```
p(x | μ, σ²) = (1 / √(2πσ²)) * exp(-(x-μ)² / (2σ²))
```

**Bernoulli Distribution:**
```
p(x | θ) = θ^x * (1-θ)^(1-x)  for x ∈ {0,1}
```

**Binomial Distribution:**
```
p(k | n, θ) = C(n,k) * θ^k * (1-θ)^(n-k)
```

### Implementation

```python
# distributions.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

class GaussianDistribution:
    """Univariate Gaussian distribution"""

    def __init__(self, mu=0.0, sigma=1.0):
        """
        Args:
            mu: mean
            sigma: standard deviation
        """
        self.mu = mu
        self.sigma = sigma
        self.variance = sigma ** 2

    def pdf(self, x):
        """
        Probability density function

        Args:
            x: point(s) to evaluate (scalar or array)
        Returns:
            p(x): probability density
        """
        coefficient = 1.0 / (self.sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - self.mu) / self.sigma) ** 2
        return coefficient * np.exp(exponent)

    def log_pdf(self, x):
        """Log probability density (for numerical stability)"""
        return -0.5 * np.log(2 * np.pi * self.variance) - \
               0.5 * ((x - self.mu) ** 2) / self.variance

    def sample(self, size=1):
        """Generate random samples"""
        return np.random.normal(self.mu, self.sigma, size)

    def fit(self, data):
        """
        Fit distribution parameters using Maximum Likelihood Estimation

        Args:
            data: array of observations
        """
        self.mu = np.mean(data)
        self.sigma = np.std(data, ddof=1)  # Use sample std
        self.variance = self.sigma ** 2
        return self


class BernoulliDistribution:
    """Bernoulli distribution for binary outcomes"""

    def __init__(self, theta=0.5):
        """
        Args:
            theta: probability of success (x=1)
        """
        self.theta = theta

    def pmf(self, x):
        """
        Probability mass function

        Args:
            x: binary outcome(s) {0, 1}
        Returns:
            p(x): probability
        """
        return np.where(x == 1, self.theta, 1 - self.theta)

    def log_pmf(self, x):
        """Log probability"""
        return x * np.log(self.theta) + (1 - x) * np.log(1 - self.theta)

    def sample(self, size=1):
        """Generate random samples"""
        return (np.random.rand(size) < self.theta).astype(int)

    def fit(self, data):
        """MLE: θ = sum(x_i) / n"""
        self.theta = np.mean(data)
        return self


class BinomialDistribution:
    """Binomial distribution for n trials"""

    def __init__(self, n=10, theta=0.5):
        """
        Args:
            n: number of trials
            theta: probability of success per trial
        """
        self.n = n
        self.theta = theta

    def pmf(self, k):
        """
        Probability mass function

        Args:
            k: number of successes
        Returns:
            p(k): probability of k successes in n trials
        """
        return comb(self.n, k, exact=True) * \
               (self.theta ** k) * ((1 - self.theta) ** (self.n - k))

    def sample(self, size=1):
        """Generate random samples"""
        return np.random.binomial(self.n, self.theta, size)

    def fit(self, data):
        """MLE for θ given fixed n"""
        self.theta = np.mean(data) / self.n
        return self


# Test distributions
if __name__ == "__main__":
    # Test Gaussian
    gaussian = GaussianDistribution(mu=0, sigma=1)
    x = np.linspace(-5, 5, 1000)

    plt.figure(figsize=(15, 5))

    # Plot Gaussian PDFs with different parameters
    plt.subplot(1, 3, 1)
    for sigma in [0.5, 1.0, 2.0]:
        gauss = GaussianDistribution(mu=0, sigma=sigma)
        plt.plot(x, gauss.pdf(x), label=f'σ={sigma}')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Gaussian Distribution (varying σ)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Test MLE
    true_gaussian = GaussianDistribution(mu=2.0, sigma=1.5)
    samples = true_gaussian.sample(1000)

    fitted_gaussian = GaussianDistribution().fit(samples)
    print(f"True: μ={true_gaussian.mu:.2f}, σ={true_gaussian.sigma:.2f}")
    print(f"MLE:  μ={fitted_gaussian.mu:.2f}, σ={fitted_gaussian.sigma:.2f}")

    # Plot Bernoulli PMF
    plt.subplot(1, 3, 2)
    for theta in [0.3, 0.5, 0.7]:
        bern = BernoulliDistribution(theta=theta)
        x_vals = [0, 1]
        y_vals = [bern.pmf(x) for x in x_vals]
        plt.bar(x_vals, y_vals, alpha=0.5, label=f'θ={theta}', width=0.2)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Bernoulli Distribution')
    plt.legend()
    plt.xticks([0, 1])

    # Plot Binomial PMF
    plt.subplot(1, 3, 3)
    binom = BinomialDistribution(n=20, theta=0.5)
    k_vals = np.arange(21)
    pmf_vals = [binom.pmf(k) for k in k_vals]
    plt.bar(k_vals, pmf_vals, alpha=0.7)
    plt.xlabel('k (successes)')
    plt.ylabel('p(k)')
    plt.title('Binomial Distribution (n=20, θ=0.5)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('distributions.png', dpi=150)
    plt.show()
```

---

## Part 2: Naive Bayes Classifier

### Theory

**Bayes' Theorem:**
```
P(y|x) = P(x|y) * P(y) / P(x)
```

**Naive Bayes assumption:** Features are independent given class:
```
P(x₁,...,xₙ | y) = ∏ P(xᵢ | y)
```

**Classification:** Choose class with maximum posterior:
```
ŷ = argmax_y P(y) * ∏ P(xᵢ | y)
```

### Implementation

```python
# naive_bayes.py
import numpy as np
from distributions import GaussianDistribution

class GaussianNaiveBayes:
    """
    Naive Bayes classifier with Gaussian likelihood.
    Assumes continuous features follow Gaussian distribution.
    """

    def __init__(self):
        self.classes = None
        self.priors = {}       # P(y)
        self.distributions = {}  # P(x|y) for each feature
        self.n_features = None

    def fit(self, X, y):
        """
        Fit Naive Bayes model

        Args:
            X: feature matrix (n_samples, n_features)
            y: labels (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        self.classes = np.unique(y)
        self.n_features = X.shape[1]

        # Learn parameters for each class
        for c in self.classes:
            # Get samples for this class
            X_c = X[y == c]

            # Prior: P(y=c) = count(c) / n_total
            self.priors[c] = len(X_c) / len(X)

            # Likelihood: P(x_i | y=c) for each feature
            self.distributions[c] = []
            for feature_idx in range(self.n_features):
                feature_data = X_c[:, feature_idx]
                # Fit Gaussian to this feature
                gauss = GaussianDistribution().fit(feature_data)
                self.distributions[c].append(gauss)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: feature matrix (n_samples, n_features)
        Returns:
            probabilities: (n_samples, n_classes)
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Store log probabilities (for numerical stability)
        log_probs = np.zeros((n_samples, n_classes))

        for class_idx, c in enumerate(self.classes):
            # Start with log prior
            log_prior = np.log(self.priors[c])

            # Add log likelihood for each feature
            log_likelihood = 0
            for feature_idx in range(self.n_features):
                feature_vals = X[:, feature_idx]
                gauss = self.distributions[c][feature_idx]
                log_likelihood += gauss.log_pdf(feature_vals)

            log_probs[:, class_idx] = log_prior + log_likelihood

        # Convert log probabilities to probabilities
        # Use log-sum-exp trick for numerical stability
        max_log_prob = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_log_prob)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Test on Iris dataset
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)

    # Predict
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Naive Bayes Confusion Matrix (Accuracy: {accuracy:.2%})')
    plt.tight_layout()
    plt.savefig('naive_bayes_confusion_matrix.png', dpi=150)
    plt.show()

    # Compare with sklearn
    from sklearn.naive_bayes import GaussianNB
    sklearn_nb = GaussianNB()
    sklearn_nb.fit(X_train, y_train)
    sklearn_acc = sklearn_nb.score(X_test, y_test)

    print(f"\nComparison:")
    print(f"Our implementation: {accuracy:.4f}")
    print(f"Sklearn:           {sklearn_acc:.4f}")
```

---

## Part 3: Spam Classifier Project

### Complete Implementation

```python
# spam_classifier.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from naive_bayes import GaussianNaiveBayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def entropy(p):
    """Calculate Shannon entropy of probability distribution"""
    p = np.array(p)
    p = p[p > 0]  # Remove zeros
    return -np.sum(p * np.log2(p))

class SpamClassifier:
    """
    Spam classifier using Naive Bayes with text features.
    Includes uncertainty quantification.
    """

    def __init__(self, vectorizer_type='tfidf', max_features=100):
        """
        Args:
            vectorizer_type: 'count' or 'tfidf'
            max_features: maximum number of features to extract
        """
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features,
                                             stop_words='english')
        else:
            self.vectorizer = CountVectorizer(max_features=max_features,
                                             stop_words='english')

        self.classifier = GaussianNaiveBayes()
        self.is_fitted = False

    def fit(self, texts, labels):
        """
        Train spam classifier

        Args:
            texts: list of text messages
            labels: binary labels (0=ham, 1=spam)
        """
        # Extract features
        X = self.vectorizer.fit_transform(texts).toarray()

        # Train classifier
        self.classifier.fit(X, labels)
        self.is_fitted = True

        return self

    def predict(self, texts):
        """Predict spam/ham labels"""
        X = self.vectorizer.transform(texts).toarray()
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """Predict probabilities with uncertainty"""
        X = self.vectorizer.transform(texts).toarray()
        return self.classifier.predict_proba(X)

    def evaluate(self, texts, labels):
        """Comprehensive evaluation"""
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Calculate uncertainty metrics
        entropies = [entropy(prob) for prob in probabilities]
        avg_uncertainty = np.mean(entropies)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_uncertainty': avg_uncertainty,
            'predictions': predictions,
            'probabilities': probabilities
        }

        return results


# Main execution
if __name__ == "__main__":
    # Load SMS Spam Collection dataset
    # Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

    # For demonstration, create synthetic data
    # In practice, load real dataset
    print("Loading data...")

    # Example: synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Simple spam vs ham patterns
    spam_words = ['free', 'win', 'prize', 'cash', 'urgent', 'click', 'buy']
    ham_words = ['hello', 'meeting', 'tomorrow', 'thanks', 'please', 'regards']

    texts = []
    labels = []

    for i in range(n_samples):
        if np.random.rand() < 0.3:  # 30% spam
            # Generate spam message
            words = np.random.choice(spam_words, size=np.random.randint(3, 6))
            texts.append(' '.join(words))
            labels.append(1)
        else:
            # Generate ham message
            words = np.random.choice(ham_words, size=np.random.randint(3, 6))
            texts.append(' '.join(words))
            labels.append(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Spam ratio: {np.mean(labels):.2%}")

    # Train classifier
    print("\nTraining spam classifier...")
    spam_clf = SpamClassifier(vectorizer_type='tfidf', max_features=50)
    spam_clf.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating...")
    results = spam_clf.evaluate(X_test, y_test)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"Avg Uncertainty: {results['avg_uncertainty']:.4f} bits")

    # Analyze predictions
    probabilities = results['probabilities']
    predictions = results['predictions']

    # Find confident vs uncertain predictions
    confidences = np.max(probabilities, axis=1)
    uncertain_indices = np.where(confidences < 0.7)[0]

    print(f"\nUncertain predictions: {len(uncertain_indices)}")
    print("Examples:")
    for idx in uncertain_indices[:5]:
        text = X_test[idx]
        true_label = y_test[idx]
        pred_label = predictions[idx]
        prob = probabilities[idx]
        print(f"  Text: '{text}'")
        print(f"  True: {true_label}, Pred: {pred_label}")
        print(f"  P(ham)={prob[0]:.3f}, P(spam)={prob[1]:.3f}")
        print()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Confidence distribution
    ax1 = axes[0, 0]
    ax1.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.axvline(0.7, color='r', linestyle='--', label='Uncertainty threshold')
    ax1.legend()

    # Plot 2: Spam probability distribution
    ax2 = axes[0, 1]
    spam_probs = probabilities[:, 1]
    ax2.hist(spam_probs[y_test == 0], bins=20, alpha=0.5, label='Ham (true)', color='blue')
    ax2.hist(spam_probs[y_test == 1], bins=20, alpha=0.5, label='Spam (true)', color='red')
    ax2.set_xlabel('P(spam)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Spam Probability by True Class')
    ax2.legend()

    # Plot 3: Confusion matrix
    from sklearn.metrics import confusion_matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, predictions)
    im = ax3.imshow(cm, cmap='Blues', aspect='auto')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Ham', 'Spam'])
    ax3.set_yticklabels(['Ham', 'Spam'])

    for i in range(2):
        for j in range(2):
            ax3.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=20)

    # Plot 4: Feature importance (top words)
    ax4 = axes[1, 1]
    feature_names = spam_clf.vectorizer.get_feature_names_out()

    # Get feature importance from classifier
    spam_class_idx = 1
    feature_scores = []
    for i, feature_name in enumerate(feature_names):
        # Calculate difference in means between spam and ham
        spam_mean = spam_clf.classifier.distributions[spam_class_idx][i].mu
        ham_mean = spam_clf.classifier.distributions[0][i].mu
        feature_scores.append(abs(spam_mean - ham_mean))

    top_n = 10
    top_indices = np.argsort(feature_scores)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [feature_scores[i] for i in top_indices]

    ax4.barh(range(top_n), top_scores)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(top_features)
    ax4.set_xlabel('Importance Score')
    ax4.set_title(f'Top {top_n} Discriminative Features')

    plt.tight_layout()
    plt.savefig('spam_classifier_results.png', dpi=150)
    plt.show()
```

---

## Testing & Validation

```python
# test_all.py
import numpy as np
from distributions import GaussianDistribution, BernoulliDistribution
from naive_bayes import GaussianNaiveBayes

def test_gaussian_distribution():
    """Test Gaussian distribution implementation"""
    print("Testing Gaussian Distribution...")

    # Test PDF
    gauss = GaussianDistribution(mu=0, sigma=1)
    assert abs(gauss.pdf(0) - 0.3989) < 0.001, "PDF at mean failed"

    # Test MLE
    true_gauss = GaussianDistribution(mu=5, sigma=2)
    samples = true_gauss.sample(10000)
    fitted = GaussianDistribution().fit(samples)

    assert abs(fitted.mu - 5) < 0.1, f"MLE mu failed: {fitted.mu}"
    assert abs(fitted.sigma - 2) < 0.1, f"MLE sigma failed: {fitted.sigma}"

    print("✓ Gaussian distribution tests passed")


def test_naive_bayes():
    """Test Naive Bayes classifier"""
    print("\nTesting Naive Bayes...")

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    accuracy = nb.score(X_test, y_test)

    assert accuracy > 0.9, f"Accuracy too low: {accuracy}"

    print(f"✓ Naive Bayes accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    print("="*60)
    print("RUNNING ALL TESTS")
    print("="*60)

    test_gaussian_distribution()
    test_naive_bayes()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
```

---

## Checklist

Complete these tasks in order:

- [ ] **Part 1:** Implement Gaussian, Bernoulli, Binomial distributions
- [ ] Test MLE fitting on synthetic data
- [ ] Visualize PDFs/PMFs for different parameters
- [ ] **Part 2:** Implement Gaussian Naive Bayes from scratch
- [ ] Test on Iris dataset (target: >90% accuracy)
- [ ] Compare with sklearn implementation
- [ ] **Part 3:** Build spam classifier with uncertainty quantification
- [ ] Achieve >90% accuracy on SMS Spam dataset
- [ ] Analyze confident vs uncertain predictions
- [ ] Visualize results (confusion matrix, feature importance)
- [ ] Run all tests in `test_all.py`

---

## Resources

- **Books:**
  - [Murphy's Probabilistic Machine Learning](https://probml.github.io/pml-book/) - Ch 2-4
  - [MML Book Chapter 6: Probability](https://mml-book.github.io/)
  - [Pattern Recognition and Machine Learning (Bishop)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

- **Online:**
  - [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)
  - [Khan Academy: Probability & Statistics](https://www.khanacademy.org/math/statistics-probability)
  - [3Blue1Brown: Bayes' Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)

- **Datasets:**
  - [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
  - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

---

## What's Next?

After completing this guide:
1. Move to **Module 5: Information Theory** (entropy, cross-entropy, KL divergence)
2. Try the **Mini-Project:** Build production-grade probabilistic spam detector
3. Study **Bayesian Networks** and graphical models
4. Learn **Variational Inference** and MCMC methods
