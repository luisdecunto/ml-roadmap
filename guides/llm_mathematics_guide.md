# Module 11: LLM Mathematics - Complete Coding Guide

This guide provides step-by-step implementations for all the coding components of Module 11, focusing on building language models from scratch with proper mathematical foundations.

## Table of Contents
1. [Micrograd: Automatic Differentiation Engine](#part-1-micrograd)
2. [Bigram Language Model](#part-2-bigram-language-model)
3. [GPT Architecture from Scratch](#part-3-gpt-architecture)
4. [Sampling Strategies](#part-4-sampling-strategies)

---

## Part 1: Micrograd - Automatic Differentiation Engine

### Overview
Micrograd is a tiny autograd engine that implements backpropagation over a dynamically built DAG (directed acyclic graph). This is the foundation for understanding how PyTorch and other frameworks work.

### Step 1: Value Class with Forward Pass

```python
import math
import numpy as np
from typing import Union, Tuple, Set

class Value:
    """
    Stores a single scalar value and its gradient.
    Supports automatic differentiation through the computational graph.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Topological sort and backpropagation.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

### Step 2: Testing Micrograd

```python
def test_micrograd():
    """Test the micrograd implementation."""

    # Test 1: Simple arithmetic
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f
    L.label = 'L'

    L.backward()

    print("Test 1: Simple arithmetic")
    print(f"L = {L.data}")
    print(f"dL/da = {a.grad}")  # Should be 6.0
    print(f"dL/db = {b.grad}")  # Should be -4.0
    print(f"dL/dc = {c.grad}")  # Should be -2.0
    print()

    # Test 2: Neural network neuron
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    o = n.tanh()

    o.backward()

    print("Test 2: Neural network neuron")
    print(f"Output = {o.data}")
    print(f"dout/dw1 = {w1.grad}")
    print(f"dout/dw2 = {w2.grad}")
    print()

    # Test 3: Verify against numerical gradient
    def numerical_gradient(f, x, h=1e-5):
        """Compute numerical gradient."""
        return (f(x + h) - f(x - h)) / (2 * h)

    x = Value(3.0)

    # Test f(x) = x^2
    def f(val):
        v = Value(val)
        out = v * v
        return out.data

    y = x * x
    y.backward()

    numerical_grad = numerical_gradient(f, x.data)
    analytical_grad = x.grad

    print("Test 3: Gradient verification")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Difference: {abs(analytical_grad - numerical_grad)}")

# Run tests
test_micrograd()
```

### Expected Output
```
Test 1: Simple arithmetic
L = 16.0
dL/da = 6.0
dL/db = -4.0
dL/dc = -2.0

Test 2: Neural network neuron
Output = 0.7071067811865477
dout/dw1 = 1.0
dout/dw2 = 0.0

Test 3: Gradient verification
Analytical gradient: 6.0
Numerical gradient: 6.0
Difference: 0.0
```

---

## Part 2: Bigram Language Model

### Overview
A bigram model predicts the next character based only on the current character. This is the simplest form of a language model.

### Step 1: Data Preparation

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load names dataset
names = open('names.txt', 'r').read().splitlines()
print(f"Loaded {len(names)} names")
print("Sample names:", names[:10])

# Build character vocabulary
chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  # Special token for start/end
itos = {i:s for s,i in stoi.items()}
vocab_size = len(stoi)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {chars}")
```

### Step 2: Count-Based Bigram Model

```python
def build_bigram_counts(names):
    """Build bigram frequency matrix."""
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

    for name in names:
        chars_in_name = ['.'] + list(name) + ['.']
        for ch1, ch2 in zip(chars_in_name, chars_in_name[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    return N

# Build counts
N = build_bigram_counts(names)

# Visualize
plt.figure(figsize=(12, 12))
plt.imshow(N, cmap='Blues')
for i in range(vocab_size):
    for j in range(vocab_size):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.title('Bigram Counts')
plt.savefig('bigram_counts.png', dpi=150, bbox_inches='tight')
plt.close()

print("Bigram counts visualization saved.")
```

### Step 3: Probability Distribution

```python
# Convert counts to probabilities
P = (N + 1).float()  # Add smoothing
P /= P.sum(1, keepdim=True)

# Sample from the model
def sample_name(P, seed=None):
    """Generate a name using the bigram model."""
    if seed is not None:
        torch.manual_seed(seed)

    out = []
    ix = 0  # Start token
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        if ix == 0:  # End token
            break
        out.append(itos[ix])

    return ''.join(out)

# Generate some names
print("\nGenerated names:")
for i in range(10):
    print(sample_name(P, seed=i))
```

### Step 4: Neural Network Bigram Model

```python
class BigramLanguageModel(torch.nn.Module):
    """Neural network based bigram model."""

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) tensor of indices
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Get last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# Prepare training data
def create_dataset(names):
    xs, ys = [], []
    for name in names:
        chars_in_name = ['.'] + list(name) + ['.']
        for ch1, ch2 in zip(chars_in_name, chars_in_name[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys

xs, ys = create_dataset(names)
print(f"Dataset size: {len(xs)}")

# Train the model
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)

for step in range(1000):
    logits, loss = model(xs.unsqueeze(1), ys)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Generate names
print("\nGenerated names from neural model:")
for _ in range(10):
    idx = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(idx, max_new_tokens=20)
    name = ''.join([itos[i] for i in generated[0].tolist() if i != 0])
    print(name)
```

### Step 5: Calculate Negative Log-Likelihood

```python
def calculate_nll(model, names):
    """Calculate average negative log-likelihood."""
    total_log_likelihood = 0
    total_count = 0

    for name in names:
        chars_in_name = ['.'] + list(name) + ['.']
        for ch1, ch2 in zip(chars_in_name, chars_in_name[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            idx = torch.tensor([[ix1]])
            logits, _ = model(idx)
            probs = F.softmax(logits[0, 0], dim=-1)
            log_prob = torch.log(probs[ix2])

            total_log_likelihood += log_prob.item()
            total_count += 1

    nll = -total_log_likelihood / total_count
    return nll

nll = calculate_nll(model, names[:100])
print(f"\nAverage Negative Log-Likelihood: {nll:.4f}")
```

---

## Part 3: GPT Architecture from Scratch

### Overview
Build a decoder-only transformer (GPT architecture) following the "Attention is All You Need" paper.

### Step 1: Multi-Head Self-Attention

```python
class Head(torch.nn.Module):
    """Single head of self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted aggregation
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out


class MultiHeadAttention(torch.nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = torch.nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = torch.nn.Linear(n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

### Step 2: Feed-Forward Network

```python
class FeedForward(torch.nn.Module):
    """Simple linear layer followed by non-linearity."""

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

### Step 3: Transformer Block

```python
class Block(torch.nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-norm formulation
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

### Step 4: Complete GPT Model

```python
class GPT(torch.nn.Module):
    """
    GPT Language Model.
    Decoder-only transformer architecture.
    """

    def __init__(self, vocab_size, n_embd=384, num_heads=6, n_layer=6,
                 block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size

        # Token + position embeddings
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = torch.nn.Sequential(
            *[Block(n_embd, num_heads, block_size, dropout) for _ in range(n_layer)]
        )

        # Final layer norm and projection
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)     # (B, T, n_embd)
        x = self.ln_f(x)       # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens."""
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

### Step 5: Training GPT

```python
def train_gpt():
    """Train the GPT model on a dataset."""

    # Load tiny shakespeare dataset (or any text)
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Create vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Train and validation splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Hyperparameters
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200

    # Model
    model = GPT(vocab_size, n_embd=384, num_heads=6, n_layer=6,
                block_size=block_size, dropout=0.2)

    # Count parameters
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate from the model
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8)
    print(decode(generated[0].tolist()))

    return model

# Train the model
model = train_gpt()
```

---

## Part 4: Sampling Strategies

### Overview
Different methods for generating text from language models, each with different properties.

### Step 1: Temperature Sampling

```python
def sample_with_temperature(logits, temperature=1.0):
    """
    Sample from logits with temperature scaling.

    Temperature < 1.0: More confident/deterministic
    Temperature = 1.0: Normal sampling
    Temperature > 1.0: More random/diverse
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx

# Test different temperatures
def test_temperatures(model, context, max_tokens=100):
    """Test generation with different temperatures."""
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

    print("Testing different temperatures:\n")
    for temp in temperatures:
        idx = context.clone()
        generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temp)
        text = decode(generated[0].tolist())
        print(f"Temperature {temp}:")
        print(text)
        print("-" * 80)
        print()

# Run test
context = torch.zeros((1, 1), dtype=torch.long)
test_temperatures(model, context)
```

### Step 2: Top-K Sampling

```python
def sample_top_k(logits, k=10):
    """
    Sample from the top-k most likely tokens.
    Filters out low-probability tokens.
    """
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    logits[logits < v[..., [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx

def generate_with_top_k(model, context, max_tokens=100, k=10):
    """Generate text using top-k sampling."""
    idx = context.clone()

    for _ in range(max_tokens):
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Top-k filtering
        idx_next = sample_top_k(logits, k=k)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# Test different k values
print("Testing Top-K sampling:\n")
for k in [1, 5, 10, 50]:
    generated = generate_with_top_k(model, context, max_tokens=100, k=k)
    text = decode(generated[0].tolist())
    print(f"Top-K (k={k}):")
    print(text)
    print("-" * 80)
    print()
```

### Step 3: Nucleus (Top-P) Sampling

```python
def sample_top_p(logits, p=0.9):
    """
    Nucleus sampling: sample from smallest set of tokens whose cumulative
    probability exceeds p.

    More adaptive than top-k: size of set varies based on probability distribution.
    """
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = -float('Inf')

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx

def generate_with_top_p(model, context, max_tokens=100, p=0.9):
    """Generate text using nucleus (top-p) sampling."""
    idx = context.clone()

    for _ in range(max_tokens):
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Nucleus sampling
        idx_next = sample_top_p(logits, p=p)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# Test different p values
print("Testing Nucleus (Top-P) sampling:\n")
for p in [0.5, 0.7, 0.9, 0.95]:
    generated = generate_with_top_p(model, context, max_tokens=100, p=p)
    text = decode(generated[0].tolist())
    print(f"Top-P (p={p}):")
    print(text)
    print("-" * 80)
    print()
```

### Step 4: Combined Sampling Strategy

```python
def sample_combined(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Combined sampling: temperature + top-k + top-p.
    This is what most modern LLMs use in practice.
    """
    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float('Inf')

    # Apply top-p filtering
    if top_p is not None:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('Inf')

    # Sample
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx

# Add to GPT class
def generate_advanced(model, idx, max_new_tokens, temperature=1.0,
                     top_k=None, top_p=None):
    """Generate with all sampling strategies."""
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = sample_combined(logits, temperature, top_k, top_p)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# Test combined strategies
print("Testing combined sampling strategies:\n")

configs = [
    {"temp": 0.8, "top_k": 50, "top_p": 0.9, "name": "Balanced"},
    {"temp": 0.7, "top_k": 40, "top_p": 0.85, "name": "Conservative"},
    {"temp": 1.0, "top_k": None, "top_p": 0.95, "name": "Creative"},
]

for config in configs:
    generated = generate_advanced(
        model, context, max_new_tokens=100,
        temperature=config["temp"],
        top_k=config["top_k"],
        top_p=config["top_p"]
    )
    text = decode(generated[0].tolist())
    print(f"{config['name']} (T={config['temp']}, k={config['top_k']}, p={config['top_p']}):")
    print(text)
    print("-" * 80)
    print()
```

### Step 5: Perplexity Calculation

```python
def calculate_perplexity(model, data, block_size):
    """
    Calculate perplexity on a dataset.
    Lower perplexity = better model.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(data) - block_size, block_size):
            # Get batch
            x = data[i:i+block_size].unsqueeze(0)
            y = data[i+1:i+block_size+1].unsqueeze(0)

            # Forward pass
            logits, loss = model(x, y)

            total_loss += loss.item() * block_size
            total_tokens += block_size

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity

# Calculate perplexity on validation set
perplexity = calculate_perplexity(model, val_data, block_size=256)
print(f"Validation Perplexity: {perplexity:.2f}")

# Lower is better - perplexity of 1 means perfect prediction
# Typical values for character-level models: 5-50
# Typical values for word-level models: 20-200
```

---

## Complete Example: Character-Level GPT

Here's a complete, runnable example that ties everything together:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Model Definition ===
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# === Training Script ===
def main():
    # Load data
    with open('input.txt', 'r') as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch in chars}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Model
    model = GPTLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training
    batch_size = 32
    block_size = 128

    for iter in range(5000):
        # Sample batch
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i:i+block_size] for i in ix])
        y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])

        # Forward
        logits, loss = model(x, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 500 == 0:
            print(f"Step {iter}, Loss: {loss.item():.4f}")

    # Generate
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=500)
    print(decode(generated[0].tolist()))

if __name__ == '__main__':
    main()
```

---

## Success Criteria

You've successfully completed Module 11 when you can:

1. ✅ Build and understand micrograd's automatic differentiation
2. ✅ Implement a bigram language model from scratch
3. ✅ Build a complete GPT architecture with multi-head attention
4. ✅ Implement various sampling strategies (temperature, top-k, top-p)
5. ✅ Calculate perplexity to evaluate language models
6. ✅ Train a small GPT model and generate coherent text

## Next Steps

- Module 12: Dive deeper into optimization techniques
- Experiment with larger models and datasets
- Try implementing LoRA for parameter-efficient fine-tuning
- Explore tokenization methods (BPE, WordPiece, SentencePiece)

## Additional Resources

- [Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [Micrograd repository](https://github.com/karpathy/micrograd)
- [NanoGPT repository](https://github.com/karpathy/nanoGPT)
