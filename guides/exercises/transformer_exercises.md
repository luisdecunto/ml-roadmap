# Transformer Exercises - Modules 9-10

**Time:** 4-5 hours
**Difficulty:** Advanced
**Materials needed:** Paper, pencil, calculator, NumPy

Implement Transformers from scratch. Solutions are in `guides/solutions/transformer_solutions.md`

---

## Part 1: Attention Mechanism (35 min)

### Exercise 1.1: Attention Basics
Query q = [1, 0, 1], Keys K = [[1, 1, 0], [0, 1, 1], [1, 0, 0]], Values V = [[2, 0], [0, 2], [1, 1]]

1. Compute attention scores: s_i = q · k_i for each key
2. Apply softmax: α = softmax(s)
3. Compute weighted sum: output = Σ α_i · v_i
4. What does high attention weight mean?

### Exercise 1.2: Attention as Soft Lookup
Database with 3 key-value pairs:

Keys: ["cat", "dog", "bird"] (encoded as vectors)
Values: [definition of cat, dog, bird]

Query: "feline" (similar to "cat")

1. Calculate similarities (dot products)
2. Softmax to get weights
3. Retrieve weighted combination of values
4. Why "soft" vs hard lookup?

### Exercise 1.3: Self-Attention
Sentence: "The cat sat"
Word embeddings (simplified):
- "The" = [1, 0]
- "cat" = [0, 1]
- "sat" = [1, 1]

1. Each word is Q, K, V
2. Calculate attention: "cat" attending to all words
3. Which words does "cat" attend to most?
4. How is this different from fixed window?

### Exercise 1.4: Attention Visualization
Sentence: "I love machine learning"

1. Compute 4×4 attention matrix (self-attention)
2. Which words attend to which?
3. Why do words often attend to themselves?
4. Draw attention heatmap

---

## Part 2: Scaled Dot-Product Attention (30 min)

### Exercise 2.1: Scaling Factor
Q, K dimensions: d_k = 64

1. Without scaling: scores can be very large
2. Why? Dot product grows with dimension
3. Compute score for random q, k: dot(q, k)
4. Scaled: dot(q, k) / √d_k - why does this help?

### Exercise 2.2: Implement Scaled Attention
Q shape: (batch=2, seq_len=3, d_k=4)
K shape: (2, 3, 4)
V shape: (2, 3, 8)

1. Compute scores: QK^T (shape: 2×3×3)
2. Scale by 1/√4 = 0.5
3. Apply softmax over last dimension
4. Multiply by V: output shape?

### Exercise 2.3: Attention Masking
Decoder self-attention: prevent looking at future tokens

Sequence length: 4

1. Create causal mask: upper triangular of -inf
2. Add to attention scores before softmax
3. After softmax: future positions have weight 0
4. Why is this necessary for language modeling?

---

## Part 3: Multi-Head Attention (40 min)

### Exercise 3.1: Single Head vs Multi-Head
Single head: d_model = 512, 1 attention mechanism

Multi-head: d_model = 512, h = 8 heads

1. Each head: d_k = d_v = 512/8 = 64
2. Why split into multiple heads?
3. Different heads learn different patterns
4. Concatenate outputs: 8 × 64 = 512

### Exercise 3.2: Implement Multi-Head Attention
Input: X shape (batch=1, seq=4, d_model=8)
Heads: h = 2
d_k = d_v = 4 per head

1. Linear projections: W_Q, W_K, W_V for each head
2. Reshape: (1, 4, 8) → (1, 2, 4, 4) [batch, heads, seq, d_k]
3. Apply scaled attention per head
4. Concatenate: (1, 2, 4, 4) → (1, 4, 8)
5. Final linear projection W_O

### Exercise 3.3: Parameter Count
Transformer block: d_model = 512, h = 8

Multi-head attention:
1. W_Q, W_K, W_V: each 512 × 512
2. W_O: 512 × 512
3. Total MHA parameters?
4. Compare with standard attention (no multi-head)

### Exercise 3.4: Cross-Attention vs Self-Attention
Encoder-Decoder attention:

1. Q from decoder
2. K, V from encoder
3. Decoder attends to all encoder outputs
4. Use case: machine translation

---

## Part 4: Positional Encoding (30 min)

### Exercise 4.1: Why Positional Encoding?
Without position info:

Sentence 1: "cat eats fish"
Sentence 2: "fish eats cat"

1. Same words, different meaning
2. Attention is permutation invariant
3. Need to inject position information
4. Add positional encoding to embeddings

### Exercise 4.2: Sinusoidal Positional Encoding
Formula:
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

For pos = 0, 1, 2 and d_model = 4:

1. Calculate PE for position 0
2. Calculate PE for position 1
3. Calculate PE for position 2
4. Why sine/cosine? (unique, extrapolates to longer sequences)

### Exercise 4.3: Learned vs Fixed Positional Encoding
Compare two approaches:

1. Fixed: sine/cosine (no parameters)
2. Learned: embedding layer for positions
3. Pros/cons of each?
4. Which does original Transformer use?

---

## Part 5: Transformer Blocks (40 min)

### Exercise 5.1: Encoder Block
Components:
1. Multi-Head Self-Attention
2. Add & Norm (residual connection + layer norm)
3. Feed-Forward Network (2 linear layers with ReLU)
4. Add & Norm

Input: X shape (batch, seq, d_model)

Trace through:
1. MHA output
2. Add residual: X + MHA(X)
3. Layer norm
4. FFN
5. Add & Norm again
6. Final output shape?

### Exercise 5.2: Layer Normalization
Input: X shape (2, 3, 4) [batch, seq, features]

1. Compute mean and variance across features (last dim)
2. Normalize: (X - mean) / √(var + ε)
3. Scale and shift: γ * X_norm + β
4. Why layer norm instead of batch norm in Transformers?

### Exercise 5.3: Feed-Forward Network
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

d_model = 512, d_ff = 2048

1. First linear: 512 → 2048
2. ReLU activation
3. Second linear: 2048 → 512
4. Parameters: (512×2048) + (2048×512) = ?

---

## Part 6: Full Transformer (45 min)

### Exercise 6.1: Encoder Stack
6 encoder layers, d_model = 512, h = 8

Input: sentence "Hello world"
1. Token embedding + positional encoding
2. Pass through encoder layer 1
3. Output goes to encoder layer 2
4. ... through layer 6
5. Final encoder output shape?

### Exercise 6.2: Decoder Stack
6 decoder layers:

1. Masked self-attention (causal)
2. Add & Norm
3. Cross-attention (attend to encoder output)
4. Add & Norm
5. FFN
6. Add & Norm

Why masked self-attention in decoder?

### Exercise 6.3: Full Transformer Forward Pass
Translation: "Hello" → "Bonjour"

Encoder:
1. Embed "Hello" + positional encoding
2. 6 encoder layers
3. Output: encoder_output

Decoder:
1. Embed "<BOS> Bonjour" (teacher forcing)
2. Masked self-attention
3. Cross-attention with encoder_output
4. 6 decoder layers
5. Linear + softmax → next token probabilities

### Exercise 6.4: Parameter Count
Full Transformer: L=6 layers, d_model=512, h=8, d_ff=2048

Per encoder layer:
1. MHA: 4 × 512×512
2. FFN: 512×2048 + 2048×512
3. Layer norms (small)

Total encoder: 6 × (MHA + FFN)
Total decoder: similar + cross-attention
Calculate total parameters

---

## Challenge Problems (Optional)

### Challenge 1: Implement Transformer from Scratch
NumPy only:

1. Multi-head attention
2. Positional encoding
3. Encoder and decoder blocks
4. Full transformer
5. Train on tiny sequence-to-sequence task

### Challenge 2: Attention Patterns
Analyze attention in trained model:

1. Extract attention weights from each layer
2. Visualize attention patterns
3. Do different heads learn different patterns?
4. Do later layers attend differently than early layers?

---

## NumPy Implementation

```python
import numpy as np

# Exercise 1.1 - Attention Basics
def attention(Q, K, V):
    """
    Q: (d_k,) query vector
    K: (n, d_k) key matrix
    V: (n, d_v) value matrix
    """
    scores = np.dot(K, Q)  # (n,)
    weights = softmax(scores)  # (n,)
    output = np.dot(weights, V)  # (d_v,)
    return output, weights

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

q = np.array([1, 0, 1])
K = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
V = np.array([[2, 0], [0, 2], [1, 1]])

output, weights = attention(q, K, V)
print("Attention output:", output)
print("Attention weights:", weights)

# Exercise 2.2 - Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_k, d_v)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # (batch, seq_q, seq_k)

    if mask is not None:
        scores = scores + mask  # Add -inf to masked positions

    weights = softmax(scores)  # (batch, seq_q, seq_k)
    output = np.matmul(weights, V)  # (batch, seq_q, d_v)

    return output, weights

# Test
Q = np.random.randn(2, 3, 4)
K = np.random.randn(2, 3, 4)
V = np.random.randn(2, 3, 8)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"\nScaled attention output shape: {output.shape}")

# Exercise 2.3 - Causal Mask
def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask * -1e9  # -inf
    return mask

mask = create_causal_mask(4)
print("\nCausal mask:")
print(mask)

# Exercise 3.2 - Multi-Head Attention
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads

        # Initialize weights
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01

    def split_heads(self, x, batch_size):
        """
        x: (batch, seq, d_model)
        return: (batch, num_heads, seq, d_k)
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # Linear projections
        Q = np.dot(Q, self.W_Q)  # (batch, seq, d_model)
        K = np.dot(K, self.W_K)
        V = np.dot(V, self.W_V)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch, heads, seq, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        d_k = self.d_k
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        weights = softmax(scores)
        attention_output = np.matmul(weights, V)  # (batch, heads, seq, d_k)

        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_k)
        attention_output = attention_output.reshape(batch_size, -1, self.d_model)

        # Final linear projection
        output = np.dot(attention_output, self.W_O)

        return output

# Test
mha = MultiHeadAttention(d_model=8, num_heads=2)
X = np.random.randn(1, 4, 8)
output = mha.forward(X, X, X)
print(f"\nMulti-head attention output shape: {output.shape}")

# Exercise 4.2 - Positional Encoding
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))

    return PE

PE = positional_encoding(seq_len=10, d_model=512)
print(f"\nPositional encoding shape: {PE.shape}")

# Exercise 5.2 - Layer Normalization
def layer_norm(x, gamma, beta, eps=1e-8):
    """
    x: (batch, seq, features)
    gamma, beta: (features,)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

X = np.random.randn(2, 3, 4)
gamma = np.ones(4)
beta = np.zeros(4)

X_norm = layer_norm(X, gamma, beta)
print(f"\nLayer norm output shape: {X_norm.shape}")

# Exercise 5.3 - Feed-Forward Network
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # First layer + ReLU
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)

        # Second layer
        output = np.dot(hidden, self.W2) + self.b2

        return output

ff = FeedForward(d_model=512, d_ff=2048)
X = np.random.randn(1, 10, 512)
output = ff.forward(X)
print(f"\nFeed-forward output shape: {output.shape}")

# Challenge 1 - Transformer Encoder Block
class TransformerEncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.mha.forward(x, x, x, mask)

        # Add & Norm
        x = layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)

        # Feed-forward
        ffn_output = self.ffn.forward(x)

        # Add & Norm
        x = layer_norm(x + ffn_output, self.ln2_gamma, self.ln2_beta)

        return x

# Test encoder block
encoder_block = TransformerEncoderBlock(d_model=8, num_heads=2, d_ff=32)
X = np.random.randn(1, 4, 8)
output = encoder_block.forward(X)
print(f"\nEncoder block output shape: {output.shape}")
```

---

## Tips for Success

1. **Understand attention first** - Core mechanism of Transformers
2. **Shapes are key** - Track dimensions through every operation
3. **Residual connections** - Enable deep networks
4. **Multi-head = multiple perspectives** - Different patterns
5. **Positional encoding** - Injects sequence order info
6. **Masking** - Critical for autoregressive generation
7. **Layer norm** - Stabilizes training
8. **Start simple** - Build single attention, then multi-head, then full block
