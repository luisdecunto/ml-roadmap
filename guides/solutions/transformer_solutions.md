# Transformer Solutions - Modules 9-10

Complete step-by-step solutions for all Transformer exercises.

---

## Part 1: Attention Mechanism

### Solution 1.1: Attention Basics

**Given:**
- Query q = [1, 0, 1]
- Keys K = [[1, 1, 0], [0, 1, 1], [1, 0, 0]]
- Values V = [[2, 0], [0, 2], [1, 1]]

**Step 1: Compute attention scores (s_i = q · k_i)**
```
s_1 = [1,0,1] · [1,1,0] = 1×1 + 0×1 + 1×0 = 1
s_2 = [1,0,1] · [0,1,1] = 1×0 + 0×1 + 1×1 = 1
s_3 = [1,0,1] · [1,0,0] = 1×1 + 0×0 + 1×0 = 1

Scores: [1, 1, 1]
```

**Step 2: Apply softmax**
```
exp(scores) = [e^1, e^1, e^1] = [2.718, 2.718, 2.718]
sum = 3 × 2.718 = 8.154

α = softmax([1, 1, 1]) = [2.718/8.154, 2.718/8.154, 2.718/8.154]
  = [0.333, 0.333, 0.333]
```

**Step 3: Weighted sum of values**
```
output = 0.333×[2,0] + 0.333×[0,2] + 0.333×[1,1]
       = [0.666, 0] + [0, 0.666] + [0.333, 0.333]
       = [0.999, 0.999]
       ≈ [1, 1]
```

**Step 4: What does high attention weight mean?**
- High attention weight = query is "similar" to that key
- That value contributes more to the output
- Attention is a **soft, differentiable lookup mechanism**
- Unlike hard indexing, all values contribute (weighted by similarity)

---

### Solution 1.2: Attention as Soft Lookup

**Database:**
- Keys: ["cat", "dog", "bird"] → k_cat, k_dog, k_bird
- Values: [def_cat, def_dog, def_bird]
- Query: "feline" → q_feline

**Step 1: Calculate similarities**
```
Assume embeddings make "feline" similar to "cat":
score_cat = dot(q_feline, k_cat) = 0.9
score_dog = dot(q_feline, k_dog) = 0.2
score_bird = dot(q_feline, k_bird) = 0.1
```

**Step 2: Softmax to get weights**
```
α = softmax([0.9, 0.2, 0.1])
  ≈ [0.72, 0.15, 0.13]
```

**Step 3: Retrieve weighted combination**
```
output = 0.72×def_cat + 0.15×def_dog + 0.13×def_bird
```

**Result:** Mostly cat definition, with small contributions from dog/bird

**Step 4: Why "soft" vs hard lookup?**

**Hard lookup:**
- Discrete: retrieve exactly one value
- Not differentiable (can't backprop)
- All-or-nothing retrieval

**Soft lookup (attention):**
- Continuous: weighted combination of all values
- Fully differentiable (trainable)
- Graceful handling of ambiguity
- Can retrieve multiple relevant items

**Example:** Query "pet" might attend to both cat AND dog, getting a blend of both definitions!

---

### Solution 1.3: Self-Attention

**Sentence:** "The cat sat"
**Embeddings:**
- "The" = [1, 0]
- "cat" = [0, 1]
- "sat" = [1, 1]

**Step 1: Each word is Q, K, V**
```
For self-attention, each word looks at ALL words (including itself)
```

**Step 2: Calculate attention for "cat" attending to all words**

Query = "cat" = [0, 1]

```
Score with "The": [0,1] · [1,0] = 0
Score with "cat": [0,1] · [0,1] = 1
Score with "sat": [0,1] · [1,1] = 1

Scores: [0, 1, 1]
Softmax: [0.12, 0.44, 0.44]
```

**Step 3: Which words does "cat" attend to most?**
- **Itself ("cat")**: 44%
- **"sat"**: 44% (high similarity)
- **"The"**: 12% (low similarity)

**Step 4: How is this different from fixed window?**

**Fixed window (e.g., convolution):**
- Only looks at nearby words (e.g., window size 3)
- Long-range dependencies require stacking many layers
- Fixed, non-learnable attention pattern

**Self-attention:**
- Looks at ALL words simultaneously
- Can capture long-range dependencies in one layer
- Learned, data-driven attention patterns
- Computational cost: O(n²) vs O(nk) for window

**Key insight:** Self-attention is **content-based** (depends on word meanings), not position-based.

---

### Solution 1.4: Attention Visualization

**Sentence:** "I love machine learning" (4 words)

**Step 1: Compute 4×4 attention matrix (self-attention)**

Each row = one word as query, columns = attention distribution over all words

```
Example attention matrix (hypothetical):
                 I    love  machine  learning
        I     [0.4   0.3    0.2      0.1    ]
        love  [0.2   0.4    0.2      0.2    ]
        machine [0.1 0.1    0.5      0.3    ]
        learning [0.1 0.1   0.3      0.5    ]
```

**Step 2: Which words attend to which?**
- **"I"** attends mostly to itself, some to "love"
- **"love"** distributes attention across all words
- **"machine"** attends strongly to itself and "learning" (compound concept)
- **"learning"** attends to itself and "machine"

**Step 3: Why do words often attend to themselves?**
- **Preserve own information**: Residual-like behavior
- **Context integration**: Blend own embedding with context
- **Softmax bias**: When similarities are similar, softmax spreads out
- **Self-attention includes self**: By design, helps gradient flow

**Step 4: Attention heatmap**
```
        I  love  mach  learn
    I  [█   ▓    ░     ░   ]
  love [░   █    ░     ░   ]
  mach [░   ░    █     ▓   ]
 learn [░   ░    ▓     █   ]
```
- █ = high attention (0.4-0.5)
- ▓ = medium attention (0.2-0.3)
- ░ = low attention (0.1-0.2)

**Observation:** Diagonal is often bright (self-attention), and semantically related words attend to each other.

---

## Part 2: Scaled Dot-Product Attention

### Solution 2.1: Scaling Factor

**Given:** Q, K dimensions: d_k = 64

**Step 1: Without scaling, scores can be very large**

For random vectors from N(0,1):
```python
q = np.random.randn(64)
k = np.random.randn(64)
score = np.dot(q, k)
# Typical magnitude: ±8 to ±10
```

**Step 2: Why? Dot product grows with dimension**

For independent random variables:
```
E[q·k] = 0
Var[q·k] = Var[Σ q_i·k_i] = Σ Var[q_i·k_i] = d_k × (1×1) = d_k
```

Standard deviation grows as **√d_k**

**Step 3: Example computation**
```
For d_k=64:
q = [random 64-dim vector]
k = [random 64-dim vector]
dot(q, k) ≈ 8.5 (typical value, std dev = √64 = 8)
```

**Step 4: Scaled: dot(q, k) / √d_k - why does this help?**

**Problem:** Large scores → softmax saturates
```
softmax([20, 19, 18]) ≈ [0.999, 0.0009, 0.0001]
```
Gradients vanish! Almost all weight on one value.

**Solution:** Scale by 1/√d_k
```
Scaled scores: [20/8, 19/8, 18/8] = [2.5, 2.375, 2.25]
softmax([2.5, 2.375, 2.25]) ≈ [0.43, 0.32, 0.25]
```
More balanced attention, better gradients!

**Math intuition:** Scaling makes variance = 1 regardless of d_k
```
Var[q·k / √d_k] = Var[q·k] / d_k = d_k / d_k = 1
```

---

### Solution 2.2: Implement Scaled Attention

**Given:**
- Q shape: (batch=2, seq_len=3, d_k=4)
- K shape: (2, 3, 4)
- V shape: (2, 3, 8)

**Step 1: Compute scores QK^T**
```python
# Q: (2, 3, 4)
# K^T: (2, 4, 3)  [transpose last two dimensions]
# Scores: (2, 3, 4) @ (2, 4, 3) → (2, 3, 3)
```

Each of 2 batches has a 3×3 attention matrix (3 queries × 3 keys)

**Step 2: Scale by 1/√d_k = 1/√4 = 0.5**
```python
scores = scores * 0.5
# or equivalently: scores / 2
```

**Step 3: Apply softmax over last dimension**
```python
# For each query (row), softmax over all keys (columns)
weights = softmax(scores, axis=-1)  # shape: (2, 3, 3)

# Each row sums to 1:
# weights[0, 0, :] = [w1, w2, w3] where w1+w2+w3 = 1
```

**Step 4: Multiply by V**
```python
# weights: (2, 3, 3)
# V: (2, 3, 8)
# output: (2, 3, 3) @ (2, 3, 8) → (2, 3, 8)
```

**Output shape: (2, 3, 8)**
- Same batch size (2)
- Same sequence length as Q (3)
- d_v dimension from V (8)

**Complete code:**
```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V)
    return output, weights
```

---

### Solution 2.3: Attention Masking

**Task:** Decoder self-attention - prevent looking at future tokens

**Sequence length: 4** (e.g., generating "I love ML <EOS>")

**Step 1: Create causal mask**
```python
mask = np.triu(np.ones((4, 4)), k=1)
# Upper triangular matrix (above diagonal):
# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]

# Convert to -inf for masked positions:
mask = mask * -1e9
# [[ 0,  -inf, -inf, -inf],
#  [ 0,   0,   -inf, -inf],
#  [ 0,   0,    0,   -inf],
#  [ 0,   0,    0,    0  ]]
```

**Step 2: Add to attention scores before softmax**
```python
scores = QK^T / √d_k  # e.g., [[2, 3, 1, 2],
                      #        [1, 2, 3, 1],
                      #        [3, 1, 2, 3],
                      #        [2, 2, 1, 3]]

scores = scores + mask  # [[2, -inf, -inf, -inf],
                        #  [1,  2,   -inf, -inf],
                        #  [3,  1,    2,   -inf],
                        #  [2,  2,    1,    3  ]]
```

**Step 3: After softmax: future positions have weight 0**
```python
weights = softmax(scores)
# [[1.0, 0,   0,   0  ],  # Token 0 only attends to itself
#  [0.4, 0.6, 0,   0  ],  # Token 1 attends to 0,1
#  [0.5, 0.1, 0.4, 0  ],  # Token 2 attends to 0,1,2
#  [0.2, 0.3, 0.1, 0.4]]  # Token 3 attends to all
```

**Step 4: Why is this necessary for language modeling?**

**Without masking:**
- Model could "cheat" by looking at future tokens during training
- At test time, future tokens don't exist yet!
- Would learn to copy from future instead of predicting

**With masking:**
- Forces autoregressive generation: predict next token from previous only
- Training matches test-time conditions
- Learns true language modeling distribution P(x_t | x_<t)

**Example:**
```
Training: "I love machine learning"
Without mask: When predicting "learning", model sees "learning" in input → cheating!
With mask: When predicting "learning", model only sees "I love machine" → real task!
```

---

## Part 3: Multi-Head Attention

### Solution 3.1: Single Head vs Multi-Head

**Single head:** d_model = 512, 1 attention mechanism
- All information processed by one attention operation
- Single "perspective" on the data

**Multi-head:** d_model = 512, h = 8 heads

**Step 1: Each head dimension**
```
d_k = d_v = d_model / h = 512 / 8 = 64
```

**Step 2: Why split into multiple heads?**

**Multiple perspectives:**
- Head 1 might learn syntactic relationships (subject-verb)
- Head 2 might learn semantic relationships (synonyms)
- Head 3 might learn positional patterns
- Head 4 might learn long-range dependencies

**Analogy:** Like having multiple experts, each specializing in different patterns

**Step 3: Different heads learn different patterns**

Example from real BERT model:
- Some heads attend to next word
- Some heads attend to previous word
- Some heads attend to same syntactic role (all verbs)
- Some heads attend to related entities

**Step 4: Concatenate outputs**
```
Each head outputs: (batch, seq, 64)
8 heads: [(batch, seq, 64), ..., (batch, seq, 64)]
Concatenate: (batch, seq, 8×64) = (batch, seq, 512)
```

Then apply final linear layer W_O to mix information from all heads.

---

### Solution 3.2: Implement Multi-Head Attention

**Given:**
- Input X: (batch=1, seq=4, d_model=8)
- Heads h=2
- d_k = d_v = 4 per head

**Step 1: Linear projections for each head**
```python
# W_Q, W_K, W_V: each (8, 8)
# These project AND implicitly split into heads

Q = X @ W_Q  # (1, 4, 8)
K = X @ W_K  # (1, 4, 8)
V = X @ W_V  # (1, 4, 8)
```

**Step 2: Reshape to separate heads**
```python
# Reshape: (batch, seq, d_model) → (batch, seq, num_heads, d_k)
Q = Q.reshape(1, 4, 2, 4)
K = K.reshape(1, 4, 2, 4)
V = V.reshape(1, 4, 2, 4)

# Transpose: (batch, seq, heads, d_k) → (batch, heads, seq, d_k)
Q = Q.transpose(0, 2, 1, 3)  # (1, 2, 4, 4)
K = K.transpose(0, 2, 1, 3)  # (1, 2, 4, 4)
V = V.transpose(0, 2, 1, 3)  # (1, 2, 4, 4)
```

**Step 3: Apply scaled attention per head**
```python
# For each of 2 heads independently:
# Q: (1, 2, 4, 4) - 2 heads, each with 4 queries of dim 4
# K: (1, 2, 4, 4)
# V: (1, 2, 4, 4)

scores = Q @ K.transpose(-2, -1) / np.sqrt(4)  # (1, 2, 4, 4)
weights = softmax(scores, axis=-1)
attn_output = weights @ V  # (1, 2, 4, 4)
```

**Step 4: Concatenate heads**
```python
# Transpose back: (1, 2, 4, 4) → (1, 4, 2, 4)
attn_output = attn_output.transpose(0, 2, 1, 3)

# Reshape: (1, 4, 2, 4) → (1, 4, 8)
attn_output = attn_output.reshape(1, 4, 8)
```

**Step 5: Final linear projection**
```python
# W_O: (8, 8)
output = attn_output @ W_O  # (1, 4, 8)
```

**Final output shape: (1, 4, 8)** - same as input!

---

### Solution 3.3: Parameter Count

**Transformer block: d_model = 512, h = 8**

**Multi-head attention parameters:**

**1. W_Q, W_K, W_V: each 512 × 512**
- W_Q: 512 × 512 = 262,144
- W_K: 512 × 512 = 262,144
- W_V: 512 × 512 = 262,144
- Subtotal: 786,432

**2. W_O: 512 × 512**
- W_O: 512 × 512 = 262,144

**3. Total MHA parameters:**
- **1,048,576 parameters** (≈ 1M)
- Ignoring biases (usually small: 4×512 = 2,048 more)

**4. Compare with standard attention (no multi-head):**

Standard attention would also have W_Q, W_K, W_V, W_O:
- Same parameter count!

**Key difference:**
- Multi-head doesn't add parameters
- Instead, splits capacity across h heads
- Each head is lower-dimensional (d_k = d_model/h)
- Benefit is in representation power, not parameter count

**With biases:**
```
W_Q: 512×512 + 512 = 262,656
W_K: 512×512 + 512 = 262,656
W_V: 512×512 + 512 = 262,656
W_O: 512×512 + 512 = 262,656
Total: 1,050,624 parameters
```

---

### Solution 3.4: Cross-Attention vs Self-Attention

**Encoder-Decoder Attention (Cross-Attention)**

**Step 1: Q from decoder**
```python
Q = decoder_hidden_state  # (batch, tgt_seq, d_model)
```

**Step 2: K, V from encoder**
```python
K = encoder_output  # (batch, src_seq, d_model)
V = encoder_output  # (batch, src_seq, d_model)
```

**Step 3: Decoder attends to all encoder outputs**
```python
# Attention scores: (batch, tgt_seq, src_seq)
scores = Q @ K^T / √d_k

# Each decoder position attends to ALL encoder positions
weights = softmax(scores, axis=-1)
output = weights @ V
```

**Step 4: Use case - Machine Translation**

**Example:** English → French
- English input: "I love machine learning"
- French output: "J'aime l'apprentissage automatique"

**Encoder (self-attention):**
- English words attend to each other
- Builds contextual representations

**Decoder (self-attention + cross-attention):**
- French words attend to previous French words (self-attention)
- French words attend to ALL English words (cross-attention)

**Cross-attention matrix (tgt × src):**
```
                I    love  machine  learning
J'             [0.8  0.1   0.05     0.05   ]
aime           [0.1  0.8   0.05     0.05   ]
l'apprentissage[0.0  0.0   0.3      0.7    ]
automatique    [0.0  0.0   0.7      0.3    ]
```

**Key differences:**

| Aspect | Self-Attention | Cross-Attention |
|--------|---------------|-----------------|
| Q source | Same sequence | Decoder |
| K, V source | Same sequence | Encoder |
| Purpose | Context within sequence | Attend to other sequence |
| Use | Encoder, Decoder | Decoder only |
| Attention shape | (seq, seq) | (tgt_seq, src_seq) |

---

## Part 4: Positional Encoding

### Solution 4.1: Why Positional Encoding?

**Without position info:**

**Sentence 1:** "cat eats fish"
**Sentence 2:** "fish eats cat"

**Step 1: Same words, different meaning**
- Sentence 1: Cat is subject, fish is object
- Sentence 2: Fish is subject, cat is object
- Completely different meanings!

**Step 2: Attention is permutation invariant**
```python
# Self-attention doesn't care about order
attention(["cat", "eats", "fish"])
== attention(["fish", "eats", "cat"])  # Same output!
```

**Step 3: Need to inject position information**
Without position encoding, model can't distinguish word order.

**Step 4: Add positional encoding to embeddings**
```python
# Word embeddings
word_emb = embedding(tokens)  # (seq, d_model)

# Positional encodings
pos_enc = positional_encoding(seq_len, d_model)  # (seq, d_model)

# Add together
input = word_emb + pos_enc
```

**Why add, not concatenate?**
- Simpler, fewer parameters
- Allows model to trade off word vs position info
- Empirically works well

---

### Solution 4.2: Sinusoidal Positional Encoding

**Formula:**
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

**Given:** pos = 0, 1, 2 and d_model = 4

**Position 0:**
```
i=0 (dim 0): PE(0, 0) = sin(0 / 10000^(0/4)) = sin(0) = 0
i=0 (dim 1): PE(0, 1) = cos(0 / 10000^(0/4)) = cos(0) = 1
i=1 (dim 2): PE(0, 2) = sin(0 / 10000^(2/4)) = sin(0) = 0
i=1 (dim 3): PE(0, 3) = cos(0 / 10000^(2/4)) = cos(0) = 1

PE(0) = [0, 1, 0, 1]
```

**Position 1:**
```
i=0 (dim 0): PE(1, 0) = sin(1 / 10000^0) = sin(1) ≈ 0.841
i=0 (dim 1): PE(1, 1) = cos(1 / 10000^0) = cos(1) ≈ 0.540
i=1 (dim 2): PE(1, 2) = sin(1 / 10000^0.5) = sin(1/100) = sin(0.01) ≈ 0.01
i=1 (dim 3): PE(1, 3) = cos(1 / 10000^0.5) = cos(0.01) ≈ 0.9999

PE(1) ≈ [0.841, 0.540, 0.01, 0.9999]
```

**Position 2:**
```
i=0 (dim 0): PE(2, 0) = sin(2) ≈ 0.909
i=0 (dim 1): PE(2, 1) = cos(2) ≈ -0.416
i=1 (dim 2): PE(2, 2) = sin(0.02) ≈ 0.02
i=1 (dim 3): PE(2, 3) = cos(0.02) ≈ 0.9998

PE(2) ≈ [0.909, -0.416, 0.02, 0.9998]
```

**Why sine/cosine?**

1. **Unique encodings:** Each position has unique pattern
2. **Bounded:** Values in [-1, 1], won't dominate embeddings
3. **Smooth:** Adjacent positions have similar encodings
4. **Relative positions:** PE(pos+k) can be computed from PE(pos) via linear transform
5. **Extrapolation:** Can handle longer sequences than seen during training

**Mathematical property:**
```
PE(pos + k) = PE(pos) * M_k  # Linear transformation exists!
```

This allows model to learn relative positions easily.

---

### Solution 4.3: Learned vs Fixed Positional Encoding

**Two approaches:**

**1. Fixed (Sinusoidal):**
```python
PE = np.zeros((seq_len, d_model))
for pos in range(seq_len):
    for i in range(0, d_model, 2):
        PE[pos, i] = sin(pos / 10000^(2i/d_model))
        PE[pos, i+1] = cos(pos / 10000^(2i/d_model))
```

**2. Learned (Embedding):**
```python
position_embedding = nn.Embedding(max_seq_len, d_model)
# Trained via backprop like word embeddings
```

**Comparison:**

| Aspect | Fixed (Sinusoidal) | Learned |
|--------|-------------------|---------|
| **Parameters** | 0 (no learning) | max_seq_len × d_model |
| **Training** | Not trained | Learned via backprop |
| **Extrapolation** | Can handle longer sequences | Limited to max_seq_len |
| **Theory** | Nice mathematical properties | Purely data-driven |
| **Performance** | Slightly worse | Slightly better on training data |
| **Used in** | Original Transformer | BERT, GPT |

**Pros/Cons:**

**Fixed (Sinusoidal):**
- ✅ No parameters, less overfitting
- ✅ Can extrapolate to longer sequences
- ✅ Interpretable (frequency-based)
- ❌ Can't adapt to data

**Learned:**
- ✅ Can learn task-specific patterns
- ✅ Often performs slightly better
- ❌ More parameters, can overfit
- ❌ Limited to max training length

**What does original Transformer use?**
- **Sinusoidal (fixed)** in "Attention is All You Need" paper
- Modern models (BERT, GPT) use **learned** embeddings

---

## Part 5: Transformer Blocks

### Solution 5.1: Encoder Block

**Components:**
1. Multi-Head Self-Attention
2. Add & Norm (residual + layer norm)
3. Feed-Forward Network
4. Add & Norm

**Input:** X shape (batch, seq, d_model)

**Trace through:**

**Step 1: Multi-Head Attention**
```python
attn_output = MultiHeadAttention(X, X, X)  # Self-attention
# Shape: (batch, seq, d_model)
```

**Step 2: Add residual**
```python
X = X + attn_output  # Residual connection
# Shape: (batch, seq, d_model)
```

**Step 3: Layer normalization**
```python
X = LayerNorm(X)
# Shape: (batch, seq, d_model)
```

**Step 4: Feed-Forward Network**
```python
ffn_output = FFN(X)  # Position-wise FFN
# Shape: (batch, seq, d_model)
```

**Step 5: Add & Norm again**
```python
X = X + ffn_output  # Residual connection
X = LayerNorm(X)
# Shape: (batch, seq, d_model)
```

**Step 6: Final output shape**
**(batch, seq, d_model)** - same as input!

**Key insight:** Residual connections preserve dimensions, allowing deep stacking.

**Complete code:**
```python
def encoder_block(X):
    # Multi-head self-attention
    attn = MultiHeadAttention(X, X, X)
    X = LayerNorm(X + attn)

    # Feed-forward
    ffn = FFN(X)
    X = LayerNorm(X + ffn)

    return X
```

---

### Solution 5.2: Layer Normalization

**Input:** X shape (2, 3, 4) [batch, seq, features]

**Step 1: Compute mean and variance across features (last dim)**
```python
# For each (batch, seq) position, normalize across features
mean = np.mean(X, axis=-1, keepdims=True)  # (2, 3, 1)
var = np.var(X, axis=-1, keepdims=True)    # (2, 3, 1)

# Example for X[0, 0, :] = [1, 2, 3, 4]:
# mean[0, 0] = (1+2+3+4)/4 = 2.5
# var[0, 0] = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4
#           = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
```

**Step 2: Normalize**
```python
X_norm = (X - mean) / np.sqrt(var + eps)
# eps (e.g., 1e-8) prevents division by zero

# After normalization:
# mean(X_norm, axis=-1) ≈ 0
# var(X_norm, axis=-1) ≈ 1
```

**Step 3: Scale and shift (learnable γ, β)**
```python
# γ, β: (features,) = (4,)
output = γ * X_norm + β

# Allows model to learn optimal scale/shift
# Can even undo normalization if needed!
```

**Step 4: Why layer norm instead of batch norm in Transformers?**

**Batch Norm:**
- Normalizes across batch dimension
- Requires large batches for stable statistics
- Different behavior train vs test (running mean/var)
- **Problem:** Sequence lengths vary, batch may be small

**Layer Norm:**
- Normalizes across feature dimension
- Independent of batch size (even works with batch=1)
- Same behavior train and test
- **Perfect for sequences:** Each position normalized independently

**Example:**
```
Batch Norm: Normalize all samples' feature_i together
Layer Norm: Normalize each sample's all features together

Batch Norm: Mean across (batch, seq) for each feature
Layer Norm: Mean across features for each (batch, seq) position
```

**Why Layer Norm works for Transformers:**
- Sequential data has variable lengths
- Want stable training with small batches
- Need consistent behavior train/test
- Each token should be normalized independently

---

### Solution 5.3: Feed-Forward Network

**FFN(x) = max(0, xW_1 + b_1)W_2 + b_2**

**Given:** d_model = 512, d_ff = 2048

**Step 1: First linear layer: 512 → 2048**
```python
hidden = x @ W_1 + b_1
# x: (batch, seq, 512)
# W_1: (512, 2048)
# b_1: (2048,)
# hidden: (batch, seq, 2048)
```

**Step 2: ReLU activation**
```python
hidden = np.maximum(0, hidden)
# Apply element-wise: negative → 0, positive unchanged
```

**Step 3: Second linear layer: 2048 → 512**
```python
output = hidden @ W_2 + b_2
# hidden: (batch, seq, 2048)
# W_2: (2048, 512)
# b_2: (512,)
# output: (batch, seq, 512)
```

**Step 4: Parameters**
```
W_1: 512 × 2048 = 1,048,576
b_1: 2048
W_2: 2048 × 512 = 1,048,576
b_2: 512

Total: 2,097,152 + 2,560 ≈ 2.1M parameters
```

**Key points:**
- FFN is **position-wise**: Applied independently to each position
- **Expansion and compression**: 512 → 2048 → 512
- Typically d_ff = 4 × d_model (expansion factor of 4)
- Most parameters in Transformer are in FFN!

**Why position-wise?**
- Attention mixes information across positions
- FFN processes each position independently
- Gives model capacity to transform representations

**Comparison:**
- Multi-head attention: ~1M params (for d_model=512)
- FFN: ~2.1M params
- **FFN has more parameters!**

---

## Part 6: Full Transformer

### Solution 6.1: Encoder Stack

**6 encoder layers, d_model = 512, h = 8**

**Input:** Sentence "Hello world" (2 tokens)

**Step 1: Token embedding + positional encoding**
```python
# Token embeddings
tokens = ["Hello", "world"]
token_ids = [4321, 8765]  # vocabulary indices
token_emb = embedding_matrix[token_ids]  # (2, 512)

# Positional encoding
pos_enc = positional_encoding(seq_len=2, d_model=512)  # (2, 512)

# Add together
encoder_input = token_emb + pos_enc  # (2, 512)
# Or with batch: (batch, 2, 512)
```

**Step 2: Pass through encoder layer 1**
```python
output_layer1 = EncoderBlock1(encoder_input)
# Self-attention + Add&Norm + FFN + Add&Norm
# Shape: (batch, 2, 512)
```

**Step 3: Output goes to encoder layer 2**
```python
output_layer2 = EncoderBlock2(output_layer1)
# Shape: (batch, 2, 512)
```

**Step 4: ... through layer 6**
```python
output_layer3 = EncoderBlock3(output_layer2)
output_layer4 = EncoderBlock4(output_layer3)
output_layer5 = EncoderBlock5(output_layer4)
output_layer6 = EncoderBlock6(output_layer5)
```

**Step 5: Final encoder output shape**
**(batch, 2, 512)**

- Batch dimension unchanged
- Sequence length unchanged (2 tokens)
- d_model unchanged (512)

**Key insight:** Encoder transforms input embeddings into contextualized representations, preserving shape.

---

### Solution 6.2: Decoder Stack

**6 decoder layers, each with 3 sub-layers:**

**Step 1: Masked self-attention (causal)**
```python
# Decoder input attends to previous decoder positions only
attn1 = MultiHeadAttention(X, X, X, mask=causal_mask)
```

**Step 2: Add & Norm**
```python
X = LayerNorm(X + attn1)
```

**Step 3: Cross-attention (attend to encoder output)**
```python
# Q from decoder, K and V from encoder
attn2 = MultiHeadAttention(Q=X, K=encoder_output, V=encoder_output)
```

**Step 4: Add & Norm**
```python
X = LayerNorm(X + attn2)
```

**Step 5: FFN**
```python
ffn_out = FFN(X)
```

**Step 6: Add & Norm**
```python
X = LayerNorm(X + ffn_out)
```

**Why masked self-attention in decoder?**

**Purpose:** Autoregressive generation
- During training: target sequence is available
- Must prevent decoder from "cheating" by seeing future tokens
- Causal mask ensures position t can only attend to positions ≤ t

**Example:**
```
Target: "I love ML <EOS>"
Position 0 ("I"): attends to nothing (just positional encoding)
Position 1 ("love"): attends to "I" only
Position 2 ("ML"): attends to "I", "love"
Position 3 ("<EOS>"): attends to "I", "love", "ML"
```

**At test time:**
- Generate one token at a time
- Feed back generated tokens as decoder input
- Masking ensures training matches test-time behavior

---

### Solution 6.3: Full Transformer Forward Pass

**Translation:** "Hello" → "Bonjour"

**Encoder:**

**Step 1: Embed "Hello" + positional encoding**
```python
# Token embedding
hello_emb = embedding_matrix[token_id("Hello")]  # (512,)

# Add positional encoding for position 0
hello_emb = hello_emb + PE(0)  # (512,)

# With batch: (batch, 1, 512)
```

**Step 2: 6 encoder layers**
```python
enc_out = encoder_input  # (batch, 1, 512)
for layer in encoder_layers:  # 6 layers
    enc_out = layer(enc_out)

# Final encoder_output: (batch, 1, 512)
```

**Step 3: encoder_output stored for decoder cross-attention**

**Decoder:**

**Step 1: Embed "<BOS> Bonjour" (teacher forcing)**
```python
# During training, we give decoder the target sequence
decoder_tokens = ["<BOS>", "Bonjour"]
decoder_emb = embedding_matrix[token_ids(decoder_tokens)]  # (2, 512)
decoder_emb = decoder_emb + PE([0, 1])  # Add positional encoding
```

**Step 2: Masked self-attention**
```python
# Causal mask: position 0 can't see position 1
mask = [[0, -inf],
        [0,   0 ]]

# First decoder layer - masked self-attention
attn1 = MultiHeadAttention(decoder_emb, decoder_emb, decoder_emb, mask)
```

**Step 3: Cross-attention with encoder_output**
```python
# Decoder attends to encoder
attn2 = MultiHeadAttention(
    Q=after_attn1,           # From decoder: (batch, 2, 512)
    K=encoder_output,        # From encoder: (batch, 1, 512)
    V=encoder_output
)

# Attention weights: (batch, 2, 1)
# Each decoder position attends to the single encoder position
```

**Step 4: 6 decoder layers**
```python
dec_out = decoder_input  # (batch, 2, 512)
for layer in decoder_layers:  # 6 layers
    dec_out = layer(dec_out, encoder_output, mask)

# Final: (batch, 2, 512)
```

**Step 5: Linear + softmax → next token probabilities**
```python
# Project to vocabulary size
logits = dec_out @ W_vocab  # (batch, 2, vocab_size)

# Softmax to get probabilities
probs = softmax(logits, axis=-1)

# For position 1 (after "<BOS>"):
# probs[0, 1, :] = distribution over all words
# Ideally: high probability for "Bonjour"
```

**Training:**
- Loss = cross_entropy(probs, target_tokens)
- Target: ["<BOS>", "Bonjour", "<EOS>"]
- Compare probs[0] with "<BOS>", probs[1] with "Bonjour", etc.

**Inference (test time):**
```python
1. Start with "<BOS>"
2. Decoder generates probs for next token
3. Sample token (e.g., "Bonjour")
4. Append to decoder input: ["<BOS>", "Bonjour"]
5. Generate next token
6. Repeat until "<EOS>"
```

---

### Solution 6.4: Parameter Count

**Full Transformer: L=6 layers, d_model=512, h=8, d_ff=2048**

**Per encoder layer:**

**1. Multi-Head Attention:**
```
W_Q: 512 × 512 = 262,144
W_K: 512 × 512 = 262,144
W_V: 512 × 512 = 262,144
W_O: 512 × 512 = 262,144
Total MHA: 1,048,576 parameters
```

**2. Feed-Forward Network:**
```
W_1: 512 × 2048 = 1,048,576
b_1: 2048
W_2: 2048 × 512 = 1,048,576
b_2: 512
Total FFN: 2,099,712 parameters
```

**3. Layer Norms (2 per block):**
```
LN1: γ (512) + β (512) = 1,024
LN2: γ (512) + β (512) = 1,024
Total LN: 2,048 parameters
```

**Per encoder layer total:**
1,048,576 + 2,099,712 + 2,048 = **3,150,336 parameters**

**Total encoder (6 layers):**
6 × 3,150,336 = **18,902,016 parameters** ≈ 18.9M

**Per decoder layer:**

Same as encoder, PLUS cross-attention:
```
Cross-attention MHA: 1,048,576
Total per decoder layer: 3,150,336 + 1,048,576 = 4,198,912
```

**Total decoder (6 layers):**
6 × 4,198,912 = **25,193,472 parameters** ≈ 25.2M

**Other components:**

**Embeddings:**
```
Token embedding: vocab_size × 512
(e.g., 30,000 × 512 = 15,360,000)

Positional encoding: 0 (if sinusoidal) or max_seq_len × 512 (if learned)
```

**Output projection:**
```
Final linear: 512 × vocab_size
(e.g., 512 × 30,000 = 15,360,000)
```

**Total Transformer:**
- Encoder: 18.9M
- Decoder: 25.2M
- Embeddings: ~15.4M (input) + ~15.4M (output) = 30.8M
- **Grand Total: ≈ 75M parameters** (for vocab_size=30K)

**Key observation:** Most parameters are in embeddings and FFN layers!

**Parameter distribution:**
- Embeddings: ~41%
- FFN: ~47%
- Attention: ~12%

---

## Challenge Problems

### Challenge 1: Implement Transformer from Scratch

Complete NumPy implementation of a mini-Transformer:

```python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 1. Multi-Head Attention
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize projection matrices
        self.W_Q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_K = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_V = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_O = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def split_heads(self, x):
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # Linear projections
        Q = Q @ self.W_Q
        K = K @ self.W_K
        V = V @ self.W_V

        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)
        attn_output = attn_weights @ V

        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)

        # Output projection
        output = attn_output @ self.W_O

        return output

# 2. Positional Encoding
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))
    return PE

# 3. Layer Normalization
def layer_norm(x, gamma, beta, eps=1e-8):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# 4. Feed-Forward Network
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        output = hidden @ self.W2 + self.b2
        return output

# 5. Encoder Block
class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        # Layer norm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(self, x, mask=None):
        # Multi-head attention + Add & Norm
        attn = self.mha.forward(x, x, x, mask)
        x = layer_norm(x + attn, self.gamma1, self.beta1)

        # Feed-forward + Add & Norm
        ffn = self.ffn.forward(x)
        x = layer_norm(x + ffn, self.gamma2, self.beta2)

        return x

# 6. Decoder Block
class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.gamma3 = np.ones(d_model)
        self.beta3 = np.zeros(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn1 = self.masked_mha.forward(x, x, x, tgt_mask)
        x = layer_norm(x + attn1, self.gamma1, self.beta1)

        # Cross-attention
        attn2 = self.cross_mha.forward(x, encoder_output, encoder_output, src_mask)
        x = layer_norm(x + attn2, self.gamma2, self.beta2)

        # Feed-forward
        ffn = self.ffn.forward(x)
        x = layer_norm(x + ffn, self.gamma3, self.beta3)

        return x

# 7. Full Transformer
class Transformer:
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=2, max_seq_len=100):
        self.d_model = d_model

        # Embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.PE = positional_encoding(max_seq_len, d_model)

        # Encoder and decoder stacks
        self.encoder_blocks = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.01

    def encode(self, src_ids):
        # Embedding + positional encoding
        src_emb = self.embedding[src_ids] + self.PE[:len(src_ids)]
        src_emb = np.expand_dims(src_emb, 0)  # Add batch dim

        # Pass through encoder blocks
        enc_out = src_emb
        for block in self.encoder_blocks:
            enc_out = block.forward(enc_out)

        return enc_out

    def decode(self, tgt_ids, encoder_output):
        # Embedding + positional encoding
        tgt_emb = self.embedding[tgt_ids] + self.PE[:len(tgt_ids)]
        tgt_emb = np.expand_dims(tgt_emb, 0)  # Add batch dim

        # Causal mask
        seq_len = len(tgt_ids)
        tgt_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        tgt_mask = tgt_mask.reshape(1, 1, seq_len, seq_len)

        # Pass through decoder blocks
        dec_out = tgt_emb
        for block in self.decoder_blocks:
            dec_out = block.forward(dec_out, encoder_output, tgt_mask=tgt_mask)

        return dec_out

    def forward(self, src_ids, tgt_ids):
        enc_out = self.encode(src_ids)
        dec_out = self.decode(tgt_ids, enc_out)

        # Project to vocabulary
        logits = dec_out @ self.output_proj
        return logits

# Test on tiny sequence-to-sequence task
# Task: Reverse sequences of digits 0-9

vocab_size = 12  # 0-9 + <BOS> + <EOS>
model = Transformer(vocab_size, d_model=32, num_heads=2, d_ff=64, num_layers=2)

# Example: [1, 2, 3] → [3, 2, 1]
src = np.array([1, 2, 3, 11])  # 11 = <EOS>
tgt = np.array([10, 3, 2, 1, 11])  # 10 = <BOS>

logits = model.forward(src, tgt[:-1])  # Teacher forcing
print(f"Output shape: {logits.shape}")  # (1, 4, 12)

# Training would optimize cross-entropy loss between logits and tgt[1:]
```

**Training tips:**
- Use Adam optimizer (lr=0.001)
- Warmup schedule: linearly increase LR for first 4000 steps
- Label smoothing (ε=0.1)
- Dropout (0.1) during training
- Train on toy tasks first (reverse, copy, addition)

---

### Challenge 2: Attention Patterns

**Analyze attention in trained model:**

```python
def visualize_attention(model, src, tgt, layer_idx=0, head_idx=0):
    """Extract and visualize attention weights"""

    # Forward pass with attention weights stored
    enc_out = model.encode(src)
    dec_out, attn_weights = model.decode_with_attention(tgt, enc_out)

    # Get specific layer and head
    # attn_weights: list of (batch, heads, seq, seq) per layer
    layer_attn = attn_weights[layer_idx]  # (1, num_heads, seq, seq)
    head_attn = layer_attn[0, head_idx]   # (seq, seq)

    # Visualize heatmap
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(head_attn, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.title(f'Attention Pattern (Layer {layer_idx}, Head {head_idx})')
    plt.show()

    return head_attn

# Example analysis
src = ["The", "cat", "sat"]
tgt = ["Le", "chat", "assis"]

# Visualize all heads in first layer
for head in range(num_heads):
    attn = visualize_attention(model, src, tgt, layer_idx=0, head_idx=head)
    print(f"\nHead {head} attention pattern:")
    print(attn)

# Questions to investigate:
# 1. Do different heads learn different patterns?
#    - Some heads: attend to previous word
#    - Some heads: attend to next word
#    - Some heads: attend to related syntactic positions

# 2. Do later layers attend differently than early layers?
#    - Early layers: local, syntactic patterns
#    - Later layers: semantic, long-range dependencies

# 3. Cross-attention patterns in translation:
#    - Which source words does each target word attend to?
#    - Alignment between source and target
```

**Common attention patterns found in practice:**

1. **Positional heads:** Attend to next/previous token
2. **Syntactic heads:** Attend to syntactic parents (e.g., verbs to subjects)
3. **Semantic heads:** Attend to related entities/concepts
4. **Rare heads:** Attend to rare tokens (punctuation, special tokens)

---

**You've completed the Transformer exercises! You now understand:**
- ✅ Attention mechanism (soft lookup, self-attention)
- ✅ Scaled dot-product attention (why scaling matters)
- ✅ Multi-head attention (multiple perspectives)
- ✅ Positional encoding (injecting position information)
- ✅ Transformer blocks (encoder and decoder architecture)
- ✅ Full Transformer (complete sequence-to-sequence model)
- ✅ Implementation from scratch (NumPy only!)

**Next steps:**
- Implement and train on real tasks (translation, summarization)
- Study modern variants (BERT, GPT, T5)
- Explore vision transformers (ViT)
- Dive into efficient attention mechanisms (linear attention, Flash Attention)
