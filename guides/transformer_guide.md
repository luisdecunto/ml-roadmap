# Transformer from Scratch

**Time:** 12-15 hours | **Difficulty:** Advanced

## Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    Returns: (batch, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    
    # Scores: QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask (for decoder)
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # Attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Weighted sum of values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights
```

## Positional Encoding

```python
def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding"""
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    
    angles = pos / np.power(10000, (2 * (i // 2)) / d_model)
    
    # Apply sin to even indices, cos to odd
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    
    return angles
```

## Multi-Head Attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Linear projections and split into heads
        Q = np.matmul(Q, self.W_q).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.matmul(K, self.W_k).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.matmul(V, self.W_v).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Attention for each head
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Final linear
        output = np.matmul(attn_output, self.W_o)
        
        return output
```

## Encoder Block

```python
class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention + residual
        attn_out = self.attention.forward(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN + residual
        ffn_out = self.ffn.forward(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

## Project: Addition Transformer

Train transformer to learn addition: "23+45" → "68"

**Target:** >95% accuracy on random 2-digit additions

## Resources
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Karpathy's minGPT](https://github.com/karpathy/minGPT)
