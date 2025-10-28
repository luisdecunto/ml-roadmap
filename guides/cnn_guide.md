# CNN from Scratch - Implementation Guide

**Time:** 10-12 hours | **Difficulty:** Advanced

## Core Implementations

### 2D Convolution
```python
def conv2d(image, kernel):
    H, W = image.shape
    KH, KW = kernel.shape
    output = np.zeros((H - KH + 1, W - KW + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = np.sum(image[i:i+KH, j:j+KW] * kernel)
    return output
```

### Max Pooling
```python
def max_pool2d(x, pool_size=2):
    H, W = x.shape
    H_out = H // pool_size
    W_out = W // pool_size
    output = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            output[i,j] = np.max(x[i*pool_size:(i+1)*pool_size, 
                                   j*pool_size:(j+1)*pool_size])
    return output
```

### Batch Normalization
```python
def batch_norm_forward(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    return out, (x, x_norm, mean, var, gamma, beta, eps)

def batch_norm_backward(dout, cache):
    x, x_norm, mean, var, gamma, beta, eps = cache
    m = x.shape[0]
    
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=0)
    dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0)
    
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mean) / m + dmean / m
    return dx, dgamma, dbeta
```

## MNIST CNN Project

**Architecture:** Conv(32) → ReLU → MaxPool → Conv(64) → ReLU → MaxPool → FC(128) → ReLU → FC(10)

**Target:** >98% accuracy

## Resources
- [CS231n Lecture 5](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture5.pdf)
- [Victor Zhou's CNN Tutorial](https://victorzhou.com/blog/intro-to-cnns-part-1/)
