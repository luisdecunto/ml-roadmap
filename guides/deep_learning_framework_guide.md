# Capstone Project 1: Build Your Own Deep Learning Framework

This guide walks you through building a complete deep learning framework from scratch, similar to PyTorch or JAX. You'll implement automatic differentiation, neural network layers, optimizers, and training utilities.

**Time Estimate:** 40-50 hours
**Prerequisites:** Completed Modules 1-11

## Table of Contents
1. [Framework Design & Planning](#part-1-framework-design)
2. [Tensor Class & Autograd Engine](#part-2-tensor-and-autograd)
3. [Neural Network Layers](#part-3-neural-network-layers)
4. [Optimizers](#part-4-optimizers)
5. [Loss Functions](#part-5-loss-functions)
6. [Data Loading](#part-6-data-loading)
7. [Complete Examples](#part-7-complete-examples)

---

## Part 1: Framework Design & Planning

### Overview
Before coding, design your framework's API and architecture.

### Step 1: API Design

```python
"""
Framework Name: TinyTorch

Design Goals:
1. Clean, intuitive API similar to PyTorch
2. Full automatic differentiation support
3. GPU support optional (CPU first)
4. Modular and extensible

Core Components:
- Tensor: Data container with autograd
- nn.Module: Base class for layers
- nn.functional: Functional operations
- optim: Optimizers
- utils.data: DataLoader

Example Usage:

import tinytorch as tt
import tinytorch.nn as nn

# Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = tt.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training
model = Net()
optimizer = tt.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
"""
```

### Step 2: Project Structure

```
tinytorch/
├── __init__.py
├── tensor.py          # Tensor class with autograd
├── autograd.py        # Autograd engine
├── nn/
│   ├── __init__.py
│   ├── module.py      # Base Module class
│   ├── linear.py      # Linear layers
│   ├── conv.py        # Convolutional layers
│   ├── activation.py  # Activation functions
│   ├── loss.py        # Loss functions
│   └── functional.py  # Functional API
├── optim/
│   ├── __init__.py
│   ├── optimizer.py   # Base Optimizer
│   ├── sgd.py         # SGD
│   └── adam.py        # Adam
└── utils/
    ├── __init__.py
    └── data.py        # DataLoader

tests/
├── test_tensor.py
├── test_autograd.py
├── test_layers.py
└── test_gradients.py

examples/
├── mnist_classification.py
├── simple_regression.py
└── custom_layer.py

README.md
setup.py
```

---

## Part 2: Tensor and Autograd

### Overview
Build a Tensor class that tracks operations for automatic differentiation.

### Step 1: Basic Tensor Class

```python
# tensor.py
import numpy as np
from typing import Union, Tuple, Optional, List

class Tensor:
    """
    Tensor with automatic differentiation support.

    Core idea: Build a computational graph dynamically during forward pass,
    then traverse it backward to compute gradients.
    """

    def __init__(self, data, requires_grad=False, _children=(), _op='', label=''):
        """
        Args:
            data: numpy array or scalar
            requires_grad: whether to compute gradients
            _children: parent tensors in computation graph
            _op: operation that created this tensor
            label: optional name for debugging
        """
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None

        # Autograd bookkeeping
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

        if requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor({self.data}, grad_fn={self._op if self._op else None})"

    def backward(self, gradient=None):
        """
        Compute gradients via backpropagation.

        Args:
            gradient: Gradient of loss w.r.t. this tensor (default: ones)
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on tensor with requires_grad=False")

        # Build topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited and v.requires_grad:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient
        if gradient is None:
            if self.data.size == 1:
                self.grad = np.ones_like(self.data, dtype=np.float32)
            else:
                raise RuntimeError("Gradient must be specified for non-scalar output")
        else:
            self.grad = np.array(gradient, dtype=np.float32)

        # Backpropagate
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset gradient to zero."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)
```

### Step 2: Basic Operations with Autograd

```python
# tensor.py (continued)

    def __add__(self, other):
        """Addition: z = x + y"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op='+'
        )

        def _backward():
            if self.requires_grad:
                # Gradient: dL/dx = dL/dz * dz/dx = dL/dz * 1
                grad = out.grad
                # Handle broadcasting
                if self.shape != out.shape:
                    grad = self._sum_to_shape(grad, self.shape)
                self.grad += grad

            if other.requires_grad:
                grad = out.grad
                if other.shape != out.shape:
                    grad = self._sum_to_shape(grad, other.shape)
                other.grad += grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiplication: z = x * y"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op='*'
        )

        def _backward():
            if self.requires_grad:
                # dL/dx = dL/dz * dz/dx = dL/dz * y
                grad = out.grad * other.data
                if self.shape != out.shape:
                    grad = self._sum_to_shape(grad, self.shape)
                self.grad += grad

            if other.requires_grad:
                # dL/dy = dL/dz * dz/dy = dL/dz * x
                grad = out.grad * self.data
                if other.shape != out.shape:
                    grad = self._sum_to_shape(grad, other.shape)
                other.grad += grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        """Power: z = x^n"""
        assert isinstance(power, (int, float)), "Only numeric powers supported"
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{power}'
        )

        def _backward():
            if self.requires_grad:
                # dL/dx = dL/dz * dz/dx = dL/dz * n * x^(n-1)
                self.grad += out.grad * power * (self.data ** (power - 1))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    @staticmethod
    def _sum_to_shape(grad, shape):
        """Sum out added dims and reduce broadcasted dims."""
        # Sum out added dimensions
        ndims_added = grad.ndim - len(shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        # Sum over broadcasted dimensions
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad
```

### Step 3: Matrix Operations

```python
# tensor.py (continued)

    def matmul(self, other):
        """Matrix multiplication: z = x @ y"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _children=(self, other),
            _op='@'
        )

        def _backward():
            if self.requires_grad:
                # dL/dx = dL/dz @ y^T
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                # dL/dy = x^T @ dL/dz
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        """Sum reduction."""
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Expand gradient to match input shape
                if axis is not None:
                    if not keepdims:
                        grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)
                else:
                    grad = np.broadcast_to(grad, self.shape)
                self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean reduction."""
        size = self.data.size if axis is None else self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / size

    def reshape(self, *shape):
        """Reshape tensor."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='reshape'
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        """Transpose tensor."""
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='transpose'
        )

        def _backward():
            if self.requires_grad:
                # Reverse the transpose
                if axes is None:
                    self.grad += out.grad.T
                else:
                    inv_axes = np.argsort(axes)
                    self.grad += out.grad.transpose(inv_axes)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()
```

### Step 4: Activation Functions

```python
# tensor.py (continued)

    def relu(self):
        """ReLU activation: f(x) = max(0, x)"""
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='relu'
        )

        def _backward():
            if self.requires_grad:
                # df/dx = 1 if x > 0 else 0
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid activation: f(x) = 1 / (1 + e^(-x))"""
        sigmoid_val = 1 / (1 + np.exp(-self.data))
        out = Tensor(
            sigmoid_val,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sigmoid'
        )

        def _backward():
            if self.requires_grad:
                # df/dx = f(x) * (1 - f(x))
                self.grad += out.grad * sigmoid_val * (1 - sigmoid_val)

        out._backward = _backward
        return out

    def tanh(self):
        """Tanh activation."""
        tanh_val = np.tanh(self.data)
        out = Tensor(
            tanh_val,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='tanh'
        )

        def _backward():
            if self.requires_grad:
                # df/dx = 1 - f(x)^2
                self.grad += out.grad * (1 - tanh_val ** 2)

        out._backward = _backward
        return out

    def exp(self):
        """Exponential: e^x"""
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='exp'
        )

        def _backward():
            if self.requires_grad:
                # df/dx = e^x
                self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        """Natural logarithm."""
        out = Tensor(
            np.log(self.data + 1e-8),  # Add epsilon for stability
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='log'
        )

        def _backward():
            if self.requires_grad:
                # df/dx = 1/x
                self.grad += out.grad / (self.data + 1e-8)

        out._backward = _backward
        return out
```

### Step 5: Testing Autograd

```python
# test_autograd.py
import numpy as np
from tensor import Tensor

def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient for testing."""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_val = x.data[idx]

        x.data[idx] = old_val + h
        fxh_plus = f(x).data.copy()

        x.data[idx] = old_val - h
        fxh_minus = f(x).data.copy()

        grad[idx] = (fxh_plus - fxh_minus) / (2 * h)
        x.data[idx] = old_val
        it.iternext()

    return grad

def test_basic_operations():
    """Test basic arithmetic operations."""
    print("Testing basic operations...")

    # Addition
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    z = x + y
    z.backward()
    assert np.allclose(x.grad, 1.0), f"Expected 1.0, got {x.grad}"
    assert np.allclose(y.grad, 1.0), f"Expected 1.0, got {y.grad}"
    print("✓ Addition")

    # Multiplication
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    z = x * y
    z.backward()
    assert np.allclose(x.grad, 3.0), f"Expected 3.0, got {x.grad}"
    assert np.allclose(y.grad, 2.0), f"Expected 2.0, got {y.grad}"
    print("✓ Multiplication")

    # Power
    x = Tensor([2.0], requires_grad=True)
    z = x ** 3
    z.backward()
    assert np.allclose(x.grad, 12.0), f"Expected 12.0, got {x.grad}"
    print("✓ Power")

def test_matmul():
    """Test matrix multiplication."""
    print("\nTesting matrix multiplication...")

    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(4, 5), requires_grad=True)
    z = x @ y
    z.sum().backward()

    # Numerical gradient check
    def f_x(x):
        return x @ y

    def f_y(y):
        return x @ y

    num_grad_x = numerical_gradient(lambda t: (t @ y).sum(), x)
    num_grad_y = numerical_gradient(lambda t: (x @ t).sum(), y)

    assert np.allclose(x.grad, num_grad_x, atol=1e-5), "Matmul gradient (x) incorrect"
    assert np.allclose(y.grad, num_grad_y, atol=1e-5), "Matmul gradient (y) incorrect"
    print("✓ Matrix multiplication")

def test_activations():
    """Test activation functions."""
    print("\nTesting activations...")

    # ReLU
    x = Tensor(np.array([[-1, 2], [3, -4]]), requires_grad=True)
    z = x.relu()
    z.sum().backward()
    expected = np.array([[0, 1], [1, 0]])
    assert np.allclose(x.grad, expected), f"ReLU gradient incorrect"
    print("✓ ReLU")

    # Sigmoid
    x = Tensor([0.0], requires_grad=True)
    z = x.sigmoid()
    z.backward()
    # sigmoid'(0) = 0.25
    assert np.allclose(x.grad, 0.25), f"Sigmoid gradient incorrect"
    print("✓ Sigmoid")

def test_complex_graph():
    """Test complex computational graph."""
    print("\nTesting complex graph...")

    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)

    # z = (x + y) * (x * y)
    z = (x + y) * (x * y)
    z.backward()

    # Analytical: dz/dx = y*(x + y) + x*y = y*x + y^2 + x*y = 2xy + y^2
    # At x=2, y=3: 2*2*3 + 3^2 = 12 + 9 = 21
    expected_x = 21.0
    # dz/dy = x*(x + y) + x*y = x^2 + xy + xy = x^2 + 2xy
    # At x=2, y=3: 2^2 + 2*2*3 = 4 + 12 = 16
    expected_y = 16.0

    assert np.allclose(x.grad, expected_x), f"Expected {expected_x}, got {x.grad}"
    assert np.allclose(y.grad, expected_y), f"Expected {expected_y}, got {y.grad}"
    print("✓ Complex graph")

if __name__ == '__main__':
    test_basic_operations()
    test_matmul()
    test_activations()
    test_complex_graph()
    print("\n✅ All tests passed!")
```

---

## Part 3: Neural Network Layers

### Overview
Build reusable layer modules that automatically handle forward and backward passes.

### Step 1: Base Module Class

```python
# nn/module.py
from typing import Iterator
from tensor import Tensor

class Module:
    """
    Base class for all neural network modules.

    Your models should subclass this class.
    Modules can contain other modules, allowing complex architectures.
    """

    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def forward(self, *args, **kwargs):
        """
        Define the forward computation.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Make module callable."""
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Tensor]:
        """
        Return an iterator over module parameters.
        Typically used for optimizer.
        """
        for param in self._parameters.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()

    def register_parameter(self, name: str, param: Tensor):
        """Add a parameter to the module."""
        self._parameters[name] = param
        setattr(self, name, param)

    def register_module(self, name: str, module: 'Module'):
        """Add a child module to the module."""
        self._modules[name] = module
        setattr(self, name, module)

    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode=True):
        """Set module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Set module in evaluation mode."""
        return self.train(False)
```

### Step 2: Linear Layer

```python
# nn/linear.py
import numpy as np
from tensor import Tensor
from .module import Module

class Linear(Module):
    """
    Linear (fully connected) layer: y = xW^T + b

    Args:
        in_features: Size of input
        out_features: Size of output
        bias: If True, add learnable bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with He initialization
        k = np.sqrt(1 / in_features)
        weight_data = np.random.uniform(-k, k, (out_features, in_features))
        self.weight = Tensor(weight_data, requires_grad=True)
        self.register_parameter('weight', self.weight)

        if bias:
            bias_data = np.random.uniform(-k, k, (out_features,))
            self.bias = Tensor(bias_data, requires_grad=True)
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = xW^T + b

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # x: (batch, in_features)
        # weight: (out_features, in_features)
        # output: (batch, out_features)
        output = x @ self.weight.T

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'
```

### Step 3: Activation Layers

```python
# nn/activation.py
from tensor import Tensor
from .module import Module
import numpy as np

class ReLU(Module):
    """ReLU activation: f(x) = max(0, x)"""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Module):
    """Sigmoid activation: f(x) = 1 / (1 + e^(-x))"""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Tanh(Module):
    """Tanh activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

class Softmax(Module):
    """Softmax activation: softmax(x)_i = exp(x_i) / sum(exp(x_j))"""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        # Numerical stability: subtract max
        x_max = Tensor(x.data.max(axis=self.dim, keepdims=True))
        exp_x = (x + (-x_max)).exp()
        sum_exp = exp_x.sum(axis=self.dim, keepdims=True)
        return exp_x / sum_exp
```

### Step 4: Dropout Layer

```python
# nn/dropout.py
from tensor import Tensor
from .module import Module
import numpy as np

class Dropout(Module):
    """
    Dropout regularization.

    During training, randomly zeros some elements with probability p.
    During evaluation, multiplies by (1-p) for expectation matching.

    Args:
        p: Probability of an element to be zeroed
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        # Create dropout mask
        mask = np.random.binomial(1, 1-self.p, size=x.shape) / (1-self.p)
        mask_tensor = Tensor(mask)

        return x * mask_tensor
```

### Step 5: Batch Normalization

```python
# nn/batchnorm.py
from tensor import Tensor
from .module import Module
import numpy as np

class BatchNorm1d(Module):
    """
    Batch Normalization for 1D inputs.

    Normalizes input to have mean=0 and variance=1, then applies
    learnable affine transformation.

    Args:
        num_features: Number of features (C from (N, C) input)
        eps: Small constant for numerical stability
        momentum: Momentum for running statistics
    """

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.register_parameter('gamma', self.gamma)
        self.register_parameter('beta', self.beta)

        # Running statistics (not trainable)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input of shape (N, C)

        Returns:
            Normalized output of shape (N, C)
        """
        if self.training:
            # Compute batch statistics
            mean = x.mean(axis=0, keepdims=True)
            var = ((x + (-mean)) ** 2).mean(axis=0, keepdims=True)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * mean.data.flatten()
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * var.data.flatten()
        else:
            # Use running statistics
            mean = Tensor(self.running_mean.reshape(1, -1))
            var = Tensor(self.running_var.reshape(1, -1))

        # Normalize
        x_norm = (x + (-mean)) / ((var + self.eps) ** 0.5)

        # Scale and shift
        return self.gamma * x_norm + self.beta
```

---

## Part 4: Optimizers

### Overview
Implement optimization algorithms that update parameters based on gradients.

### Step 1: Base Optimizer

```python
# optim/optimizer.py
from typing import List
from tensor import Tensor

class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters: List[Tensor]):
        self.parameters = list(parameters)

    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        """
        Perform a single optimization step.
        Should be overridden by subclasses.
        """
        raise NotImplementedError
```

### Step 2: SGD Optimizer

```python
# optim/sgd.py
import numpy as np
from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: L2 penalty (default: 0)
    """

    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize velocity for momentum
        if momentum > 0:
            self.velocities = [np.zeros_like(p.data) for p in self.parameters]
        else:
            self.velocities = None

    def step(self):
        """Perform single optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Add weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Apply momentum
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]

            # Update parameters
            param.data -= self.lr * grad
```

### Step 3: Adam Optimizer

```python
# optim/adam.py
import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    """
    Adam optimizer.

    Combines momentum and RMSprop.

    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: L2 penalty (default: 0)
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        """Perform single optimization step."""
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # Add weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## Part 5: Loss Functions

### Step 1: Mean Squared Error

```python
# nn/loss.py
from tensor import Tensor
from .module import Module
import numpy as np

class MSELoss(Module):
    """
    Mean Squared Error loss.

    L = (1/N) * sum((pred - target)^2)
    """

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Scalar loss
        """
        diff = predictions + (-targets)
        squared = diff * diff
        return squared.mean()

class BCELoss(Module):
    """
    Binary Cross-Entropy loss.

    L = -1/N * sum(y*log(p) + (1-y)*log(1-p))
    """

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Clamp predictions for numerical stability
        pred_clamped = Tensor(np.clip(predictions.data, 1e-7, 1 - 1e-7))

        # BCE formula
        term1 = targets * pred_clamped.log()
        term2 = (Tensor(1.0) + (-targets)) * (Tensor(1.0) + (-pred_clamped)).log()
        return -((term1 + term2).mean())

class CrossEntropyLoss(Module):
    """
    Cross-Entropy loss for multi-class classification.

    Combines LogSoftmax and NLLLoss.
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: Raw model outputs of shape (batch, num_classes)
            targets: Class indices of shape (batch,)

        Returns:
            Scalar loss
        """
        batch_size = logits.shape[0]

        # LogSoftmax
        logits_max = Tensor(logits.data.max(axis=1, keepdims=True))
        logits_shifted = logits + (-logits_max)
        exp_logits = logits_shifted.exp()
        sum_exp = exp_logits.sum(axis=1, keepdims=True)
        log_softmax = logits_shifted + (-(sum_exp.log()))

        # Negative log-likelihood
        # Select log_softmax values for target classes
        batch_indices = np.arange(batch_size)
        target_indices = targets.data.astype(int).flatten()
        selected_log_probs = Tensor(log_softmax.data[batch_indices, target_indices])

        return -(selected_log_probs.mean())
```

---

## Part 6: Data Loading

### Step 1: Dataset Class

```python
# utils/data.py
import numpy as np
from typing import Tuple

class Dataset:
    """
    Abstract dataset class.

    All custom datasets should subclass this and override __len__ and __getitem__.
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

class TensorDataset(Dataset):
    """
    Simple dataset wrapping tensors.

    Args:
        *tensors: Tensors with same first dimension
    """

    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors), \
            "All tensors must have the same first dimension"
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        return tuple(t.data[index] for t in self.tensors)
```

### Step 2: DataLoader

```python
# utils/data.py (continued)

class DataLoader:
    """
    Data loader for batching and shuffling.

    Args:
        dataset: Dataset to load from
        batch_size: How many samples per batch
        shuffle: Whether to shuffle data at every epoch
    """

    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            np.random.shuffle(indices)

        # Generate batches
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_indices = indices[start:end]

            # Collect batch
            batch = [self.dataset[i] for i in batch_indices]

            # Stack into arrays
            if len(batch[0]) == 2:  # (data, target)
                data = np.stack([item[0] for item in batch])
                targets = np.stack([item[1] for item in batch])
                yield data, targets
            else:
                yield tuple(np.stack([item[i] for item in batch])
                          for i in range(len(batch[0])))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
```

---

## Part 7: Complete Examples

### Example 1: MNIST Classification

```python
# examples/mnist_classification.py
import numpy as np
from tensor import Tensor
import tinytorch.nn as nn
from tinytorch.optim import Adam
from tinytorch.utils.data import TensorDataset, DataLoader

# Define model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

# Load MNIST data (you need to load this separately)
def load_mnist():
    # Placeholder - implement actual MNIST loading
    # Returns (train_data, train_labels), (test_data, test_labels)
    pass

# Training
def train():
    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Create datasets
    train_dataset = TensorDataset(
        Tensor(X_train.reshape(-1, 784)),
        Tensor(y_train)
    )
    test_dataset = TensorDataset(
        Tensor(X_test.reshape(-1, 784)),
        Tensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = MNISTNet()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = Tensor(data, requires_grad=False)
            target = Tensor(target, requires_grad=False)

            # Forward
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.data.item():.6f}')

        # Validation
        model.eval()
        correct = 0
        total = 0

        for data, target in test_loader:
            data = Tensor(data, requires_grad=False)
            output = model(data)
            predictions = np.argmax(output.data, axis=1)
            correct += (predictions == target).sum()
            total += target.shape[0]

        accuracy = correct / total
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    train()
```

### Example 2: Simple Regression

```python
# examples/simple_regression.py
import numpy as np
import matplotlib.pyplot as plt
from tensor import Tensor
import tinytorch.nn as nn
from tinytorch.optim import SGD

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = 2 * X**2 + 3 * X + 1 + np.random.randn(200, 1) * 2

# Define model
class PolynomialRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

# Training
model = PolynomialRegression()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()

X_tensor = Tensor(X, requires_grad=False)
y_tensor = Tensor(y, requires_grad=False)

losses = []
for epoch in range(500):
    # Forward
    optimizer.zero_grad()
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    # Backward
    loss.backward()
    optimizer.step()

    losses.append(loss.data.item())

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.data.item():.4f}')

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
model.eval()
predictions = model(X_tensor)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, predictions.data, color='red', linewidth=2, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression Results')
plt.legend()

plt.tight_layout()
plt.savefig('regression_results.png')
plt.show()
```

---

## Success Criteria

You've successfully completed the capstone when:

1. ✅ All gradient checks pass with numerical gradients
2. ✅ Your framework can train a neural network on MNIST
3. ✅ Achieves >95% accuracy on MNIST test set
4. ✅ Code is well-documented with docstrings
5. ✅ Includes examples and README
6. ✅ Handles edge cases (broadcasting, different shapes, etc.)

## Next Steps

- Add GPU support using CuPy
- Implement convolutional layers
- Add more optimizers (RMSprop, AdaGrad)
- Implement learning rate schedulers
- Add model serialization (save/load)
- Create visualization tools for computational graphs

## Resources

- [Karpathy's Micrograd](https://github.com/karpathy/micrograd)
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [CS231n Backpropagation Notes](http://cs231n.github.io/optimization-2/)
- [Automatic Differentiation in ML](https://arxiv.org/abs/1502.05767)
