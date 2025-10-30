# Module 12: Optimization Deep Dive - Complete Coding Guide

This guide provides comprehensive implementations for advanced optimization techniques used in deep learning, including learning rate schedules, gradient clipping, and loss landscape visualization.

## Table of Contents
1. [Learning Rate Schedules](#part-1-learning-rate-schedules)
2. [Learning Rate Warmup](#part-2-learning-rate-warmup)
3. [Loss Landscape Visualization](#part-3-loss-landscape-visualization)
4. [Gradient Clipping](#part-4-gradient-clipping)
5. [Complete Training Framework](#part-5-complete-training-framework)

---

## Part 1: Learning Rate Schedules

### Overview
Learning rate schedules adjust the learning rate during training to improve convergence and final performance.

### Step 1: Step Decay

```python
import numpy as np
import matplotlib.pyplot as plt

class StepDecay:
    """
    Reduce learning rate by a factor every few epochs.
    LR = initial_lr * drop_rate^floor(epoch / epochs_drop)
    """

    def __init__(self, initial_lr=0.1, drop_rate=0.5, epochs_drop=10):
        self.initial_lr = initial_lr
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop

    def get_lr(self, epoch):
        """Calculate learning rate for given epoch."""
        lr = self.initial_lr * (self.drop_rate ** np.floor(epoch / self.epochs_drop))
        return lr

    def plot(self, epochs=100):
        """Visualize the schedule."""
        lrs = [self.get_lr(e) for e in range(epochs)]
        plt.figure(figsize=(10, 4))
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Step Decay (drop={self.drop_rate}, every {self.epochs_drop} epochs)')
        plt.grid(True, alpha=0.3)
        plt.show()

# Test
scheduler = StepDecay(initial_lr=0.1, drop_rate=0.5, epochs_drop=10)
scheduler.plot(epochs=50)
```

### Step 2: Exponential Decay

```python
class ExponentialDecay:
    """
    Exponentially decay learning rate.
    LR = initial_lr * exp(-decay_rate * epoch)
    """

    def __init__(self, initial_lr=0.1, decay_rate=0.05):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, epoch):
        """Calculate learning rate for given epoch."""
        lr = self.initial_lr * np.exp(-self.decay_rate * epoch)
        return lr

    def plot(self, epochs=100):
        """Visualize the schedule."""
        lrs = [self.get_lr(e) for e in range(epochs)]
        plt.figure(figsize=(10, 4))
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Exponential Decay (rate={self.decay_rate})')
        plt.grid(True, alpha=0.3)
        plt.show()

# Test
scheduler = ExponentialDecay(initial_lr=0.1, decay_rate=0.05)
scheduler.plot(epochs=100)
```

### Step 3: Cosine Annealing

```python
class CosineAnnealingScheduler:
    """
    Cosine annealing learning rate schedule.
    LR follows a cosine curve from initial_lr to min_lr.

    Paper: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    """

    def __init__(self, initial_lr=0.1, min_lr=0.0, T_max=100):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.T_max = T_max

    def get_lr(self, epoch):
        """Calculate learning rate for given epoch."""
        lr = self.min_lr + (self.initial_lr - self.min_lr) * \
             (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        return lr

    def plot(self, epochs=100):
        """Visualize the schedule."""
        lrs = [self.get_lr(e) for e in range(epochs)]
        plt.figure(figsize=(10, 4))
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Cosine Annealing (T_max={self.T_max})')
        plt.grid(True, alpha=0.3)
        plt.show()

# Test
scheduler = CosineAnnealingScheduler(initial_lr=0.1, min_lr=0.001, T_max=100)
scheduler.plot(epochs=100)
```

### Step 4: Cosine Annealing with Warm Restarts

```python
class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with periodic warm restarts.

    T_0: Number of epochs for the first restart
    T_mult: Factor to increase T_i after each restart
    """

    def __init__(self, initial_lr=0.1, min_lr=0.0, T_0=10, T_mult=2):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult

    def get_lr(self, epoch):
        """Calculate learning rate for given epoch."""
        T_cur = epoch
        T_i = self.T_0

        # Find which restart period we're in
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult

        # Cosine annealing within this period
        lr = self.min_lr + (self.initial_lr - self.min_lr) * \
             (1 + np.cos(np.pi * T_cur / T_i)) / 2
        return lr

    def plot(self, epochs=100):
        """Visualize the schedule."""
        lrs = [self.get_lr(e) for e in range(epochs)]
        plt.figure(figsize=(12, 4))
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Cosine Annealing with Warm Restarts (T_0={self.T_0}, T_mult={self.T_mult})')
        plt.grid(True, alpha=0.3)
        plt.show()

# Test
scheduler = CosineAnnealingWarmRestarts(initial_lr=0.1, min_lr=0.001, T_0=10, T_mult=2)
scheduler.plot(epochs=100)
```

### Step 5: Compare All Schedules

```python
def compare_schedules(epochs=100):
    """Compare all learning rate schedules."""
    schedulers = {
        'Step Decay': StepDecay(initial_lr=0.1, drop_rate=0.5, epochs_drop=20),
        'Exponential': ExponentialDecay(initial_lr=0.1, decay_rate=0.03),
        'Cosine Annealing': CosineAnnealingScheduler(initial_lr=0.1, min_lr=0.001, T_max=epochs),
        'Cosine with Restarts': CosineAnnealingWarmRestarts(initial_lr=0.1, min_lr=0.001, T_0=20, T_mult=2)
    }

    plt.figure(figsize=(14, 6))

    for name, scheduler in schedulers.items():
        lrs = [scheduler.get_lr(e) for e in range(epochs)]
        plt.plot(lrs, label=name, linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Comparison of Learning Rate Schedules', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('lr_schedules_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_schedules(epochs=100)
```

---

## Part 2: Learning Rate Warmup

### Overview
Learning rate warmup gradually increases the learning rate from a small value to the initial learning rate over the first few epochs. This helps stabilize training, especially for large batch sizes.

### Step 1: Linear Warmup

```python
class LinearWarmup:
    """
    Linear learning rate warmup.
    Linearly increase LR from 0 to target_lr over warmup_steps.
    """

    def __init__(self, target_lr=0.1, warmup_steps=1000):
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def get_lr(self, step):
        """Calculate learning rate for given step."""
        if step < self.warmup_steps:
            lr = self.target_lr * (step / self.warmup_steps)
        else:
            lr = self.target_lr
        return lr

    def plot(self, total_steps=5000):
        """Visualize the warmup."""
        lrs = [self.get_lr(s) for s in range(total_steps)]
        plt.figure(figsize=(10, 4))
        plt.plot(lrs)
        plt.axvline(x=self.warmup_steps, color='r', linestyle='--',
                   label=f'Warmup end ({self.warmup_steps} steps)')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title(f'Linear Warmup to {self.target_lr}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Test
warmup = LinearWarmup(target_lr=0.001, warmup_steps=1000)
warmup.plot(total_steps=5000)
```

### Step 2: Warmup + Cosine Decay

```python
class WarmupCosineSchedule:
    """
    Combine warmup with cosine annealing.
    Commonly used in transformer training (BERT, GPT, etc.)
    """

    def __init__(self, warmup_steps=1000, total_steps=10000,
                 max_lr=0.001, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, step):
        """Calculate learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 0.5 * (1.0 + np.cos(np.pi * progress))
        return lr

    def plot(self):
        """Visualize the complete schedule."""
        lrs = [self.get_lr(s) for s in range(self.total_steps)]
        plt.figure(figsize=(12, 4))
        plt.plot(lrs, linewidth=2)
        plt.axvline(x=self.warmup_steps, color='r', linestyle='--',
                   label=f'Warmup end', alpha=0.7)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Warmup + Cosine Decay Schedule', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Test
scheduler = WarmupCosineSchedule(warmup_steps=1000, total_steps=10000,
                                max_lr=0.001, min_lr=1e-5)
scheduler.plot()
```

### Step 3: Warmup Implementation in Optimizer

```python
import torch
import torch.nn as nn
import torch.optim as optim

class WarmupOptimizer:
    """
    Wrapper around optimizer that implements warmup.
    """

    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        """Take optimization step with warmup LR."""
        self.current_step += 1

        # Calculate warmup LR
        if self.current_step <= self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            lr = self.max_lr

        # Update LR in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Take optimization step
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# Example usage
model = nn.Linear(10, 1)
base_optimizer = optim.Adam(model.parameters())
optimizer = WarmupOptimizer(base_optimizer, warmup_steps=1000, max_lr=0.001)

# Track LRs during training
lrs = []
for step in range(5000):
    optimizer.zero_grad()
    # ... forward pass, backward pass ...
    optimizer.step()
    lrs.append(optimizer.get_lr())

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate with Warmup During Training')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Part 3: Loss Landscape Visualization

### Overview
Visualizing the loss landscape helps understand optimization difficulties and the effect of different architectures and optimizers.

### Step 1: 1D Loss Landscape

```python
def visualize_loss_1d(model, loss_fn, data, targets, param_name, param_index=0,
                     alpha_range=(-1, 1), num_points=100):
    """
    Visualize loss landscape along a single direction.

    Args:
        model: Neural network model
        loss_fn: Loss function
        data: Input data
        targets: Target labels
        param_name: Name of parameter to vary (e.g., 'weight' or 'bias')
        param_index: Index in flattened parameter vector
        alpha_range: Range of perturbation
        num_points: Number of points to sample
    """
    # Get the parameter
    param = None
    for name, p in model.named_parameters():
        if param_name in name:
            param = p
            break

    if param is None:
        raise ValueError(f"Parameter {param_name} not found")

    # Save original value
    original_param = param.data.clone()
    flat_param = original_param.flatten()

    # Create perturbation direction (unit vector)
    direction = torch.zeros_like(flat_param)
    direction[param_index] = 1.0

    # Compute losses along this direction
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = []

    for alpha in alphas:
        # Perturb parameter
        perturbed = flat_param + alpha * direction
        param.data = perturbed.reshape(original_param.shape)

        # Compute loss
        with torch.no_grad():
            output = model(data)
            loss = loss_fn(output, targets)
            losses.append(loss.item())

    # Restore original parameter
    param.data = original_param

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, losses, linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', label='Current parameters', alpha=0.7)
    plt.xlabel('Perturbation (α)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'1D Loss Landscape ({param_name}[{param_index}])', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return alphas, losses

# Example usage
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Create simple dataset
X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

loss_fn = nn.BCEWithLogitsLoss()

# Visualize
visualize_loss_1d(model, loss_fn, X, y, param_name='0.weight', param_index=0)
```

### Step 2: 2D Loss Landscape

```python
def visualize_loss_2d(model, loss_fn, data, targets, param_name,
                     indices=(0, 1), alpha_range=(-1, 1), num_points=50):
    """
    Visualize loss landscape in 2D (two parameters).

    Args:
        model: Neural network model
        loss_fn: Loss function
        data: Input data
        targets: Target labels
        param_name: Name of parameter to vary
        indices: Tuple of two parameter indices
        alpha_range: Range of perturbation
        num_points: Number of points along each axis
    """
    # Get the parameter
    param = None
    for name, p in model.named_parameters():
        if param_name in name:
            param = p
            break

    if param is None:
        raise ValueError(f"Parameter {param_name} not found")

    # Save original value
    original_param = param.data.clone()
    flat_param = original_param.flatten()

    # Create two perpendicular perturbation directions
    direction1 = torch.zeros_like(flat_param)
    direction1[indices[0]] = 1.0

    direction2 = torch.zeros_like(flat_param)
    direction2[indices[1]] = 1.0

    # Compute losses on a 2D grid
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    losses = np.zeros((num_points, num_points))

    for i, alpha1 in enumerate(alphas):
        for j, alpha2 in enumerate(alphas):
            # Perturb parameter
            perturbed = flat_param + alpha1 * direction1 + alpha2 * direction2
            param.data = perturbed.reshape(original_param.shape)

            # Compute loss
            with torch.no_grad():
                output = model(data)
                loss = loss_fn(output, targets)
                losses[i, j] = loss.item()

    # Restore original parameter
    param.data = original_param

    # Plot
    fig = plt.figure(figsize=(14, 5))

    # Contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(alphas, alphas, losses, levels=20, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(0, 0, 'r*', markersize=15, label='Current parameters')
    ax1.set_xlabel(f'Perturbation α₁ (param[{indices[0]}])', fontsize=11)
    ax1.set_ylabel(f'Perturbation α₂ (param[{indices[1]}])', fontsize=11)
    ax1.set_title('Loss Contours', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(alphas, alphas)
    surf = ax2.plot_surface(X, Y, losses, cmap='viridis', alpha=0.8)
    ax2.set_xlabel(f'α₁', fontsize=10)
    ax2.set_ylabel(f'α₂', fontsize=10)
    ax2.set_zlabel('Loss', fontsize=10)
    ax2.set_title('Loss Surface', fontsize=12)
    fig.colorbar(surf, ax=ax2, shrink=0.5)

    plt.tight_layout()
    plt.show()

    return alphas, losses

# Example usage
from mpl_toolkits.mplot3d import Axes3D

visualize_loss_2d(model, loss_fn, X, y, param_name='0.weight',
                 indices=(0, 1), alpha_range=(-0.5, 0.5), num_points=30)
```

### Step 3: Random Direction Visualization

```python
def visualize_loss_random_directions(model, loss_fn, data, targets,
                                    num_directions=5, alpha_range=(-1, 1),
                                    num_points=100):
    """
    Visualize loss along multiple random directions.
    Helps understand loss landscape sharpness.
    """
    # Save original parameters
    original_params = [p.data.clone() for p in model.parameters()]

    plt.figure(figsize=(12, 6))

    for direction_idx in range(num_directions):
        # Generate random direction (normalized)
        random_direction = []
        for param in model.parameters():
            direction = torch.randn_like(param)
            direction = direction / direction.norm()
            random_direction.append(direction)

        # Compute losses along this direction
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
        losses = []

        for alpha in alphas:
            # Perturb all parameters
            for param, orig_param, direction in zip(model.parameters(),
                                                    original_params,
                                                    random_direction):
                param.data = orig_param + alpha * direction

            # Compute loss
            with torch.no_grad():
                output = model(data)
                loss = loss_fn(output, targets)
                losses.append(loss.item())

        plt.plot(alphas, losses, alpha=0.7, label=f'Direction {direction_idx+1}')

    # Restore original parameters
    for param, orig_param in zip(model.parameters(), original_params):
        param.data = orig_param

    plt.axvline(x=0, color='r', linestyle='--', label='Current params', linewidth=2)
    plt.xlabel('Step size (α)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Landscape Along Random Directions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Test
visualize_loss_random_directions(model, loss_fn, X, y, num_directions=10)
```

---

## Part 4: Gradient Clipping

### Overview
Gradient clipping prevents exploding gradients by limiting the magnitude of gradients during backpropagation.

### Step 1: Clip by Value

```python
def clip_grad_value_(parameters, clip_value):
    """
    Clip gradients by value.
    Clamps each gradient element between -clip_value and +clip_value.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    clip_value = float(clip_value)

    for p in parameters:
        if p.grad is not None:
            p.grad.data.clamp_(-clip_value, clip_value)

# Example
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy forward/backward
x = torch.randn(32, 10)
y = torch.randn(32, 1)
loss = ((model(x) - y) ** 2).mean()
loss.backward()

print("Before clipping:")
print(f"Max gradient: {max(p.grad.abs().max() for p in model.parameters() if p.grad is not None)}")

clip_grad_value_(model.parameters(), clip_value=0.5)

print("After clipping:")
print(f"Max gradient: {max(p.grad.abs().max() for p in model.parameters() if p.grad is not None)}")
```

### Step 2: Clip by Norm

```python
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """
    Clip gradients by global norm.
    Scales down all gradients if their combined norm exceeds max_norm.

    Args:
        parameters: Iterable of parameters
        max_norm: Maximum norm value
        norm_type: Type of norm (2 for L2, inf for infinity norm)

    Returns:
        Total norm of the parameters (before clipping)
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    if norm_type == float('inf'):
        # Infinity norm
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        # L2 norm (or other)
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.data, norm_type) for p in parameters]),
            norm_type
        )

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm

# Example
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create exploding gradient scenario
x = torch.randn(32, 10) * 100
y = torch.randn(32, 1)
loss = ((model(x) - y) ** 2).mean()
loss.backward()

print("Before clipping:")
total_norm = clip_grad_norm_(model.parameters(), max_norm=float('inf'))
print(f"Total gradient norm: {total_norm:.2f}")

# Clip
loss.backward()  # Re-compute gradients
clipped_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"\nAfter clipping to max_norm=1.0:")
print(f"Total gradient norm: {clipped_norm:.2f}")
```

### Step 3: Track Gradient Norms During Training

```python
def train_with_gradient_tracking(model, train_loader, optimizer, epochs=10,
                                max_grad_norm=1.0):
    """
    Train model while tracking gradient norms.
    """
    grad_norms = []
    losses = []

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward
            output = model(data)
            loss = F.cross_entropy(output, target)

            # Backward
            loss.backward()

            # Track gradient norm BEFORE clipping
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0)
                           for p in model.parameters() if p.grad is not None]),
                2.0
            ).item()
            grad_norms.append(total_norm)
            losses.append(loss.item())

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update
            optimizer.step()

    return grad_norms, losses

# Visualize gradient norms
def plot_gradient_norms(grad_norms, losses):
    """Plot gradient norms and losses."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Gradient norms
    ax1.plot(grad_norms, alpha=0.6)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Norms During Training')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Losses
    ax2.plot(losses, alpha=0.6)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss During Training')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

---

## Part 5: Complete Training Framework

### Overview
Putting it all together: a complete training framework with all optimization techniques.

### Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class AdvancedTrainer:
    """
    Complete training framework with:
    - Learning rate scheduling
    - Warmup
    - Gradient clipping
    - Loss tracking and visualization
    """

    def __init__(self, model, optimizer, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []

    def train_epoch(self, train_loader, epoch, scheduler=None,
                   max_grad_norm=None, log_interval=100):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # Backward
            loss.backward()

            # Track gradient norm
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
                self.grad_norms.append(grad_norm.item())

            # Update
            self.optimizer.step()

            # Update learning rate (step-wise scheduler)
            if scheduler is not None and hasattr(scheduler, 'step'):
                scheduler.step()

            # Track
            total_loss += loss.item()
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # Log
            if batch_idx % log_interval == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f} '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.loss_fn(output, target)
            total_loss += loss.item()

            # Accuracy (for classification)
            if len(target.shape) == 1 or target.shape[1] == 1:
                pred = output.argmax(dim=1, keepdim=True) if output.shape[1] > 1 else \
                       (output > 0).float()
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0

        self.val_losses.append(avg_loss)
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs, scheduler=None,
             max_grad_norm=1.0):
        """Complete training loop."""
        print(f"Training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Max gradient norm: {max_grad_norm}")
        print("-" * 70)

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(
                train_loader, epoch, scheduler, max_grad_norm
            )

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 70)

            # Update epoch-wise scheduler
            if scheduler is not None and hasattr(scheduler, 'step') and \
               not hasattr(scheduler, 'get_last_lr'):
                scheduler.step()

    def plot_training_curves(self):
        """Visualize training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[0, 1].plot(self.learning_rates, linewidth=1)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)

        # Gradient norms
        if len(self.grad_norms) > 0:
            axes[1, 0].plot(self.grad_norms, alpha=0.6)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # Loss (log scale)
        axes[1, 1].semilogy(self.train_losses, label='Train', linewidth=2)
        axes[1, 1].semilogy(self.val_losses, label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].set_title('Loss (Log Scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

# Complete example
def complete_training_example():
    """Complete example with all features."""

    # Model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )

    # Data (dummy MNIST-like)
    train_data = TensorDataset(
        torch.randn(1000, 784),
        torch.randint(0, 10, (1000,))
    )
    val_data = TensorDataset(
        torch.randn(200, 784),
        torch.randint(0, 10, (200,))
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Scheduler: Warmup + Cosine
    total_steps = len(train_loader) * 20  # 20 epochs
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / warmup_steps,
            0.5 * (1.0 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        )
    )

    # Trainer
    trainer = AdvancedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device='cpu'
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        scheduler=scheduler,
        max_grad_norm=1.0
    )

    # Visualize
    trainer.plot_training_curves()

# Run
complete_training_example()
```

---

## Success Criteria

You've successfully completed Module 12 when you can:

1. ✅ Implement and compare different LR schedules (step, exponential, cosine)
2. ✅ Add warmup to any learning rate schedule
3. ✅ Visualize loss landscapes in 1D and 2D
4. ✅ Implement gradient clipping (by value and by norm)
5. ✅ Build a complete training framework with all optimization features
6. ✅ Train models with proper hyperparameter scheduling

## Next Steps

- Module 13: Explore regularization techniques
- Experiment with different scheduler combinations
- Try different warmup strategies
- Analyze loss landscapes for different architectures

## Additional Resources

- [Cyclical Learning Rates paper](https://arxiv.org/abs/1506.01186)
- [Visualizing Loss Landscapes](https://arxiv.org/abs/1712.09913)
- [PyTorch LR Scheduler Documentation](https://pytorch.org/docs/stable/optim.html)
