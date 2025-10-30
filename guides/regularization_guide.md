# Module 13: Regularization & Generalization - Complete Coding Guide

This guide provides comprehensive implementations for regularization techniques that improve model generalization and prevent overfitting.

## Table of Contents
1. [L1 and L2 Regularization](#part-1-l1-and-l2-regularization)
2. [Elastic Net Regularization](#part-2-elastic-net-regularization)
3. [Data Augmentation](#part-3-data-augmentation)
4. [Generalization Experiments](#part-4-generalization-experiments)

---

## Part 1: L1 and L2 Regularization

### Overview
Regularization adds a penalty term to the loss function to discourage complex models and prevent overfitting.

### Step 1: L2 Regularization (Weight Decay)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def l2_regularization_loss(model, lambda_reg=0.01):
    """
    Compute L2 regularization penalty.

    L2 penalty = λ * Σ(w²) for all weights w
    Also known as weight decay.
    """
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)

    return lambda_reg * l2_loss

# Example: Training with L2 regularization
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_with_l2(model, X_train, y_train, lambda_reg=0.01, epochs=100, lr=0.01):
    """Train model with L2 regularization."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    reg_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        data_loss = criterion(outputs, y_train)

        # Add L2 regularization
        reg_loss = l2_regularization_loss(model, lambda_reg)
        total_loss = data_loss + reg_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        losses.append(data_loss.item())
        reg_losses.append(reg_loss.item())

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Data Loss: {data_loss.item():.4f}, '
                  f'Reg Loss: {reg_loss.item():.4f}')

    return losses, reg_losses

# Generate synthetic data
np.random.seed(42)
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Train with different L2 strengths
lambda_values = [0.0, 0.001, 0.01, 0.1]
plt.figure(figsize=(14, 4))

for idx, lambda_reg in enumerate(lambda_values):
    model = SimpleModel(10, 20, 1)
    losses, _ = train_with_l2(model, X, y, lambda_reg=lambda_reg, epochs=200)

    plt.subplot(1, 4, idx + 1)
    plt.plot(losses)
    plt.title(f'λ = {lambda_reg}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('l2_regularization_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step 2: L2 Regularization Effect on Weights

```python
def compare_weight_magnitudes(lambda_values=[0.0, 0.01, 0.1, 1.0], epochs=500):
    """
    Compare weight magnitudes with different L2 regularization strengths.
    """
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    weight_norms = {lambda_reg: [] for lambda_reg in lambda_values}

    for lambda_reg in lambda_values:
        model = SimpleModel(10, 20, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            data_loss = criterion(outputs, y)
            reg_loss = l2_regularization_loss(model, lambda_reg)
            total_loss = data_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            # Track weight norm
            weight_norm = torch.sqrt(sum(
                torch.sum(p ** 2) for p in model.parameters()
            )).item()
            weight_norms[lambda_reg].append(weight_norm)

    # Plot
    plt.figure(figsize=(10, 6))
    for lambda_reg in lambda_values:
        plt.plot(weight_norms[lambda_reg], label=f'λ = {lambda_reg}', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Weight Norm', fontsize=12)
    plt.title('Effect of L2 Regularization on Weight Magnitudes', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

compare_weight_magnitudes()
```

### Step 3: L1 Regularization (Lasso)

```python
def l1_regularization_loss(model, lambda_reg=0.01):
    """
    Compute L1 regularization penalty.

    L1 penalty = λ * Σ|w| for all weights w
    Encourages sparsity (many weights become exactly 0).
    """
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))

    return lambda_reg * l1_loss

def train_with_l1(model, X_train, y_train, lambda_reg=0.01, epochs=100, lr=0.01):
    """Train model with L1 regularization."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        data_loss = criterion(outputs, y_train)

        # Add L1 regularization
        reg_loss = l1_regularization_loss(model, lambda_reg)
        total_loss = data_loss + reg_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        losses.append(data_loss.item())

    return losses

# Compare L1 sparsity
def analyze_sparsity(lambda_values=[0.0, 0.001, 0.01, 0.1]):
    """Analyze weight sparsity with different L1 strengths."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, lambda_reg in enumerate(lambda_values):
        model = SimpleModel(10, 50, 1)
        train_with_l1(model, X, y, lambda_reg=lambda_reg, epochs=500)

        # Get all weights
        weights = []
        for param in model.parameters():
            weights.extend(param.data.flatten().numpy())

        # Count near-zero weights
        near_zero = np.sum(np.abs(weights) < 0.01)
        total = len(weights)

        # Plot histogram
        axes[idx].hist(weights, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[idx].set_title(f'λ = {lambda_reg}\n'
                          f'Near-zero: {near_zero}/{total} ({100*near_zero/total:.1f}%)')
        axes[idx].set_xlabel('Weight Value')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('l1_sparsity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

analyze_sparsity()
```

### Step 4: Comparing L1 vs L2

```python
def compare_l1_l2_regularization():
    """
    Direct comparison of L1 and L2 regularization.
    """
    # Generate data with irrelevant features
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    n_informative = 5

    # Only first 5 features are informative
    X = np.random.randn(n_samples, n_features)
    true_weights = np.zeros(n_features)
    true_weights[:n_informative] = np.random.randn(n_informative)
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(1)

    # Train models
    models = {
        'No Regularization': SimpleModel(n_features, 30, 1),
        'L2 (λ=0.1)': SimpleModel(n_features, 30, 1),
        'L1 (λ=0.1)': SimpleModel(n_features, 30, 1),
    }

    # Train each model
    for name, model in models.items():
        if 'L2' in name:
            train_with_l2(model, X, y, lambda_reg=0.1, epochs=500, lr=0.01)
        elif 'L1' in name:
            train_with_l1(model, X, y, lambda_reg=0.1, epochs=500, lr=0.01)
        else:
            train_with_l2(model, X, y, lambda_reg=0.0, epochs=500, lr=0.01)

    # Visualize first layer weights
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, model) in enumerate(models.items()):
        weights = model.fc1.weight.data.numpy()
        im = axes[idx].imshow(weights, aspect='auto', cmap='RdBu_r',
                             vmin=-1, vmax=1)
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Input Feature')
        axes[idx].set_ylabel('Hidden Unit')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig('l1_vs_l2_weights.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_l1_l2_regularization()
```

---

## Part 2: Elastic Net Regularization

### Overview
Elastic Net combines L1 and L2 regularization to get benefits of both: sparsity from L1 and stability from L2.

### Step 1: Elastic Net Implementation

```python
def elastic_net_loss(model, lambda1=0.01, lambda2=0.01):
    """
    Elastic Net regularization = L1 + L2.

    Total penalty = λ1 * Σ|w| + λ2 * Σ(w²)

    Args:
        lambda1: L1 regularization strength
        lambda2: L2 regularization strength
    """
    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())

    return lambda1 * l1_loss + lambda2 * l2_loss

def train_with_elastic_net(model, X_train, y_train, lambda1=0.01, lambda2=0.01,
                          epochs=100, lr=0.01):
    """Train model with Elastic Net regularization."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        data_loss = criterion(outputs, y_train)

        # Add Elastic Net regularization
        reg_loss = elastic_net_loss(model, lambda1, lambda2)
        total_loss = data_loss + reg_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}')

    return losses

# Example
X = torch.randn(100, 20)
y = torch.randn(100, 1)

model = SimpleModel(20, 50, 1)
losses = train_with_elastic_net(model, X, y, lambda1=0.01, lambda2=0.01,
                                epochs=500, lr=0.01)

plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training with Elastic Net Regularization')
plt.grid(True, alpha=0.3)
plt.show()
```

### Step 2: Elastic Net Parameter Sweep

```python
def elastic_net_parameter_sweep():
    """
    Explore different combinations of λ1 and λ2.
    """
    X = torch.randn(100, 20)
    y = torch.randn(100, 1)

    lambda1_values = [0.0, 0.01, 0.1]
    lambda2_values = [0.0, 0.01, 0.1]

    fig, axes = plt.subplots(len(lambda1_values), len(lambda2_values),
                            figsize=(12, 12))

    for i, lambda1 in enumerate(lambda1_values):
        for j, lambda2 in enumerate(lambda2_values):
            model = SimpleModel(20, 50, 1)
            losses = train_with_elastic_net(model, X, y,
                                           lambda1=lambda1,
                                           lambda2=lambda2,
                                           epochs=300, lr=0.01)

            # Count near-zero weights
            weights = []
            for param in model.parameters():
                weights.extend(param.data.flatten().numpy())
            near_zero = np.sum(np.abs(weights) < 0.01)
            total = len(weights)

            axes[i, j].plot(losses, alpha=0.7)
            axes[i, j].set_title(f'λ1={lambda1}, λ2={lambda2}\n'
                                f'Sparsity: {100*near_zero/total:.1f}%',
                                fontsize=9)
            axes[i, j].grid(True, alpha=0.3)

            if i == len(lambda1_values) - 1:
                axes[i, j].set_xlabel('Epoch')
            if j == 0:
                axes[i, j].set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig('elastic_net_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()

elastic_net_parameter_sweep()
```

---

## Part 3: Data Augmentation

### Overview
Data augmentation artificially expands the training dataset by applying transformations that preserve the label but increase data diversity.

### Step 1: Image Augmentation

```python
import torchvision.transforms as transforms
from PIL import Image

class ImageAugmentation:
    """
    Common image augmentation techniques.
    """

    @staticmethod
    def get_train_transforms():
        """Training augmentations."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                  saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms():
        """Validation transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def visualize_augmentations(image_path, num_augmentations=9):
    """Visualize different augmentations of the same image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
    ])

    # Load original image
    original = Image.open(image_path) if isinstance(image_path, str) else image_path

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for idx in range(num_augmentations):
        augmented = transform(original)
        axes[idx].imshow(augmented)
        axes[idx].axis('off')
        axes[idx].set_title(f'Augmentation {idx+1}')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Step 2: Custom Augmentation Functions

```python
def cutout(image, n_holes=1, length=16):
    """
    Cutout augmentation: randomly mask out square patches.

    Paper: "Improved Regularization of CNNs with Cutout" (2017)
    """
    h, w = image.shape[1:3]
    mask = np.ones((h, w), dtype=np.float32)

    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(image)
    image = image * mask

    return image

def mixup(x, y, alpha=1.0):
    """
    Mixup augmentation: linearly interpolate between two samples.

    Paper: "mixup: Beyond Empirical Risk Minimization" (2018)

    Args:
        x: Batch of images
        y: Batch of labels
        alpha: Beta distribution parameter
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Loss function for mixup.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Example usage
def train_with_mixup(model, train_loader, optimizer, criterion, device='cpu'):
    """Training loop with mixup."""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Apply mixup
        data, targets_a, targets_b, lam = mixup(data, target, alpha=1.0)

        # Forward
        optimizer.zero_grad()
        output = model(data)

        # Mixup loss
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

### Step 3: Random Erasing

```python
class RandomErasing:
    """
    Random Erasing Data Augmentation.

    Paper: "Random Erasing Data Augmentation" (2017)
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl  # Min erasing area
        self.sh = sh  # Max erasing area
        self.r1 = r1  # Min aspect ratio

    def __call__(self, img):
        if np.random.random() > self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1/self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)

                img[0, x1:x1+h, y1:y1+w] = np.random.random()
                img[1, x1:x1+h, y1:y1+w] = np.random.random()
                img[2, x1:x1+h, y1:y1+w] = np.random.random()

                return img

        return img
```

---

## Part 4: Generalization Experiments

### Overview
Empirical experiments to understand and measure generalization.

### Step 1: Train/Val Curves with Different Regularization

```python
def compare_generalization(regularization_strengths=[0.0, 0.001, 0.01, 0.1]):
    """
    Compare generalization with different regularization strengths.
    """
    # Generate data with train/val split
    np.random.seed(42)
    X_train = torch.randn(500, 20)
    y_train = torch.randn(500, 1)
    X_val = torch.randn(100, 20)
    y_val = torch.randn(100, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, lambda_reg in enumerate(regularization_strengths):
        model = SimpleModel(20, 100, 1)  # Large model to enable overfitting
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in range(300):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            train_loss = criterion(outputs, y_train)
            reg_loss = l2_regularization_loss(model, lambda_reg)
            total_loss = train_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

        # Plot
        axes[idx].plot(train_losses, label='Train', linewidth=2)
        axes[idx].plot(val_losses, label='Validation', linewidth=2)
        axes[idx].set_title(f'λ = {lambda_reg}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

        # Calculate generalization gap
        final_gap = val_losses[-1] - train_losses[-1]
        axes[idx].text(0.05, 0.95, f'Final gap: {final_gap:.4f}',
                      transform=axes[idx].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('generalization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_generalization()
```

### Step 2: Model Complexity vs Generalization

```python
def analyze_model_complexity():
    """
    Analyze how model complexity affects generalization.
    """
    # Simple dataset
    np.random.seed(42)
    n_train = 100
    X_train = torch.randn(n_train, 10)
    y_train = torch.randn(n_train, 1)
    X_val = torch.randn(50, 10)
    y_val = torch.randn(50, 1)

    # Different model sizes
    hidden_sizes = [5, 10, 20, 50, 100, 200, 500]
    results = {'hidden_size': [], 'train_loss': [], 'val_loss': [],
               'gap': [], 'params': []}

    for hidden_size in hidden_sizes:
        model = SimpleModel(10, hidden_size, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train).item()
            val_loss = criterion(model(X_val), y_val).item()

        num_params = sum(p.numel() for p in model.parameters())

        results['hidden_size'].append(hidden_size)
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['gap'].append(val_loss - train_loss)
        results['params'].append(num_params)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs model size
    ax1.plot(results['hidden_size'], results['train_loss'],
            'o-', label='Train Loss', linewidth=2, markersize=8)
    ax1.plot(results['hidden_size'], results['val_loss'],
            's-', label='Val Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Hidden Layer Size', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Complexity vs Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Generalization gap vs parameters
    ax2.plot(results['params'], results['gap'],
            'o-', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Parameters', fontsize=12)
    ax2.set_ylabel('Generalization Gap', fontsize=12)
    ax2.set_title('Parameters vs Generalization Gap', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig('complexity_generalization.png', dpi=150, bbox_inches='tight')
    plt.show()

analyze_model_complexity()
```

### Step 3: Double Descent Visualization

```python
def visualize_double_descent():
    """
    Demonstrate the double descent phenomenon.

    Paper: "Deep Double Descent" (Nakkiran et al., 2019)
    """
    np.random.seed(42)

    # Small dataset
    n_samples = 50
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    X_train = torch.FloatTensor(X[:40])
    y_train = torch.FloatTensor(y[:40]).unsqueeze(1)
    X_test = torch.FloatTensor(X[40:])
    y_test = torch.FloatTensor(y[40:]).unsqueeze(1)

    # Vary model width
    widths = list(range(5, 300, 5))
    train_errors = []
    test_errors = []

    for width in widths:
        model = SimpleModel(n_features, width, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train to convergence
        for _ in range(1000):
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_error = criterion(model(X_train), y_train).item()
            test_error = criterion(model(X_test), y_test).item()

        train_errors.append(train_error)
        test_errors.append(test_error)

        print(f"Width: {width}, Train: {train_error:.4f}, Test: {test_error:.4f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(widths, train_errors, label='Train Error', linewidth=2)
    plt.plot(widths, test_errors, label='Test Error', linewidth=2)
    plt.axvline(x=n_samples, color='r', linestyle='--',
                label='Interpolation threshold', alpha=0.7)
    plt.xlabel('Model Width (Number of Parameters)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Double Descent Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('double_descent.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_double_descent()
```

### Step 4: Bias-Variance Tradeoff

```python
def bias_variance_decomposition():
    """
    Empirically estimate bias-variance tradeoff.
    """
    np.random.seed(42)

    # True function
    def true_function(x):
        return np.sin(x) + 0.1 * x**2

    # Training data
    n_train = 50
    X_train = np.random.uniform(-3, 3, n_train)
    y_train = true_function(X_train) + np.random.randn(n_train) * 0.3

    # Test points
    X_test = np.linspace(-3, 3, 100)
    y_true = true_function(X_test)

    # Different model complexities (polynomial degrees)
    degrees = [1, 2, 3, 5, 10, 20]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, degree in enumerate(degrees):
        # Train multiple models on different bootstrap samples
        n_models = 50
        predictions = []

        for _ in range(n_models):
            # Bootstrap sample
            indices = np.random.choice(n_train, n_train, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]

            # Fit polynomial
            coeffs = np.polyfit(X_boot, y_boot, degree)
            y_pred = np.polyval(coeffs, X_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)
        mean_prediction = predictions.mean(axis=0)
        std_prediction = predictions.std(axis=0)

        # Bias and variance
        bias_squared = ((mean_prediction - y_true) ** 2).mean()
        variance = (std_prediction ** 2).mean()

        # Plot
        axes[idx].scatter(X_train, y_train, alpha=0.5, s=20, label='Data')
        axes[idx].plot(X_test, y_true, 'g-', linewidth=2, label='True')
        axes[idx].plot(X_test, mean_prediction, 'r-', linewidth=2, label='Mean prediction')
        axes[idx].fill_between(X_test,
                              mean_prediction - 2*std_prediction,
                              mean_prediction + 2*std_prediction,
                              alpha=0.3, label='±2 std')

        axes[idx].set_title(f'Degree {degree}\n'
                          f'Bias²={bias_squared:.3f}, Var={variance:.3f}')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.show()

bias_variance_decomposition()
```

---

## Success Criteria

You've successfully completed Module 13 when you can:

1. ✅ Implement L1, L2, and Elastic Net regularization from scratch
2. ✅ Understand the effect of regularization on weight sparsity and magnitudes
3. ✅ Apply data augmentation techniques (image transformations, mixup, cutout)
4. ✅ Conduct generalization experiments comparing train/val curves
5. ✅ Visualize and understand the bias-variance tradeoff
6. ✅ Observe double descent phenomenon empirically

## Next Steps

- Module 14: Explore modern architectures (ViT, Diffusion Models, VAEs, GANs)
- Experiment with different regularization combinations
- Try advanced augmentation techniques (AutoAugment, RandAugment)
- Study PAC learning theory for theoretical foundations

## Additional Resources

- [Dropout as Bayesian Approximation](https://arxiv.org/abs/1506.02142)
- [Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)
- [Mixup paper](https://arxiv.org/abs/1710.09412)
