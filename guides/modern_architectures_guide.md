# Module 14: Modern Architectures - Complete Guide

This guide provides implementations and mathematical derivations for modern deep learning architectures including Vision Transformers, Diffusion Models, VAEs, and GANs.

## Table of Contents
1. [Vision Transformer (ViT)](#part-1-vision-transformer)
2. [Variational Autoencoders (VAE)](#part-2-variational-autoencoders)
3. [Generative Adversarial Networks (GAN)](#part-3-generative-adversarial-networks)
4. [Diffusion Models](#part-4-diffusion-models)

---

## Part 1: Vision Transformer (ViT)

### Overview
Vision Transformers apply the transformer architecture directly to images by treating image patches as tokens.

**Paper**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)

### Step 1: Patch Embedding

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.

    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Use convolution to extract patches and embed them
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - Batch of images

        Returns:
            (B, n_patches, embed_dim) - Patch embeddings
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

# Test
patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
x = torch.randn(2, 3, 224, 224)
patches = patch_embed(x)
print(f"Input shape: {x.shape}")
print(f"Patches shape: {patches.shape}")  # (2, 196, 768)
print(f"Number of patches: {patch_embed.n_patches}")  # 196 = 14*14
```

### Step 2: Position Embedding

```python
class PositionEmbedding(nn.Module):
    """
    Add learnable position embeddings to patch embeddings.
    """

    def __init__(self, n_patches, embed_dim, dropout=0.1):
        super().__init__()
        # +1 for class token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Initialize with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, n_patches+1, embed_dim)

        Returns:
            (B, n_patches+1, embed_dim)
        """
        x = x + self.pos_embed
        x = self.dropout(x)
        return x
```

### Step 3: Multi-Head Self-Attention (for ViT)

```python
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim) where N = n_patches + 1

        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*embed_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x
```

### Step 4: Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    Transformer encoder block for ViT.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm formulation
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Step 5: Complete Vision Transformer

```python
class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT) implementation.

    Args:
        img_size: Input image size
        patch_size: Size of image patches
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position embedding
        self.pos_embed = PositionEmbedding(n_patches, embed_dim, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - Batch of images

        Returns:
            (B, num_classes) - Class logits
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Add position embedding
        x = self.pos_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classify using cls token
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (B, embed_dim)
        x = self.head(cls_token_final)  # (B, num_classes)

        return x

# Test
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

x = torch.randn(2, 3, 224, 224)
logits = model(x)
print(f"Output shape: {logits.shape}")  # (2, 1000)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params / 1e6:.2f}M")
```

---

## Part 2: Variational Autoencoders (VAE)

### Overview
VAEs learn a probabilistic latent representation of data by maximizing the Evidence Lower Bound (ELBO).

**Paper**: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)

### Step 1: Understanding ELBO

```python
"""
VAE Objective: Maximize ELBO (Evidence Lower Bound)

log p(x) >= ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

Where:
- p(x|z): Decoder (likelihood)
- q(z|x): Encoder (approximate posterior)
- p(z): Prior (typically N(0, I))

The two terms:
1. Reconstruction term: E_q[log p(x|z)]
2. KL divergence: KL(q(z|x) || p(z))

For Gaussian q(z|x) = N(μ, σ²) and prior p(z) = N(0, I):
KL(q||p) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
"""
```

### Step 2: Encoder (Recognition Network)

```python
class Encoder(nn.Module):
    """
    VAE Encoder: maps input x to latent distribution parameters (μ, log_σ²).
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim)

        Returns:
            mu: (B, latent_dim) - Mean of q(z|x)
            logvar: (B, latent_dim) - Log variance of q(z|x)
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

### Step 3: Reparameterization Trick

```python
def reparameterize(mu, logvar):
    """
    Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)

    This allows backpropagation through sampling.

    Args:
        mu: (B, latent_dim) - Mean
        logvar: (B, latent_dim) - Log variance

    Returns:
        z: (B, latent_dim) - Sampled latent vector
    """
    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)    # Sample from N(0, I)
    z = mu + eps * std
    return z
```

### Step 4: Decoder (Generative Network)

```python
class Decoder(nn.Module):
    """
    VAE Decoder: maps latent z to reconstruction of x.
    """

    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, output_dim) - Reconstructed input
        """
        h = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))  # Output in [0, 1]
        return x_recon
```

### Step 5: Complete VAE

```python
class VAE(nn.Module):
    """
    Complete Variational Autoencoder.
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through VAE.

        Args:
            x: (B, input_dim)

        Returns:
            x_recon: (B, input_dim) - Reconstructed input
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def sample(self, num_samples, device='cpu'):
        """
        Generate samples from the prior p(z).

        Args:
            num_samples: Number of samples to generate

        Returns:
            samples: (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(device)
        samples = self.decoder(z)
        return samples

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + β * KL divergence.

    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Mean of q(z|x)
        logvar: Log variance of q(z|x)
        beta: Weight for KL term (beta=1 for standard VAE)

    Returns:
        loss: Total loss
        recon_loss: Reconstruction term
        kl_loss: KL divergence term
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss

# Training loop
def train_vae(model, train_loader, optimizer, epoch, device='cpu'):
    """Train VAE for one epoch."""
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)

        optimizer.zero_grad()

        # Forward pass
        x_recon, mu, logvar = model(data)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(x_recon, data, mu, logvar)

        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item()/len(data):.4f} '
                  f'Recon: {recon_loss.item()/len(data):.4f} '
                  f'KL: {kl_loss.item()/len(data):.4f}')

    avg_loss = train_loss / len(train_loader.dataset)
    return avg_loss
```

### Step 6: Beta-VAE

```python
class BetaVAE(VAE):
    """
    β-VAE: Disentangled representation learning.

    Paper: "β-VAE: Learning Basic Visual Concepts with a Constrained
            Variational Framework" (Higgins et al., 2017)

    Beta > 1: Encourage disentanglement (stronger KL penalty)
    Beta < 1: Focus on reconstruction
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, beta=4.0):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.beta = beta

    def loss_function(self, x_recon, x, mu, logvar):
        """Loss with β weighting on KL term."""
        return vae_loss(x_recon, x, mu, logvar, beta=self.beta)
```

---

## Part 3: Generative Adversarial Networks (GAN)

### Overview
GANs consist of two networks: a Generator that creates fake data, and a Discriminator that tries to distinguish real from fake.

**Paper**: "Generative Adversarial Networks" (Goodfellow et al., 2014)

### Step 1: Generator

```python
class Generator(nn.Module):
    """
    GAN Generator: maps random noise z to fake samples.
    """

    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 4),

            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim) - Random noise

        Returns:
            fake_samples: (B, output_dim)
        """
        return self.model(z)
```

### Step 2: Discriminator

```python
class Discriminator(nn.Module):
    """
    GAN Discriminator: classifies samples as real or fake.
    """

    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of being real
        )

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) - Real or fake samples

        Returns:
            prob: (B, 1) - Probability of being real
        """
        return self.model(x)
```

### Step 3: GAN Training Loop

```python
def train_gan(generator, discriminator, train_loader, g_optimizer, d_optimizer,
              latent_dim=100, device='cpu'):
    """
    Train GAN for one epoch.

    Objective:
    - Discriminator: max log(D(x)) + log(1 - D(G(z)))
    - Generator: max log(D(G(z))) or equivalently min log(1 - D(G(z)))
    """
    generator.train()
    discriminator.train()

    criterion = nn.BCELoss()

    for batch_idx, (real_data, _) in enumerate(train_loader):
        batch_size = real_data.size(0)
        real_data = real_data.view(batch_size, -1).to(device)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ==================== Train Discriminator ====================
        d_optimizer.zero_grad()

        # Real samples
        real_output = discriminator(real_data)
        d_loss_real = criterion(real_output, real_labels)

        # Fake samples
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(z).detach()  # Detach to not train G
        fake_output = discriminator(fake_data)
        d_loss_fake = criterion(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # ==================== Train Generator ====================
        g_optimizer.zero_grad()

        # Generate fake samples
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(z)
        fake_output = discriminator(fake_data)

        # Generator loss: fool discriminator
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        g_optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}] '
                  f'D Loss: {d_loss.item():.4f} '
                  f'G Loss: {g_loss.item():.4f} '
                  f'D(x): {real_output.mean().item():.4f} '
                  f'D(G(z)): {fake_output.mean().item():.4f}')

# Initialize and train
latent_dim = 100
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

### Step 4: Deep Convolutional GAN (DCGAN)

```python
class DCGANGenerator(nn.Module):
    """
    Deep Convolutional GAN Generator.

    Paper: "Unsupervised Representation Learning with Deep Convolutional
            Generative Adversarial Networks" (Radford et al., 2015)
    """

    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (B, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # (B, feature_maps*8, 4, 4)

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # (B, feature_maps*4, 8, 8)

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # (B, feature_maps*2, 16, 16)

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # (B, feature_maps, 32, 32)

            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # (B, channels, 64, 64)
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator."""

    def __init__(self, channels=3, feature_maps=64):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (B, channels, 64, 64)
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, feature_maps, 32, 32)

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, feature_maps*2, 16, 16)

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, feature_maps*4, 8, 8)

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, feature_maps*8, 4, 4)

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # (B, 1, 1, 1)
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)
```

---

## Part 4: Diffusion Models

### Overview
Diffusion models learn to reverse a gradual noising process, enabling high-quality image generation.

**Paper**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

### Step 1: Forward Diffusion Process

```python
"""
Forward Process: Gradually add Gaussian noise to data

q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)

where β_t is the noise schedule.

Closed form (using reparameterization):
q(x_t | x_0) = N(x_t; √(ᾱ_t) * x_0, (1-ᾱ_t) * I)

where ᾱ_t = ∏_{s=1}^t (1-β_s)
"""

import numpy as np

class DiffusionSchedule:
    """
    Noise schedule for diffusion models.
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

def forward_diffusion_sample(x_0, t, schedule, device='cpu'):
    """
    Sample from q(x_t | x_0) using closed form.

    Args:
        x_0: Original image (B, C, H, W)
        t: Timestep (B,)
        schedule: DiffusionSchedule object

    Returns:
        x_t: Noisy image at timestep t
        noise: The noise that was added
    """
    noise = torch.randn_like(x_0).to(device)

    sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

    # x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
    x_t = sqrt_alphas_cumprod_t.to(device) * x_0 + \
          sqrt_one_minus_alphas_cumprod_t.to(device) * noise

    return x_t, noise
```

### Step 2: U-Net Denoising Model

```python
class SimpleUNet(nn.Module):
    """
    Simplified U-Net for denoising.
    Predicts noise ε given noisy image x_t and timestep t.
    """

    def __init__(self, in_channels=3, model_channels=64, num_res_blocks=2):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )

        # Encoder (downsampling)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, 3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, model_channels * 2),
            nn.SiLU(),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(model_channels * 2, model_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, model_channels * 4),
            nn.SiLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(model_channels * 4, model_channels * 4, 3, padding=1),
            nn.GroupNorm(8, model_channels * 4),
            nn.SiLU(),
        )

        # Decoder (upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(model_channels * 4, model_channels * 2, 2, stride=2),
            nn.GroupNorm(8, model_channels * 2),
            nn.SiLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(model_channels * 4, model_channels, 2, stride=2),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(model_channels * 2, model_channels, 3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
        )

        # Output
        self.out = nn.Conv2d(model_channels, in_channels, 1)

    def forward(self, x, t):
        """
        Args:
            x: (B, C, H, W) - Noisy image
            t: (B,) - Timestep

        Returns:
            noise_pred: (B, C, H, W) - Predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(self.get_timestep_embedding(t, x.device))

        # Encoder
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)

        # Bottleneck
        h = self.bottleneck(h3)

        # Decoder with skip connections
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)

        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)

        h = self.up3(h)
        noise_pred = self.out(h)

        return noise_pred

    def get_timestep_embedding(self, timesteps, device, dim=64):
        """Sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
```

### Step 3: Training Loop

```python
def train_diffusion(model, train_loader, optimizer, schedule, device='cpu'):
    """
    Train diffusion model for one epoch.

    Loss: E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]

    where:
    - x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
    - ε_θ predicts the noise
    """
    model.train()
    total_loss = 0

    for batch_idx, (x_0, _) in enumerate(train_loader):
        x_0 = x_0.to(device)
        batch_size = x_0.size(0)

        # Sample random timesteps
        t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)

        # Forward diffusion: add noise
        x_t, noise = forward_diffusion_sample(x_0, t, schedule, device)

        # Predict noise
        optimizer.zero_grad()
        noise_pred = model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')

    return total_loss / len(train_loader)
```

### Step 4: Sampling (Reverse Process)

```python
@torch.no_grad()
def sample_diffusion(model, schedule, image_shape=(1, 3, 64, 64), device='cpu'):
    """
    Sample from the diffusion model by reversing the diffusion process.

    Args:
        model: Trained denoising model
        schedule: DiffusionSchedule object
        image_shape: Shape of images to generate

    Returns:
        Generated images
    """
    model.eval()

    # Start from pure noise
    x_t = torch.randn(image_shape).to(device)

    # Reverse process: t = T, T-1, ..., 1
    for t in reversed(range(schedule.num_timesteps)):
        t_batch = torch.full((image_shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = model(x_t, t_batch)

        # Get schedule values
        alpha_t = schedule.alphas[t]
        alpha_cumprod_t = schedule.alphas_cumprod[t]
        beta_t = schedule.betas[t]

        # Compute x_{t-1}
        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        # Mean of p(x_{t-1} | x_t)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
        )

        # Variance
        variance = schedule.posterior_variance[t]

        # Sample x_{t-1}
        x_t = mean + torch.sqrt(variance) * noise

    return x_t

# Generate samples
generated_images = sample_diffusion(model, schedule, image_shape=(4, 3, 64, 64), device='cuda')
```

---

## Success Criteria

You've successfully completed Module 14 when you can:

1. ✅ Implement Vision Transformer (ViT) from scratch
2. ✅ Understand and derive the VAE ELBO
3. ✅ Build and train a GAN (both vanilla and DCGAN)
4. ✅ Understand the forward and reverse diffusion processes
5. ✅ Implement a simple diffusion model with U-Net denoising

## Next Steps

- Read and implement more recent architectures
- Explore Papers with Code for state-of-the-art models
- Study architectural innovations (skip connections, attention, normalization)
- Experiment with different generative model types

## Additional Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Understanding Diffusion Models](https://arxiv.org/abs/2208.11970)
- [GAN Lab (Interactive)](https://poloclub.github.io/ganlab/)
- [VAE Tutorial by Kingma](https://arxiv.org/abs/1906.02691)
- [Papers with Code](https://paperswithcode.com/)
