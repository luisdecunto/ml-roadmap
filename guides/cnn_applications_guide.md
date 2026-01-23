# Module 7.5: CNN Applications with PyTorch

Now that you understand CNNs from scratch, it's time to apply them efficiently using PyTorch to real-world problems beyond image classification.

## Prerequisites
- Completed Module 7 (CNNs from Scratch)
- PyTorch installed (`pip install torch torchvision torchaudio`)

## Learning Objectives
1. Use pre-built PyTorch CNN layers efficiently
2. Apply CNNs to diverse domains (audio, text, medical, time series)
3. Understand transfer learning and when to use it
4. Work with different data modalities

---

## Quick PyTorch CNN Refresher

```python
import torch
import torch.nn as nn

# Your from-scratch conv is now just:
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
pool = nn.MaxPool2d(kernel_size=2)
relu = nn.ReLU()

# A complete CNN block:
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)  # Stabilizes training
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))
```

---

## Key Architecture Concepts to Know

Before diving into projects, understand these concepts (links provided):

### 1. Batch Normalization
Normalizes layer inputs to stabilize training. Allows higher learning rates.
- [Original Paper](https://arxiv.org/abs/1502.03167)
- [Explained Simply](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)

### 2. Dropout
Randomly zeros neurons during training to prevent overfitting.
```python
self.dropout = nn.Dropout(p=0.5)  # 50% dropout
```

### 3. Global Average Pooling
Replaces large FC layers at the end. Reduces overfitting.
```python
# Instead of flattening 7x7x512 = 25088 features:
self.gap = nn.AdaptiveAvgPool2d(1)  # Output: (batch, channels, 1, 1)
```

### 4. Residual Connections (ResNet)
Skip connections that help train very deep networks.
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)  # Skip connection!
```

### 5. 1D Convolutions for Sequences
Same concept as 2D, but slides over one dimension (time/position).
```python
# For text: (batch, channels=embedding_dim, sequence_length)
conv1d = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
```

### 6. Transfer Learning
Use pre-trained weights from large datasets (ImageNet) as starting point.
```python
from torchvision import models
model = models.resnet18(pretrained=True)
# Freeze early layers, fine-tune later ones
```

---

## Projects Overview

| Project | Domain | CNN Type | Dataset | Key Concepts |
|---------|--------|----------|---------|--------------|
| 1 | Audio | 2D on spectrograms | UrbanSound8K | Mel spectrograms, data augmentation |
| 2 | Text | 1D CNN | IMDB Reviews | Word embeddings, 1D convolutions |
| 3 | Medical | 2D + Transfer | Chest X-rays | Transfer learning, class imbalance |
| 4 | Time Series | 1D CNN | ECG signals | Sliding windows, real-time inference |

---

## Project 1: Urban Sound Classification

**Goal**: Classify urban sounds (dog bark, siren, etc.) using spectrograms.

**Key Insight**: Audio can be converted to images (spectrograms) and processed with 2D CNNs.

### Theory: Spectrograms
A spectrogram shows frequency content over time:
- X-axis: Time
- Y-axis: Frequency
- Color: Intensity (loudness)

**Mel Spectrograms** use a perceptual frequency scale (how humans hear).

### Architecture
```
Input: Mel Spectrogram (1, 128, 128)
    ↓
Conv2d(1→32) + BN + ReLU + MaxPool → (32, 64, 64)
    ↓
Conv2d(32→64) + BN + ReLU + MaxPool → (64, 32, 32)
    ↓
Conv2d(64→128) + BN + ReLU + MaxPool → (128, 16, 16)
    ↓
Global Average Pooling → (128,)
    ↓
FC(128→10) → 10 classes
```

**File**: `practice/math_foundations/module07.5_cnn_applications/01_audio_classification.py`

---

## Project 2: Sentiment Analysis with 1D CNN

**Goal**: Classify movie reviews as positive/negative.

**Key Insight**: Text is a 1D sequence. Convolutions can capture n-gram patterns.

### Theory: Word Embeddings
Words are mapped to dense vectors where similar words have similar vectors.
- "king" - "man" + "woman" ≈ "queen"
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)

### How 1D CNN Works on Text
```
Input: "This movie was great"
    ↓
Embedding: Each word → 128-dim vector
    ↓
Shape: (4 words, 128 dims) → transpose → (128, 4)
    ↓
Conv1d with kernel_size=3 captures 3-word patterns
    ↓
"was great" pattern detected!
```

### Architecture
```
Input: Token IDs (sequence_length,)
    ↓
Embedding(vocab_size, 128) → (seq_len, 128)
    ↓
Transpose → (128, seq_len)
    ↓
Conv1d(128→64, k=3) + ReLU → captures 3-grams
Conv1d(128→64, k=4) + ReLU → captures 4-grams  } Parallel
Conv1d(128→64, k=5) + ReLU → captures 5-grams
    ↓
Global Max Pool each → (64,) × 3 = (192,)
    ↓
Concat + FC(192→2) → pos/neg
```

**File**: `practice/math_foundations/module07.5_cnn_applications/02_text_classification.py`

---

## Project 3: Chest X-Ray Classification

**Goal**: Detect pneumonia from chest X-rays.

**Key Insight**: Medical imaging benefits heavily from transfer learning due to limited labeled data.

### Theory: Transfer Learning
1. Take a model pre-trained on ImageNet (millions of images)
2. Early layers learn generic features (edges, textures)
3. Replace final layer for your task
4. Fine-tune with your data

### Architecture (Using ResNet18)
```
ResNet18 pretrained on ImageNet
    ↓
Remove final FC layer
    ↓
Add: FC(512→2) for binary classification
    ↓
Freeze early layers (optional)
    ↓
Fine-tune on chest X-rays
```

### Handling Class Imbalance
Medical datasets often have many more "normal" than "disease" samples.
```python
# Weighted loss
weights = torch.tensor([1.0, 3.0])  # 3x weight for disease class
criterion = nn.CrossEntropyLoss(weight=weights)
```

**File**: `practice/math_foundations/module07.5_cnn_applications/03_medical_imaging.py`

---

## Project 4: ECG Arrhythmia Detection

**Goal**: Detect heart arrhythmias from ECG signals.

**Key Insight**: ECG is 1D time series data - perfect for 1D CNNs.

### Theory: 1D CNNs for Time Series
- Kernel slides over time dimension
- Can detect temporal patterns (spikes, waves)
- Much faster than RNNs

### Architecture
```
Input: ECG signal (1, 187) - one heartbeat
    ↓
Conv1d(1→32, k=5) + BN + ReLU + MaxPool
    ↓
Conv1d(32→64, k=5) + BN + ReLU + MaxPool
    ↓
Conv1d(64→128, k=3) + BN + ReLU + MaxPool
    ↓
Flatten + FC → 5 classes (Normal, 4 arrhythmia types)
```

**File**: `practice/math_foundations/module07.5_cnn_applications/04_time_series.py`

---

## Running the Projects

### Setup
```bash
cd practice/math_foundations/module07.5_cnn_applications
pip install -r requirements.txt
```

### Download Data
```bash
# Project 1: UrbanSound8K (requires manual download - see script)
python data/download_urbansound.py

# Project 2: IMDB (auto-downloads via torchtext)
# No action needed

# Project 3: Chest X-ray (Kaggle dataset)
python data/download_chestxray.py

# Project 4: MIT-BIH ECG (auto-downloads via wfdb)
# No action needed
```

### Train Models
```bash
python 01_audio_classification.py
python 02_text_classification.py
python 03_medical_imaging.py
python 04_time_series.py
```

Each script includes demo mode with synthetic data if real datasets aren't downloaded.

---

## Tips for Success

1. **Start with Project 2** (Text Classification) - data auto-downloads
2. **Use GPU if available** - Training is 10-50x faster
3. **Monitor overfitting** - Use validation set, apply dropout
4. **Data augmentation helps** - Especially for small datasets
5. **Transfer learning is powerful** - Use it for image tasks

---

## Further Reading

### Architecture Papers
- [VGGNet](https://arxiv.org/abs/1409.1556) - Deep networks with small filters
- [ResNet](https://arxiv.org/abs/1512.03385) - Skip connections
- [Inception](https://arxiv.org/abs/1409.4842) - Multi-scale features

### Application Papers
- [CNN for Text Classification](https://arxiv.org/abs/1408.5882) - Kim 2014
- [CNN for Audio](https://arxiv.org/abs/1610.00087) - Environmental Sound Classification
- [CheXNet](https://arxiv.org/abs/1711.05225) - Chest X-ray diagnosis

### Tutorials
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Stanford CS231n](http://cs231n.stanford.edu/) - CNN course
