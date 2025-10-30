# Capstone Project 2: Implement Classic Papers

This guide walks you through implementing landmark deep learning papers that shaped the field. You'll build architectures from LeNet to GPT-2, understanding the evolution of deep learning.

**Time Estimate:** 50-60 hours
**Prerequisites:** Completed Modules 1-14

## Table of Contents
1. [LeNet-5 (1998)](#part-1-lenet-5)
2. [AlexNet (2012)](#part-2-alexnet)
3. [ResNet (2015)](#part-3-resnet)
4. [U-Net (2015)](#part-4-u-net)
5. [Seq2Seq (2014)](#part-5-seq2seq)
6. [Attention Mechanism (2015)](#part-6-attention)
7. [BERT (2018)](#part-7-bert)
8. [GPT-2 (2019)](#part-8-gpt-2)
9. [GAN (2014)](#part-9-gan)
10. [VAE (2013)](#part-10-vae)

---

## Part 1: LeNet-5 (1998)

### Paper
**"Gradient-Based Learning Applied to Document Recognition"** by LeCun et al.

### Overview
The first successful application of CNNs to handwritten digit recognition. Revolutionary for its time.

### Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 architecture (1998).

    Architecture:
    Input (32x32x1) →
    C1: Conv(6, 5x5) → 28x28x6 →
    S2: AvgPool(2x2) → 14x14x6 →
    C3: Conv(16, 5x5) → 10x10x16 →
    S4: AvgPool(2x2) → 5x5x16 →
    C5: Conv(120, 5x5) → 1x1x120 →
    F6: FC(84) →
    Output: FC(10)
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # C1: Conv + activation
        x = torch.tanh(self.conv1(x))
        # S2: Pooling
        x = self.pool1(x)

        # C3: Conv + activation
        x = torch.tanh(self.conv2(x))
        # S4: Pooling
        x = self.pool2(x)

        # C5: Conv + activation
        x = torch.tanh(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # F6: FC + activation
        x = torch.tanh(self.fc1(x))

        # Output
        x = self.fc2(x)

        return x

# Training script
def train_lenet():
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Data loading
    transform = transforms.Compose([
        transforms.Resize(32),  # LeNet expects 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        # Test
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total

        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
    return model

# Run training
if __name__ == '__main__':
    model = train_lenet()
```

**Expected Result:** >98% accuracy on MNIST

---

## Part 2: AlexNet (2012)

### Paper
**"ImageNet Classification with Deep Convolutional Neural Networks"** by Krizhevsky et al.

### Overview
Won ImageNet 2012 by a large margin. Popularized deep learning and GPUs for training.

### Key Innovations
1. ReLU activation (much faster than tanh)
2. Dropout for regularization
3. Data augmentation
4. GPU training with model parallelism

### Architecture (Simplified for CIFAR-10)

```python
class AlexNet(nn.Module):
    """
    AlexNet architecture (2012), adapted for CIFAR-10.

    Original was for ImageNet (224x224x3 input).
    This is a simplified version for CIFAR-10 (32x32x3).
    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Conv1: 32x32x3 → 15x15x64
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: 15x15x64 → 7x7x192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: 7x7x192 → 7x7x384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 7x7x384 → 7x7x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 7x7x256 → 3x3x256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training with data augmentation
def train_alexnet():
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision

    # Data augmentation (key innovation in AlexNet)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        scheduler.step()

        # Test every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()

            test_acc = 100. * test_correct / test_total
            print(f'Epoch {epoch+1}: Test Acc: {test_acc:.2f}%')

    return model
```

**Expected Result:** ~85-90% accuracy on CIFAR-10

---

## Part 3: ResNet (2015)

### Paper
**"Deep Residual Learning for Image Recognition"** by He et al.

### Overview
Introduced skip connections (residual connections), enabling training of very deep networks (100+ layers).

### Key Innovation: Residual Block

```python
class ResidualBlock(nn.Module):
    """
    Basic residual block: F(x) + x

    If dimensions don't match, use 1x1 conv on shortcut.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add shortcut
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet18(nn.Module):
    """
    ResNet-18 architecture.

    18 layers total: 1 conv + 8 residual blocks (16 layers) + 1 fc
    """

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Compare with/without skip connections
def compare_skip_connections():
    """
    Train ResNet with and without skip connections to see the difference.
    """
    import matplotlib.pyplot as plt

    class PlainNet(nn.Module):
        """Same as ResNet18 but without skip connections."""
        # Implementation similar to ResNet18 but no shortcuts
        pass

    # Train both models
    resnet_losses = []
    plainnet_losses = []

    # ... training code ...

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(resnet_losses, label='ResNet (with skip connections)')
    plt.plot(plainnet_losses, label='PlainNet (without skip connections)')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Effect of Skip Connections')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('skip_connections_comparison.png')
    plt.show()
```

**Expected Result:** ~93-95% accuracy on CIFAR-10 with ResNet-18

---

## Part 4: U-Net (2015)

### Paper
**"U-Net: Convolutional Networks for Biomedical Image Segmentation"** by Ronneberger et al.

### Overview
Encoder-decoder architecture for image segmentation. U-shaped with skip connections between encoder and decoder.

### Architecture

```python
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.

    Structure:
    - Encoder: Downsample with max pooling
    - Decoder: Upsample with transposed convolution
    - Skip connections: Concatenate encoder features to decoder
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 (upsampled) + 512 (skip)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        return self.out(dec1)

# Example: Train on segmentation task
def train_unet_segmentation():
    """
    Train U-Net on image segmentation task.
    Example: Oxford-IIIT Pet dataset with segmentation masks.
    """
    model = UNet(in_channels=3, out_channels=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop similar to previous examples
    # ...

    return model
```

---

## Part 5: Seq2Seq (2014)

### Paper
**"Sequence to Sequence Learning with Neural Networks"** by Sutskever et al.

### Overview
Encoder-decoder architecture for sequence-to-sequence tasks (translation, summarization, etc.).

### Architecture

```python
class Encoder(nn.Module):
    """
    Encoder: Encodes input sequence into context vector.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: (seq_len, batch_size)

        Returns:
            hidden: (n_layers, batch_size, hidden_dim)
            cell: (n_layers, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    """
    Decoder: Generates output sequence from context vector.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        Args:
            input: (batch_size,)
            hidden: (n_layers, batch_size, hidden_dim)
            cell: (n_layers, batch_size, hidden_dim)

        Returns:
            prediction: (batch_size, output_dim)
            hidden: (n_layers, batch_size, hidden_dim)
            cell: (n_layers, batch_size, hidden_dim)
        """
        input = input.unsqueeze(0)  # (1, batch_size)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """
    Complete Seq2Seq model.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (src_len, batch_size)
            trg: (trg_len, batch_size)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: (trg_len, batch_size, output_dim)
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encode
        hidden, cell = self.encoder(src)

        # First input to decoder is <sos> token
        input = trg[0, :]

        # Decode
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
```

---

## Part 6: Attention Mechanism (2015)

### Paper
**"Neural Machine Translation by Jointly Learning to Align and Translate"** by Bahdanau et al.

### Overview
Attention allows the decoder to focus on different parts of the input sequence dynamically.

### Implementation

```python
class Attention(nn.Module):
    """
    Bahdanau (additive) attention.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: (batch_size, hidden_dim) - Current decoder hidden state
            encoder_outputs: (src_len, batch_size, hidden_dim)

        Returns:
            attention_weights: (batch_size, src_len)
        """
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs: (src_len, batch, hidden) → (batch, src_len, hidden)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Compute attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hidden_dim + emb_dim), hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        Args:
            input: (batch_size,)
            hidden: (n_layers, batch_size, hidden_dim)
            encoder_outputs: (src_len, batch_size, hidden_dim)

        Returns:
            prediction: (batch_size, output_dim)
            hidden: (n_layers, batch_size, hidden_dim)
        """
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        # Compute attention weights
        a = self.attention(hidden[-1], encoder_outputs)

        # Apply attention to encoder outputs
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        # Concatenate embedded input and weighted encoder outputs
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # RNN
        output, hidden = self.rnn(rnn_input, hidden)

        # Prediction
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden
```

---

## Part 7: BERT (2018)

### Paper
**"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** by Devlin et al.

### Overview
Bidirectional transformer pre-trained on masked language modeling and next sentence prediction.

### Simplified BERT Implementation

```python
class BERTEmbedding(nn.Module):
    """
    BERT Embedding: Token + Position + Segment embeddings.
    """

    def __init__(self, vocab_size, hidden_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_emb = nn.Embedding(max_len, hidden_dim)
        self.segment_emb = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment_label):
        batch_size, seq_len = sequence.size()
        position = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand_as(sequence)

        embedding = self.token_emb(sequence) + \
                   self.position_emb(position) + \
                   self.segment_emb(segment_label)

        return self.dropout(embedding)

class BERT(nn.Module):
    """
    Simplified BERT model for pre-training.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # Embedding
        self.embedding = BERTEmbedding(vocab_size=vocab_size, hidden_dim=hidden)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=attn_heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, segment_info):
        # Embedding
        x = self.embedding(x, segment_info)

        # Transformer encoding
        for transformer in self.transformer_blocks:
            x = transformer(x)

        return x

class BERTForMaskedLM(nn.Module):
    """
    BERT for Masked Language Modeling task.
    """

    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm = nn.Linear(bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.mask_lm(x)
```

---

## Part 8: GPT-2 (2019)

### Paper
**"Language Models are Unsupervised Multitask Learners"** by Radford et al.

### Overview
Decoder-only transformer trained on next-token prediction. Scaled up to 1.5B parameters.

### GPT-2 Small (124M params) Implementation

```python
class GPT2Config:
    """Configuration for GPT-2 Small."""
    vocab_size = 50257
    n_positions = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    dropout = 0.1

class GPT2(nn.Module):
    """
    GPT-2 model (decoder-only transformer).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.n_layer)
        ])

        # Layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        # Embeddings
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, seq_len, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            # Create causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=idx.device)
            x = block(x, x, tgt_mask=causal_mask)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# Training on TinyStories dataset
def train_gpt2_small():
    """
    Train GPT-2 Small on TinyStories dataset.
    TinyStories is a small dataset suitable for training small models.
    """
    config = GPT2Config()
    model = GPT2(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Training loop...
    return model
```

---

## Part 9: GAN (2014)

### Paper
**"Generative Adversarial Networks"** by Goodfellow et al.

### Overview
Two networks compete: Generator creates fake data, Discriminator tries to distinguish real from fake.

### Implementation

```python
class Generator(nn.Module):
    """GAN Generator for MNIST."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    """GAN Discriminator for MNIST."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Training GAN
def train_gan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    # Training loop (similar to previous GAN guide)
    # ...

    return generator, discriminator
```

---

## Part 10: VAE (2013)

### Paper
**"Auto-Encoding Variational Bayes"** by Kingma & Welling

### Implementation

```python
class VAE(nn.Module):
    """Variational Autoencoder for MNIST."""

    def __init__(self, latent_dim=20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = reconstruction + KL divergence."""
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

---

## Final Comparison Blog Post

### Template for comparing all architectures

```markdown
# Deep Learning Architectures: A Comprehensive Comparison

## 1. Evolution Timeline
- LeNet-5 (1998): First successful CNN
- AlexNet (2012): Deep learning revolution
- ResNet (2015): Skip connections enable very deep networks
- U-Net (2015): Encoder-decoder for segmentation
- Seq2Seq (2014): Sequence-to-sequence learning
- Attention (2015): Dynamic focus mechanism
- BERT (2018): Bidirectional pre-training
- GPT-2 (2019): Large-scale language modeling
- GAN (2014): Generative modeling
- VAE (2013): Probabilistic generation

## 2. Performance Comparison

| Model | Task | Dataset | Accuracy | Training Time |
|-------|------|---------|----------|---------------|
| LeNet-5 | Classification | MNIST | 98.5% | 10 min |
| AlexNet | Classification | CIFAR-10 | 87.3% | 2 hours |
| ResNet-18 | Classification | CIFAR-10 | 94.2% | 4 hours |
| ... | ... | ... | ... | ... |

## 3. Key Insights
- Skip connections are crucial for training deep networks
- Attention mechanisms dramatically improve sequence tasks
- Pre-training is essential for NLP tasks
- Data augmentation matters as much as architecture

## 4. Lessons Learned
[Your personal insights from implementing each architecture]
```

---

## Success Criteria

You've successfully completed the capstone when:

1. ✅ Implemented all 10 architectures
2. ✅ Each model trains and achieves reasonable accuracy
3. ✅ Understood the key innovation in each paper
4. ✅ Written comprehensive comparison blog post
5. ✅ Code is well-documented and reproducible

## Resources

- [Papers with Code](https://paperswithcode.com/)
- [Distill.pub](https://distill.pub/) - Visual explanations
- [Yannic Kilcher's channel](https://www.youtube.com/c/YannicKilcher) - Paper explanations
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
