# ML Development Environment Setup Guide

**Time:** 45-60 minutes
**Difficulty:** Beginner
**Disk Space:** ~7-8GB

## What You'll Install
- Anaconda (Python distribution + package manager)
- VS Code (code editor)
- PyTorch (deep learning framework)
- Essential ML libraries

---

## Step 1: Install Anaconda

### Why Anaconda?
- Manages Python versions and packages easily
- Prevents dependency conflicts
- Includes many data science packages pre-installed
- Create isolated environments for different projects

### Download & Install

**All Platforms:**
1. Visit [anaconda.com/download](https://www.anaconda.com/download)
2. Download for your OS (Windows/macOS/Linux)
3. **Note:** ~600MB download, ~5GB installed

**Windows:**
- Run installer
- **Important:** Check "Add Anaconda to PATH" (installer discourages this, but it makes life easier)
- Install for: "Just Me"
- Installation location: Default (C:\Users\YourName\anaconda3)
- Complete installation (takes 5-10 minutes)

**macOS:**
```bash
# Download installer, then:
bash ~/Downloads/Anaconda3-2024.xx-MacOSX-x86_64.sh
# Follow prompts, accept license, confirm location
# When asked "initialize Anaconda3?": yes
```

**Linux:**
```bash
# Download installer, then:
bash ~/Downloads/Anaconda3-2024.xx-Linux-x86_64.sh
# Follow prompts, accept license
# When asked "initialize Anaconda3?": yes
```

### Verify Installation

```bash
# Open new terminal/command prompt (important: NEW terminal)
conda --version
# Should output: conda 23.x.x or similar

python --version
# Should output: Python 3.11.x

# Update conda
conda update conda
```

---

## Step 2: Install VS Code

### Download & Install

1. Visit [code.visualstudio.com](https://code.visualstudio.com/)
2. Download for your OS
3. Install with default settings

### Install Essential Extensions

Open VS Code, then:

1. Click Extensions icon (left sidebar) or `Ctrl+Shift+X`
2. Install these extensions:
   - **Python** (by Microsoft) - Must have!
   - **Jupyter** (by Microsoft) - For notebooks
   - **Pylance** (usually comes with Python extension)
   - **GitLens** (optional, great for Git)
   - **Docker** (optional, if using containers)

### Configure Python in VS Code

1. Open Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose your Anaconda Python (should show path like `.../anaconda3/bin/python`)

---

## Step 3: Create Your First Conda Environment

### Why Environments?
- Keep project dependencies isolated
- Avoid version conflicts
- Easy to reproduce setup

### Create ML Environment

```bash
# Create environment named 'ml' with Python 3.10
conda create -n ml python=3.10

# Activate it
conda activate ml

# Your prompt should change to show (ml)
```

### Make it permanent (optional)
```bash
# To auto-activate ml environment in new terminals:
# Add to ~/.bashrc (Linux/Mac) or run each time (Windows):
conda config --set auto_activate_base false
```

---

## Step 4: Install PyTorch

### Check if you have NVIDIA GPU

**Windows:**
```cmd
nvidia-smi
# If this works and shows GPU info, you have CUDA-capable GPU
# If error, you have CPU only
```

**Linux:**
```bash
lspci | grep -i nvidia
```

**macOS:**
```bash
# M1/M2 Macs: Use MPS (Metal Performance Shaders)
# Intel Macs: CPU only
```

### Install PyTorch

1. Visit [pytorch.org](https://pytorch.org/get-started/locally/)
2. Select your configuration:
   - **PyTorch Build:** Stable
   - **Your OS:** Windows/Mac/Linux
   - **Package:** Conda
   - **Language:** Python
   - **Compute Platform:**
     - CUDA 11.8 or 12.1 (if you have NVIDIA GPU)
     - CPU (if no GPU or macOS Intel)
     - Default (for M1/M2 Mac)

3. Copy the generated command, for example:

**NVIDIA GPU (Windows/Linux):**
```bash
conda activate ml
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**CPU Only (any OS):**
```bash
conda activate ml
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**M1/M2 Mac:**
```bash
conda activate ml
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

**Installation takes 5-10 minutes**

### Verify PyTorch Installation

```python
python
>>> import torch
>>> print(torch.__version__)
2.1.0  # or similar

# Check CUDA availability (if you have NVIDIA GPU)
>>> print(torch.cuda.is_available())
True  # or False if CPU only

# Check MPS availability (M1/M2 Mac)
>>> print(torch.backends.mps.is_available())
True  # or False

>>> exit()
```

---

## Step 5: Install Essential ML Packages

```bash
# Make sure ml environment is activated
conda activate ml

# Install with conda (faster, handles dependencies better)
conda install numpy pandas matplotlib seaborn scikit-learn jupyter notebook

# Install with pip (for packages not in conda)
pip install tensorboard fastapi uvicorn python-multipart pillow requests

# Install additional tools
pip install ipywidgets tqdm

# Verify installations
python -c "import numpy, pandas, matplotlib, sklearn; print('All imports successful!')"
```

---

## Step 6: Setup Jupyter Notebook

```bash
# Activate environment
conda activate ml

# Start Jupyter
jupyter notebook

# Browser should open automatically
# If not, copy the URL from terminal (looks like http://localhost:8888/?token=...)
```

### Test Jupyter:
1. Click "New" → "Python 3 (ipykernel)"
2. In first cell, type:
   ```python
   import torch
   import numpy as np
   import pandas as pd
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```
3. Press `Shift+Enter` to run
4. Should see output without errors

---

## Step 7: Configure Git for ML Projects

### Create .gitignore Template

Create file `.gitignore` in your project folder:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb_checkpoints

# ML/Data Science
data/
datasets/
*.csv
*.h5
*.pkl
*.pickle
models/
*.pth
*.pt
*.ckpt
*.pb
*.onnx
wandb/
mlruns/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.venv
conda-meta/
```

---

## Step 8: Quick Test - Train Your First Model

Create `test_pytorch.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load MNIST
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Train for 1 epoch
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training...")
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx >= 10:  # Just 10 batches for quick test
        break

    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

print("Test successful!")
```

Run it:
```bash
python test_pytorch.py
```

---

## Verification Checklist

- [ ] Anaconda installed, `conda --version` works
- [ ] VS Code installed with Python extension
- [ ] Created `ml` conda environment
- [ ] PyTorch installed and imports successfully
- [ ] Can check GPU availability (cuda or mps)
- [ ] Jupyter Notebook works
- [ ] Essential packages installed (numpy, pandas, etc.)
- [ ] Successfully ran test PyTorch script
- [ ] .gitignore file created

---

## Common Issues

### Issue: "conda: command not found"
- **Solution:**
  - Close and reopen terminal (conda added to PATH during install)
  - Or manually add to PATH
  - Windows: `C:\Users\YourName\anaconda3\Scripts`
  - Mac/Linux: `/Users/YourName/anaconda3/bin` or `/home/YourName/anaconda3/bin`

### Issue: PyTorch CUDA not available despite having NVIDIA GPU
- **Solution:**
  1. Update NVIDIA drivers
  2. Reinstall PyTorch with correct CUDA version
  3. Check: `nvidia-smi` to see CUDA version
  4. Install matching PyTorch CUDA version

### Issue: "Kernel dead" in Jupyter
- **Solution:**
  - Memory issue - restart kernel
  - Or: `conda install notebook --force-reinstall`

### Issue: Import errors
- **Solution:**
  - Make sure environment is activated: `conda activate ml`
  - Reinstall package: `pip install --force-reinstall package_name`

### Issue: VS Code doesn't find Python interpreter
- **Solution:**
  - `Ctrl+Shift+P` → "Python: Select Interpreter"
  - If not listed, manually enter path: `.../anaconda3/envs/ml/bin/python`

---

## Conda Environment Cheat Sheet

```bash
# Create environment
conda create -n myenv python=3.10

# Activate environment
conda activate myenv

# Deactivate
conda deactivate

# List all environments
conda env list

# Install package in current environment
conda install package_name
pip install package_name

# Remove environment
conda env remove -n myenv

# Export environment (for sharing)
conda env export > environment.yml

# Create from environment file
conda env create -f environment.yml

# Update all packages
conda update --all
```

---

## Next Steps

1. Complete Fast.ai Lesson 1
2. Work through Learn PyTorch tutorials
3. Build your first image classifier
4. Start ML portfolio project

---

## Resources

- [Anaconda Docs](https://docs.anaconda.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
