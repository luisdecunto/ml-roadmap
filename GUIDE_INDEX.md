# ML Roadmap - Complete Guide Index

**Last Updated:** 2025-10-28

This document lists all available guides and guides that need to be created.

---

## ✅ Available Guides

### Setup Guides
1. **[PostgreSQL Setup](guides/postgresql_setup.html)** - Database installation and configuration (~45 min)
2. **[Docker & Docker Compose](guides/docker_setup.html)** - Container basics and orchestration (~60 min)
3. **[ML Environment Setup](guides/ml_environment_setup.html)** - Anaconda, PyTorch, VS Code (~60 min)

### ML Project Guides
4. **[ETL Pipeline](guides/etl_pipeline_guide.html)** - Weather API → Pandas → PostgreSQL (4-6 hrs)

### Math Coding Guides
5. **[Linear Algebra Coding](guides/linear_algebra_coding_guide.html)** - Matrix ops, SVD, image compression (8-12 hrs)
6. **[Optimization Algorithms](guides/optimization_coding_guide.html)** - GD, SGD, Momentum, RMSprop, Adam (6-8 hrs)

### Exercise Sets (with Solutions)
7. **[Linear Algebra Exercises](guides/exercises/linear_algebra_exercises.html)** - 40+ problems (3-4 hrs)
8. **[Calculus & Gradients Exercises](guides/exercises/calculus_gradients_exercises.html)** - ML-focused calculus (3-4 hrs)

---

## 📋 Guides TODO (Prioritized)

### High Priority - Core ML Skills

#### Neural Networks from Scratch
**Status:** Not started
**Tasks covered:**
- Code: Implement single neuron with sigmoid
- Code: Implement ReLU, Tanh, Softmax activations
- Code: Build 2-layer network forward pass
- Code: Implement full backprop
- Project: Train on MNIST (>95% accuracy)

**Content needed:**
- Single neuron implementation
- Activation functions (ReLU, Sigmoid, Tanh, Softmax) with derivatives
- Forward propagation (matrix operations)
- Backpropagation (chain rule implementation)
- Gradient checking (numerical vs analytical)
- MNIST training loop
- Comparison with PyTorch

**Estimated:** 8-10 hours

#### CNN from Scratch
**Status:** Not started
**Tasks covered:**
- Code: Implement 2D convolution
- Code: Implement max/average pooling
- Code: Build CNN (Conv→ReLU→MaxPool→FC)
- Code: Implement batch normalization
- Code: Implement dropout
- Project: MNIST CNN (>98% accuracy)

**Content needed:**
- 2D convolution (im2col method)
- Pooling layers (forward + backward)
- Complete CNN architecture
- Batch normalization (forward + backward + running stats)
- Dropout implementation
- Training on MNIST/CIFAR-10

**Estimated:** 10-12 hours

#### Transformer from Scratch
**Status:** Not started
**Tasks covered:**
- Code: Implement basic attention
- Code: Scaled dot-product attention
- Code: Positional encoding
- Code: Transformer encoder/decoder blocks
- Project: Train on addition task

**Content needed:**
- Attention mechanism (score, softmax, weighted sum)
- Scaled dot-product attention
- Multi-head attention
- Positional encoding (sinusoidal)
- Layer normalization
- Encoder block (self-attention + FFN)
- Decoder block (masked attention + cross-attention)
- Complete transformer
- Training example (addition or translation)

**Estimated:** 12-15 hours

### Medium Priority - Theory Implementation

#### Probability & Information Theory
**Status:** Not started
**Tasks covered:**
- Code: Implement Gaussian PDF
- Code: Implement Naive Bayes classifier
- Code: Implement entropy H(X)
- Code: Calculate mutual information I(X;Y)
- Code: Implement cross-entropy and KL divergence
- Project: Spam classifier (MLE vs MAP)

**Content needed:**
- Gaussian distribution implementation
- Naive Bayes (multinomial, Gaussian)
- MLE vs MAP estimation
- Entropy calculations
- Mutual information
- Cross-entropy loss
- KL divergence
- Applications to ML

**Estimated:** 6-8 hours

#### Advanced Training Techniques
**Status:** Not started
**Tasks covered:**
- Code: Learning rate schedules
- Code: Warmup implementation
- Code: Gradient clipping
- Code: Loss landscape visualization
- Code: L1/L2/Elastic net regularization

**Content needed:**
- LR schedules (step decay, exponential, cosine annealing)
- Warmup strategies
- Gradient clipping (by norm, by value)
- 1D/2D loss landscape plots
- Regularization implementations

**Estimated:** 4-6 hours

### Low Priority - Advanced Topics

#### Attention Mechanisms (Standalone)
**Status:** Not started
**Tasks covered:**
- Code: Basic attention
- Code: Scaled dot-product
- Code: Cross-attention

**Note:** This overlaps with Transformer guide. Could be merged.

#### Autograd Framework (Build Your Own PyTorch)
**Status:** Not started
**Tasks covered:**
- All Module 14 tasks (12 coding tasks)

**Content needed:**
- Tensor class with gradient tracking
- Computational graph
- Backward pass
- Basic operations (+, -, *, matmul) with autograd
- Layer implementations
- Optimizer base class
- Loss functions
- DataLoader

**Estimated:** 15-20 hours (very comprehensive)

#### Paper Implementations (Capstone 1)
**Status:** Not started
**Tasks covered:**
- LeNet-5, AlexNet, ResNet-18
- U-Net, Seq2Seq, BERT, GPT-2
- GAN, VAE

**Note:** These could be individual mini-guides or a single comprehensive guide

**Estimated:** 20-30 hours total

---

## 📝 Quick Reference: Guide Creation Checklist

When creating a new guide:

- [ ] Create `.md` file with complete content
- [ ] Create `.html` wrapper that fetches the `.md`
- [ ] Add to this index
- [ ] Link from roadmap tasks in `index.html`
- [ ] Test locally (open `.html` in browser)
- [ ] Commit both files to GitHub
- [ ] Update STATUS.md

---

## 🎯 Recommended Creation Order

**Session 1:** (Highest impact)
1. Neural Networks from Scratch
2. CNN from Scratch

**Session 2:** (Advanced architectures)
3. Transformer from Scratch
4. Probability & Information Theory

**Session 3:** (Training techniques)
5. Advanced Training Techniques
6. Attention Mechanisms (if not covered in Transformer)

**Session 4:** (Framework understanding)
7. Autograd Framework

**Session 5:** (Classic papers)
8. Paper Implementations (split into multiple guides)

---

## 📊 Progress Tracking

**Total Guides Needed:** ~15
**Completed:** 6
**In Progress:** 0
**TODO:** 9 (high/medium priority)

**Completion:** 40%

---

## 🔗 External Resources

For each guide, reference these materials:

**Neural Networks:**
- 3Blue1Brown Neural Networks series
- Understanding Deep Learning Ch 3-4
- CS231n Lecture 4

**CNNs:**
- CS231n Lecture 5
- Understanding Deep Learning Ch 10
- Victor Zhou's CNN tutorial

**Transformers:**
- Attention Is All You Need paper
- The Illustrated Transformer
- Karpathy's minGPT

**Probability:**
- Murphy's PML Ch 2-4
- MML Ch 6

**Autograd:**
- Karpathy's micrograd
- PyTorch autograd documentation

---

**Note:** This index will be updated as new guides are created. Check STATUS.md for overall project status.
