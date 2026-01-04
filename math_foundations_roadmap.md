# Mathematical ML Foundations Roadmap

**Purpose:** Build deep mathematical understanding to differentiate yourself from bootcamp graduates. Work through this alongside your main ML roadmap.

**Time Commitment:** 1-2 modules per week (8-12 hours/week)

**Approach:** Theory + Implementation (mix of NumPy and PyTorch)

---

## Module 1: Linear Algebra Fundamentals

**Time:** 15-20 hours

### Tasks
- [ ] Watch 3Blue1Brown Essence of Linear Algebra (complete series)
- [ ] Read Mathematics for Machine Learning Ch 2: Linear Algebra
- [ ] Complete exercises: Vectors and vector spaces
- [ ] Complete exercises: Matrix operations
- [ ] Implement matrix multiplication in NumPy from scratch
- [ ] Study eigenvalues and eigenvectors (theory + intuition)
- [ ] Implement eigenvalue decomposition
- [ ] Learn Singular Value Decomposition (SVD)
- [ ] Implement SVD from scratch in NumPy
- [ ] Apply SVD to image compression project

**Resources:**
- [3Blue1Brown Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Mathematics for Machine Learning (Free PDF)](https://mml-book.com/)
- [Fast.ai Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra)

---

## Module 2: Matrix Calculus for ML

**Time:** 15-20 hours

### Tasks
- [ ] Read Mathematics for Machine Learning Ch 5: Vector Calculus
- [ ] Study gradients and Jacobians
- [ ] Learn chain rule for backpropagation
- [ ] Practice computing gradients manually (on paper)
- [ ] Implement gradient computation in NumPy
- [ ] Study Hessian matrices and their applications
- [ ] Watch 3Blue1Brown calculus series (optional review)
- [ ] Complete Fast.ai Computational Linear Algebra videos
- [ ] Implement QR decomposition
- [ ] Study numerical stability in matrix operations

**Resources:**
- [Mathematics for Machine Learning Ch 5](https://mml-book.com/)
- [Fast.ai Computational Linear Algebra Course](https://github.com/fastai/numerical-linear-algebra)

---

## Module 3: Optimization Foundations

**Time:** 12-15 hours

### Tasks
- [ ] Study gradient descent algorithm (theory + derivation)
- [ ] Implement gradient descent from scratch
- [ ] Learn stochastic gradient descent (SGD)
- [ ] Implement SGD with mini-batches
- [ ] Study momentum and its intuition
- [ ] Implement SGD with momentum
- [ ] Learn RMSprop algorithm
- [ ] Learn Adam optimizer mathematics
- [ ] Implement Adam from scratch (follow Aman Arora tutorial)
- [ ] Compare all optimizers on same task and visualize

**Resources:**
- [Adam and Friends Tutorial](https://amaarora.github.io/posts/2021-03-13-optimizers.html)
- Understanding Deep Learning Ch 7 (Gradients)

---

## Module 4: Probability & Statistics

**Time:** 15-20 hours

### Tasks
- [ ] Read Mathematics for Machine Learning Ch 6: Probability
- [ ] Study probability distributions (Gaussian, Bernoulli, etc.)
- [ ] Learn maximum likelihood estimation (MLE)
- [ ] Study Bayes' theorem and applications
- [ ] Read Murphy's Probabilistic ML Ch 1-2
- [ ] Learn expectation and variance
- [ ] Study covariance and correlation
- [ ] Practice probability problem sets
- [ ] Learn Central Limit Theorem
- [ ] Apply probability concepts to a simple ML problem

**Resources:**
- [Mathematics for Machine Learning Ch 6](https://mml-book.com/)
- [Probabilistic ML (Murphy) - Free Draft](https://probml.github.io/pml-book/book1.html)

---

## Module 5: Information Theory

**Time:** 10-15 hours

### Tasks
- [ ] Read MacKay's Information Theory Ch 1-2
- [ ] Watch: [3Blue1Brown - Solving Wordle using Information Theory](https://www.3blue1brown.com/lessons/wordle) (entropy explanation)
- [ ] Watch: [StatQuest - Entropy, Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw)
- [ ] Learn entropy and its intuition
- [ ] Study cross-entropy loss derivation
- [ ] Read: [Cross-Entropy Loss Derivation Tutorial](https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning)
- [ ] Understand KL divergence
- [ ] Read: [KL Divergence Explained - DataCamp](https://www.datacamp.com/tutorial/kl-divergence)
- [ ] Watch: [StatQuest - Mutual Information, Clearly Explained](https://www.youtube.com/watch?v=U9p5WNuwTQw)
- [ ] Learn mutual information
- [ ] Study Jensen-Shannon divergence
- [ ] Read: [Jensen-Shannon Divergence Tutorial](https://medium.com/data-science/how-to-understand-and-use-jensen-shannon-divergence-b10e11b03fd6)
- [ ] Implement entropy calculations
- [ ] Connect information theory to ML loss functions
- [ ] Read: [Information Bottleneck Theory (Tishby Paper)](https://arxiv.org/pdf/physics/0004057)
- [ ] Read about information bottleneck theory
- [ ] Study VAE from information theory perspective
- [ ] Read: [What is a Variational Autoencoder? Tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)

**Resources:**
- [Information Theory, Inference, and Learning Algorithms (MacKay) - Free PDF](http://www.inference.org.uk/mackay/itila/)
- [Dive into Deep Learning - Information Theory Chapter](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)

---

## Module 6: Feedforward Networks from Scratch

**Time:** 20-25 hours

### Tasks
- [ ] Watch 3Blue1Brown Neural Networks series (all 4 videos)
- [ ] Read Understanding Deep Learning Ch 1-3 (Prince)
- [ ] Implement neuron forward pass in NumPy
- [ ] Implement activation functions (ReLU, Sigmoid, Tanh)
- [ ] Build complete forward propagation
- [ ] Derive backpropagation equations on paper
- [ ] Implement backpropagation from scratch (follow tutorials)
- [ ] Test on XOR problem
- [ ] Train on MNIST with your implementation
- [ ] Compare results with PyTorch implementation

**Resources:**
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Understanding Deep Learning (Prince) - Free PDF](https://udlbook.github.io/udlbook/)
- [Coding Neural Network from Scratch in NumPy](https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605/)

---

## Module 7: CNNs from Scratch

**Time:** 20-25 hours

### Tasks
- [ ] Read Understanding Deep Learning Ch 10 (CNNs)
- [ ] Watch CS231n CNN lectures (Stanford)
- [ ] Understand convolution operation mathematically
- [ ] Implement 2D convolution in NumPy
- [ ] Implement max pooling
- [ ] Derive CNN backpropagation
- [ ] Follow Victor Zhou CNN tutorial completely
- [ ] Implement batch normalization
- [ ] Build simple CNN for MNIST
- [ ] Analyze and visualize learned filters

**Resources:**
- [CNNs from Scratch (Victor Zhou)](https://victorzhou.com/blog/intro-to-cnns-part-1/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Understanding Deep Learning Ch 10](https://udlbook.github.io/udlbook/)

---

## Module 8: Advanced Architectures

**Time:** 15-20 hours

### Tasks
- [ ] Study ResNet architecture and skip connections
- [ ] Understand batch normalization mathematics
- [ ] Learn layer normalization
- [ ] Study dropout from probabilistic perspective
- [ ] Implement dropout from scratch
- [ ] Learn weight initialization (Xavier, He)
- [ ] Study vanishing/exploding gradients problem
- [ ] Read about residual learning theory
- [ ] Implement simple ResNet block
- [ ] Compare different normalization techniques

**Resources:**
- [Deep Residual Learning Paper](https://arxiv.org/abs/1512.03385)
- Understanding Deep Learning (relevant chapters)

---

## Module 9: Attention Mechanisms

**Time:** 20-25 hours

### Tasks
- [ ] Read 'Attention Is All You Need' paper carefully
- [ ] Watch 3Blue1Brown attention visualization
- [ ] Follow Sebastian Raschka self-attention tutorial
- [ ] Implement self-attention from scratch
- [ ] Derive attention gradient computations
- [ ] Implement scaled dot-product attention
- [ ] Understand query, key, value intuition
- [ ] Implement multi-head attention
- [ ] Implement cross-attention
- [ ] Test attention on simple sequence task

**Resources:**
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [Self-Attention from Scratch (Sebastian Raschka)](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
- [3Blue1Brown on Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc)

---

## Module 10: Transformer Architecture

**Time:** 20-25 hours

### Tasks
- [ ] Read Understanding Deep Learning Ch 12 (Transformers)
- [ ] Study positional encoding mathematics
- [ ] Implement positional encoding
- [ ] Build transformer encoder from scratch
- [ ] Build transformer decoder from scratch
- [ ] Understand masked attention
- [ ] Implement complete transformer
- [ ] Study layer norm in transformers
- [ ] Read about transformer variants (BERT, GPT architecture)
- [ ] Watch Karpathy's GPT video

**Resources:**
- [Understanding Deep Learning Ch 12](https://udlbook.github.io/udlbook/)
- [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

## Module 11: LLM Mathematics

**Time:** 25-30 hours

### Tasks
- [ ] Complete Karpathy's Neural Networks: Zero to Hero series
- [ ] Implement bigram language model
- [ ] Build GPT from scratch (follow Karpathy)
- [ ] Study tokenization (BPE) mathematics
- [ ] Understand temperature sampling
- [ ] Learn nucleus sampling (top-p)
- [ ] Study perplexity as evaluation metric
- [ ] Understand RLHF at high level
- [ ] Read about LoRA mathematics
- [ ] Study quantization techniques (4-bit, 8-bit)

**Resources:**
- [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Understanding and Coding Self-Attention](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)

---

## Module 12: Optimization Deep Dive

**Time:** 15-20 hours

### Tasks
- [ ] Read Understanding Deep Learning Ch 7 (Gradients)
- [ ] Study learning rate schedules
- [ ] Learn cosine annealing
- [ ] Understand warmup strategies
- [ ] Study second-order methods (Newton, L-BFGS)
- [ ] Learn about natural gradients
- [ ] Study adaptive learning rates
- [ ] Implement learning rate finder
- [ ] Read about loss landscape visualization
- [ ] Study gradient clipping mathematics

**Resources:**
- Understanding Deep Learning Ch 7
- Papers on optimization techniques

---

## Module 13: Regularization & Generalization

**Time:** 15-20 hours

### Tasks
- [ ] Study L1 and L2 regularization derivations
- [ ] Understand dropout as ensemble method
- [ ] Learn about early stopping theory
- [ ] Study data augmentation mathematics
- [ ] Read about double descent phenomenon
- [ ] Learn PAC learning basics
- [ ] Study VC dimension (optional, theoretical)
- [ ] Understand bias-variance tradeoff deeply
- [ ] Read about lottery ticket hypothesis
- [ ] Study neural tangent kernels (advanced, optional)

**Resources:**
- Understanding Deep Learning (relevant chapters)
- Papers on generalization in deep learning

---

## Module 14: Modern Architectures

**Time:** 20-25 hours

### Tasks
- [ ] Study Vision Transformers (ViT)
- [ ] Learn about diffusion models mathematics
- [ ] Read about score-based models
- [ ] Study GANs from game theory perspective
- [ ] Learn VAE mathematics (ELBO derivation)
- [ ] Understand normalizing flows
- [ ] Study graph neural networks basics
- [ ] Learn about meta-learning
- [ ] Read about neural ODEs
- [ ] Study latest architecture papers (2023-2024)

**Resources:**
- Recent papers on arXiv
- Understanding Deep Learning (advanced chapters)

---

## Project 1: Build Your Own Deep Learning Framework

**Time:** 40-50 hours

### Tasks
- [ ] Design autograd system like PyTorch
- [ ] Implement Tensor class with gradient tracking
- [ ] Build automatic differentiation
- [ ] Implement common layers (Linear, Conv2d, ReLU, etc.)
- [ ] Create optimizer base class
- [ ] Implement SGD and Adam optimizers
- [ ] Build loss functions (MSE, CrossEntropy)
- [ ] Create data loading utilities
- [ ] Add training loop utilities
- [ ] Document your framework with examples and tutorials

**Goal:** Understand how PyTorch works under the hood

---

## Project 2: Implement Classic Papers

**Time:** 50-60 hours

### Tasks
- [ ] Implement LeNet from scratch (1998)
- [ ] Implement AlexNet architecture (2012)
- [ ] Build ResNet with skip connections (2015)
- [ ] Implement U-Net for segmentation (2015)
- [ ] Build seq2seq with attention (2014-2015)
- [ ] Implement BERT pretraining (2018)
- [ ] Build GPT-2 from scratch (2019)
- [ ] Implement simple GAN (2014)
- [ ] Build VAE with good visualizations (2013)
- [ ] Create comparison report of all implementations

**Goal:** Deep understanding of ML history and architecture evolution

---

## Recommended Learning Path

### Parallel Track with Main Roadmap:

**Weeks 1-4 (SQL Focus):**
- Work on Modules 1-2 (Linear Algebra, Matrix Calculus)
- 2-3 hours per day on math

**Weeks 5-8 (PyTorch + MLOps):**
- Work on Module 3 (Optimization) - very relevant!
- Work on Module 6 (Feedforward Networks from Scratch)

**Weeks 9-12 (Transformers + LLMs):**
- Work on Modules 9-11 (Attention, Transformers, LLMs)
- This directly complements your main roadmap work

**Weeks 13-16 (K8s + Jobs):**
- Work on Modules 4-5 (Probability, Info Theory)
- Work on Modules 12-13 (Optimization, Regularization)

**After Week 16:**
- Complete Module 14 (Modern Architectures)
- Work on Projects 1-2 as time allows

---

## Resources Master List

### Core Textbooks (All Free)
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/)
- [Understanding Deep Learning (Prince)](https://udlbook.github.io/udlbook/)
- [Mathematics for Machine Learning (Deisenroth)](https://mml-book.com/)
- [Probabilistic Machine Learning (Murphy)](https://probml.github.io/pml-book/)

### Video Courses
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Fast.ai Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra)
- [MIT 6.S191 Deep Learning](https://introtodeeplearning.com/)

### Implementation Tutorials
- [Self-Attention from Scratch (Raschka)](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
- [CNNs from Scratch (Victor Zhou)](https://victorzhou.com/blog/intro-to-cnns-part-1/)
- [Optimizers from Scratch (Aman Arora)](https://amaarora.github.io/posts/2021-03-13-optimizers.html)
- [Neural Network from Scratch in NumPy](https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605/)

### GitHub Repositories
- [ML-From-Scratch (24k+ stars)](https://github.com/eriklindernoren/ML-From-Scratch)
- [numpy-ml (15k+ stars)](https://github.com/ddbourgin/numpy-ml)
- [PyTorch Deep Learning Course](https://github.com/mrdbourke/pytorch-deep-learning)
- [PyTorch Tutorial (30k+ stars)](https://github.com/yunjey/pytorch-tutorial)

---

## Progress Tracking

Use checkboxes above to track your progress. Aim for:
- **1-2 modules per week** alongside main roadmap
- **~10 hours per week** on math foundations
- **Focus on understanding over speed**

---

## Why This Matters

When you interview for ML Engineer positions, you'll be able to:

✅ "I implemented backpropagation from first principles in NumPy"  
✅ "I built a transformer architecture from scratch and understand the math"  
✅ "I can derive the gradient equations for Adam optimizer"  
✅ "I understand attention mechanisms mathematically, not just `nn.MultiheadAttention`"  
✅ "I implemented CNNs in pure NumPy to understand convolution backprop"

**This is what separates senior ML Engineers from bootcamp graduates.** 🚀

---

*Work through this at your own pace. The goal is deep understanding, not rushing through checkboxes.*