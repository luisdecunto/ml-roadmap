# ML Roadmap - Complete Guide Index

**Last Updated:** 2025-10-30

This document lists all available guides and guides that need to be created.

---

## ✅ Available Guides

### Setup Guides
1. **[PostgreSQL Setup](guides/postgresql_setup.html)** - Database installation and configuration (~45 min)
2. **[Docker & Docker Compose](guides/docker_setup.html)** - Container basics and orchestration (~60 min)
3. **[ML Environment Setup](guides/ml_environment_setup.html)** - Anaconda, PyTorch, VS Code (~60 min)

### SQL Practice Guides
4. **[SQL CTEs (Common Table Expressions)](guides/sql_cte_practice_guide.html)** - WITH clause, multi-step calculations, recursive queries (2-3 hrs) **NEW!**
5. **[SQL Window Functions](guides/sql_window_functions_guide.html)** - ROW_NUMBER, RANK, LAG, LEAD, moving averages (3-4 hrs) **NEW!**
6. **[SQL Query Optimization](guides/sql_optimization_guide.html)** - EXPLAIN ANALYZE, indexes, performance tuning (2-3 hrs) **NEW!**

### ML Project Guides
7. **[ETL Pipeline](guides/etl_pipeline_guide.html)** - Weather API → Pandas → PostgreSQL (4-6 hrs)

### Math Coding Guides
8. **[Linear Algebra Coding](guides/linear_algebra_coding_guide.html)** - Matrix ops, SVD, image compression (8-12 hrs)
9. **[Matrix Calculus Coding](guides/matrix_calculus_coding_guide.html)** - Gradient checking, Jacobians, QR decomposition (6-8 hrs)
10. **[Optimization Algorithms](guides/optimization_coding_guide.html)** - GD, SGD, Momentum, RMSprop, Adam (6-8 hrs)
11. **[Neural Networks from Scratch](guides/neural_networks_guide.html)** - Neurons, backprop, MNIST (12-16 hrs)
12. **[CNNs from Scratch](guides/cnn_guide.html)** - Convolution, pooling, complete CNN (15-20 hrs)
13. **[Reinforcement Learning Fundamentals](guides/reinforcement_learning_guide.html)** - Q-Learning, DQN, Policy Gradients, Actor-Critic (20-25 hrs)
14. **[Transformers from Scratch](guides/transformer_guide.html)** - Attention, multi-head, encoder/decoder (20-25 hrs)
15. **[Probability & Information Theory](guides/probability_guide.html)** - Distributions, entropy, KL divergence (6-8 hrs)

### Advanced Deep Learning Guides
16. **[LLM Mathematics](guides/llm_mathematics_guide.html)** - Micrograd, Bigram, GPT, Sampling strategies (25-30 hrs)
17. **[Optimization Deep Dive](guides/optimization_deep_dive_guide.html)** - LR schedules, warmup, loss landscapes, gradient clipping (15-20 hrs)
18. **[Regularization & Generalization](guides/regularization_guide.html)** - L1/L2/Elastic Net, data augmentation, bias-variance (15-20 hrs)
19. **[Modern Architectures](guides/modern_architectures_guide.html)** - ViT, VAE, GAN, Diffusion Models (20-25 hrs)

### Capstone Project Guides
20. **[Build Your Own Deep Learning Framework](guides/deep_learning_framework_guide.html)** - Complete autograd engine, layers, optimizers, DataLoader (40-50 hrs)
21. **[Implement Classic Papers](guides/classic_papers_guide.html)** - LeNet, AlexNet, ResNet, U-Net, Seq2Seq, BERT, GPT-2, GAN, VAE (50-60 hrs)

### Exercise Sets (with Solutions)
22. **[Linear Algebra Exercises](guides/exercises/linear_algebra_exercises.html)** - 22 exercises (3-4 hrs)
23. **[Calculus & Gradients Exercises](guides/exercises/calculus_gradients_exercises.html)** - 23 exercises (3-4 hrs)
24. **[Optimization Exercises](guides/exercises/optimization_exercises.html)** - 20 exercises (2-3 hrs)
25. **[Probability & Statistics Exercises](guides/exercises/probability_statistics_exercises.html)** - 19 exercises (3-4 hrs)
26. **[Information Theory Exercises](guides/exercises/information_theory_exercises.html)** - 18 exercises (2-3 hrs)
27. **[Neural Networks Exercises](guides/exercises/neural_networks_exercises.html)** - 22 exercises (4-5 hrs)
28. **[CNN Exercises](guides/exercises/cnn_exercises.html)** - 22 exercises (4-5 hrs)
29. **[Transformer Exercises](guides/exercises/transformer_exercises.html)** - 22 exercises (4-5 hrs)

---

## 📋 Guides TODO (Prioritized)

### High Priority - Core ML Skills

#### Advanced CNN Techniques
**Status:** Not started
**Tasks covered:**
- Code: Detailed BatchNorm implementation
- Code: Residual blocks
- Code: Dropout implementation

**Content needed:**
- In-depth BatchNorm (forward, backward, running stats)
- Skip connections and residual blocks
- Dropout regularization

**Estimated:** 4-6 hours

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

**Coding Guides:** 15 complete (Linear Algebra, Matrix Calculus, Optimization, Neural Networks, CNNs, Transformers, Probability, LLM Math, Optimization Deep Dive, Regularization, Modern Architectures)
**Capstone Guides:** 2 complete (Deep Learning Framework, Classic Papers)
**SQL Practice Guides:** 3 complete (CTEs, Window Functions, Query Optimization)
**Exercise Sets:** 8 complete (all with solutions)
**Setup Guides:** 3 complete

**Total Complete:** 31 guides
**Still TODO:** 1 guide (Reinforcement Learning Fundamentals)

**Coverage:** 🔄 31/32 guides complete - RL guide needed!

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
