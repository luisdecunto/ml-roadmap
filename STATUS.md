# ML Roadmap Project - Status Document

**Last Updated:** 2025-10-28
**Session:** Context continuation document

---

## 🎯 Project Overview

Building a comprehensive ML Engineer career roadmap with:
- Interactive progress tracking (checkboxes, progress bars)
- Dual roadmaps: ML Engineer + Math Foundations
- Viewer/Edit modes (GitHub sync for progress)
- Comprehensive guides for all setup, coding, and project tasks
- Exercise system with solutions

**GitHub Repository:** https://github.com/luisdecunto/ml-roadmap
**Live Site:** https://luisdecunto.github.io/ml-roadmap/ (if GitHub Pages enabled)

---

## ✅ Completed Features

### Core Functionality
- [x] Tabbed UI (ML Engineer tab + Math Foundations tab)
- [x] Progress bars (one for each roadmap, roadmap-specific calculation)
- [x] Checkbox system with localStorage persistence
- [x] Viewer mode (default, loads progress without auth)
- [x] Edit mode (requires GitHub token for syncing)
- [x] GitHub progress sync (save/load to progress.json)

### Content
- [x] ML Engineer roadmap (14 weeks, fully detailed)
- [x] Math Foundations roadmap (14 modules + 2 capstones, 167 tasks)
- [x] Specific references added to all tasks (books, videos, success criteria)

### Guides System

#### ✅ Setup Guides (3/3 complete)
1. **PostgreSQL Setup** (`guides/postgresql_setup.html` + `.md`)
   - 8 steps with troubleshooting
   - Platform-specific instructions (Windows/Mac/Linux)
   - Verification checklist
   - ~45 minutes

2. **Docker & Docker Compose** (`guides/docker_setup.html` + `.md`)
   - 8 steps including WSL2 setup
   - Dockerfile and docker-compose examples
   - Best practices
   - ~60 minutes

3. **ML Environment Setup** (`guides/ml_environment_setup.html` + `.md`)
   - Anaconda installation
   - PyTorch setup (CPU/CUDA/M1)
   - VS Code configuration
   - ~60 minutes

#### ✅ Project Guides (2/3 complete)
1. **ETL Pipeline** (`guides/etl_pipeline_guide.html` + `.md`)
   - Extract from OpenWeather API
   - Transform with Pandas
   - Load to PostgreSQL
   - Scheduling automation
   - 4-6 hours

2. **Linear Algebra Coding** (`guides/linear_algebra_coding_guide.html` + `.md`)
   - Matrix multiplication (3 implementations)
   - Power iteration for eigenvalues
   - SVD from scratch
   - Image compression project
   - 8-12 hours

3. **Optimization Algorithms** (`guides/optimization_coding_guide.md` - JUST CREATED)
   - Gradient Descent, SGD, Mini-batch
   - Momentum, RMSprop, Adam
   - MNIST comparison project
   - 6-8 hours
   - ⚠️ **TODO:** Create HTML version

#### ✅ Exercise System (2/2 complete)
1. **Linear Algebra Exercises** (`guides/exercises/linear_algebra_exercises.html` + `.md`)
   - 5 parts: vectors, matrices, eigenvalues, SVD, applications
   - 40+ problems
   - NumPy verification code
   - 3-4 hours

2. **Calculus & Gradients Exercises** (`guides/exercises/calculus_gradients_exercises.html` + `.md`)
   - 7 parts: derivatives, partials, gradients, chain rule, Jacobians, Hessians, ML applications
   - 20 problems + 3 challenges
   - ML-specific (backprop, gradient checking)
   - 3-4 hours

#### ✅ Solutions (2/2 complete)
1. **Linear Algebra Solutions** (`guides/solutions/linear_algebra_solutions.html` + `.md`)
   - Complete step-by-step solutions
   - All work shown
   - 18,913 characters

2. **Calculus & Gradients Solutions** (`guides/solutions/calculus_gradients_solutions.html` + `.md`)
   - Complete solutions with explanations
   - Challenge problems included
   - 17,266 characters

#### ✅ Interactive Tools
- **Exercise Tracker** (`exercises.html`)
  - Progress tracking per exercise set
  - Toggle between exercises and solutions
  - localStorage persistence
  - Uses marked.js for markdown rendering

### UX Improvements
- [x] Clickable guide links in tasks (using `dangerouslySetInnerHTML`)
- [x] Color-coded link badges:
  - 📖 **Setup Guide** (blue #2563eb)
  - 📖 **Project Guide** (green #10b981)
  - 💻 **Coding Guide** (green #10b981)
- [x] Cleaner task text (removed verbose "follow guides/...")
- [x] Direct links to exercises and solutions in resources section
- [x] Tab-specific resources (only show relevant resources per tab)

---

## 📋 TODO: Guides Needed

### Math Roadmap - Coding Guides Still Needed

#### Module 2: Matrix Calculus (3 coding tasks)
- [ ] **Numerical Gradient Checking Guide**
  - Implement finite difference approximation
  - Compare analytical vs numerical gradients
  - Relative error calculation
  - Testing framework

- [ ] **QR Decomposition Guide**
  - Gram-Schmidt process
  - QR factorization
  - Applications to least squares

#### Module 3: Optimization (PARTIALLY DONE)
- [x] Optimization guide created (`.md` only)
- [ ] **TODO:** Create `optimization_coding_guide.html`
- [ ] Link to tasks in index.html

#### Module 4: Probability & Statistics (3 coding tasks + 1 project)
- [ ] **Probability Distributions Guide**
  - Implement Gaussian PDF from scratch
  - Plot for different parameters
  - Sampling methods

- [ ] **Naive Bayes Classifier Guide**
  - Implement on iris dataset
  - MLE vs MAP estimation
  - Spam classifier project (compare MLE/MAP)

#### Module 5: Information Theory (3 coding tasks)
- [ ] **Information Theory Guide**
  - Implement entropy H(X)
  - Joint entropy, conditional entropy
  - Mutual information I(X;Y)
  - Cross-entropy and KL divergence
  - Applications to ML loss functions

#### Module 6: Feedforward Networks (5 coding tasks + 1 project)
- [ ] **Neural Networks from Scratch Guide**
  - Single neuron implementation
  - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
  - 2-layer network (forward + backward)
  - Numerical gradient testing
  - MNIST training project (95%+ accuracy)

#### Module 7: CNNs from Scratch (4 coding tasks + 1 project)
- [ ] **CNN Implementation Guide**
  - 2D convolution in NumPy
  - Max pooling, average pooling
  - Complete CNN: Conv→ReLU→MaxPool→FC
  - MNIST CNN project (98%+ accuracy)

#### Module 8: Advanced Techniques (4 coding tasks)
- [ ] **Advanced CNN Techniques Guide**
  - Batch normalization (forward + backward)
  - Dropout implementation
  - Residual blocks

#### Module 9: Attention Mechanisms (3 coding tasks)
- [ ] **Attention Mechanisms Guide**
  - Basic attention (score, softmax, weighted sum)
  - Scaled dot-product attention
  - Cross-attention

#### Module 10: Transformers (7 coding tasks + 1 project)
- [ ] **Transformer from Scratch Guide**
  - Positional encoding
  - Self-attention
  - Encoder block
  - Decoder block (masked attention)
  - Complete transformer
  - Mini-transformer project (addition task)

#### Module 11: Language Models (5 coding tasks)
- [ ] **Language Model Guide**
  - Bigram model
  - Decoder-only transformer (GPT architecture)
  - Temperature sampling
  - Top-k and nucleus sampling

#### Module 12: Training Techniques (5 coding tasks)
- [ ] **Advanced Training Guide**
  - Learning rate schedules (step, exponential, cosine)
  - Warmup implementation
  - Gradient clipping
  - Loss landscape visualization

#### Module 13: Regularization (1 coding task)
- [ ] **Regularization Guide**
  - L1, L2, elastic net
  - Implementation in training loop

#### Module 14: Autograd Framework (12 coding tasks)
- [ ] **Build Your Own PyTorch Guide**
  - Tensor class with autograd
  - Automatic differentiation
  - Basic operations (+, -, *, /, matmul)
  - Layers (Linear, Conv2d, MaxPool, Dropout, BatchNorm)
  - Activations with gradients
  - Optimizer base class + SGD, Adam
  - Loss functions
  - DataLoader

#### Capstone 1: Paper Implementations (11 coding tasks)
- [ ] **Classic Papers Implementation Guide**
  - LeNet-5
  - AlexNet
  - ResNet-18
  - U-Net
  - Seq2Seq
  - Attention
  - BERT
  - GPT-2
  - GAN
  - VAE

---

## 🗂️ File Structure

```
ml-roadmap/
├── index.html (main roadmap app)
├── exercises.html (exercise tracker)
├── math_foundations_roadmap.md (source content)
├── STATUS.md (this file)
│
├── guides/
│   ├── postgresql_setup.html + .md ✅
│   ├── docker_setup.html + .md ✅
│   ├── ml_environment_setup.html + .md ✅
│   ├── etl_pipeline_guide.html + .md ✅
│   ├── linear_algebra_coding_guide.html + .md ✅
│   ├── optimization_coding_guide.md ⚠️ (need .html)
│   │
│   ├── exercises/
│   │   ├── linear_algebra_exercises.html + .md ✅
│   │   └── calculus_gradients_exercises.html + .md ✅
│   │
│   └── solutions/
│       ├── linear_algebra_solutions.html + .md ✅
│       └── calculus_gradients_solutions.html + .md ✅
│
└── [other files]
    ├── Alan_Beaulieu-Learning_SQL-EN.pdf (NOT committed)
    ├── index2.html (NOT committed - backup)
    └── viewer.html (NOT committed - old version)
```

---

## 🔧 Technical Implementation Details

### Progress Tracking
- Uses `localStorage` for client-side persistence
- Key format: `ml-roadmap-progress`
- GitHub sync via API (requires token for edit mode)
- Progress saved to `progress.json` in repository

### Task Rendering
- Tasks support HTML via `dangerouslySetInnerHTML`
- Guide links embedded in task strings with inline styles
- Example: `"Setup PostgreSQL (<a href='guides/postgresql_setup.html' target='_blank' style='color: #2563eb; font-weight: 600;'>📖 Setup Guide</a>)"`

### Guide HTML Template Pattern
All guides use same pattern:
1. Load markdown file via `fetch('guide_name.md')`
2. Parse with `marked.js`
3. Render into `.markdown-body` div
4. Consistent styling (Tailwind CSS)
5. Navigation back to roadmap

**This requires BOTH `.html` and `.md` files to be uploaded to GitHub!**

---

## 📊 Guide Creation Template

When creating a new guide, follow this pattern:

### 1. Create `.md` file with content
```markdown
# Guide Title

**Time:** X hours
**Difficulty:** Level
**Prerequisites:** List

## What You'll Build
- Item 1
- Item 2

## Part 1: Topic
[Detailed content]

## Verification Checklist
- [ ] Item 1
- [ ] Item 2

## Resources
- [Link 1](url)
```

### 2. Create `.html` wrapper
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Guide Title - ML Roadmap</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Styling from existing guides -->
</head>
<body>
    <!-- Header with back button -->
    <!-- Content container -->
    <script>
        fetch('guide_name.md')
            .then(response => response.text())
            .then(markdown => {
                document.getElementById('content').innerHTML = marked.parse(markdown);
            });
    </script>
</body>
</html>
```

### 3. Link in `index.html`
Update task text:
```javascript
"Code: Task description (<a href='guides/guide_name.html' target='_blank' style='color: #10b981; font-weight: 600; text-decoration: underline;'>💻 Coding Guide</a>)"
```

### 4. Commit both files
```bash
git add guides/guide_name.html guides/guide_name.md
git commit -m "Add [topic] coding guide"
git push
```

---

## 🎨 Color Scheme

- **Blue (#2563eb)**: Setup guides, system installations
- **Green (#10b981)**: Coding/project guides, exercises
- **Emerald (#10b981)**: Math-specific content
- **Yellow (#fbbf24)**: Important callouts
- **Red (#dc2626)**: Inline code snippets

---

## 🚀 Next Session Tasks

**High Priority:**
1. Create HTML version of `optimization_coding_guide.md`
2. Link optimization guide to tasks in index.html
3. Create Neural Networks from Scratch guide (Module 6)
4. Create CNN Implementation guide (Module 7)
5. Create Transformer from Scratch guide (Module 10)

**Medium Priority:**
6. Create Probability & Information Theory guide (Modules 4-5)
7. Create Attention Mechanisms guide (Module 9)
8. Create Autograd Framework guide (Module 14)

**Low Priority:**
9. Create remaining specialized guides
10. Consider creating video walkthroughs
11. Add more interactive elements

---

## 📝 Notes for Next Session

### Current State
- All files successfully pushed to GitHub (last push: 2025-10-28)
- Main roadmap fully functional
- 3 setup guides complete
- 2 project guides complete (1 needs HTML)
- Exercise system complete
- ~15 more coding guides needed for full math roadmap coverage

### Known Issues
- None currently

### Decisions Made
1. Use external guide files (not inline) for maintainability
2. Both `.html` and `.md` files needed for GitHub Pages
3. Color-code guides by type (blue=setup, green=coding)
4. Create comprehensive guides, not just snippets
5. Include projects/capstones in guides where possible

### Questions for User
- Which guides to prioritize? (Neural Networks, CNNs, Transformers most impactful)
- Should we create video walkthroughs?
- Any specific format preferences?

---

## 📚 Resources Referenced

- Mathematics for Machine Learning (MML) book
- 3Blue1Brown video series
- Understanding Deep Learning book
- CS231n course materials
- Fast.ai courses
- PyTorch documentation
- NumPy documentation

---

## 🔗 Quick Links

- [GitHub Repo](https://github.com/luisdecunto/ml-roadmap)
- [Live Site](https://luisdecunto.github.io/ml-roadmap/)
- [MML Book](https://mml-book.github.io/)
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown)
- [Fast.ai](https://www.fast.ai/)

---

**Remember:** When resuming, check this STATUS.md file first to understand what's been completed and what needs to be done next!
