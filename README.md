# Neural Network Optimization from Scratch (NumPy Implementation) 🚀

This project is a comprehensive implementation of regression-based classification and various optimization algorithms built entirely from scratch using **NumPy**, without relying on high-level autograd frameworks like PyTorch or TensorFlow.

The study focuses on the entire end-to-end pipeline: from synthetic QA dataset generation and feature extraction using transformer-based embeddings to 2D trajectory visualization of high-dimensional optimization paths via t-SNE.

---

## 🎯 Project Goals

- **Mathematical Depth:** Understanding the core mechanics of optimization (manual backpropagation, chain rule, and matrix calculus) by implementing the $y = \tanh(w \cdot x)$ single-layer model from scratch.
- **Optimization Benchmarking:** Empirically comparing the performance, convergence speed, and stability of **Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)**, and **Adam**.
- **Loss Landscape Analysis:** Visualizing how different optimizers navigate high-dimensional parameter spaces ($2049$-dim) by projecting optimization trajectories into 2D using **t-SNE**.

---

## 🛠️ Technologies & Methodology

### 1. Synthetic Data Generation & Balancing
- **Generation:** Balanced QA dataset generated across 10 distinct knowledge domains.
- **Dataset Scale:** 1,000 unique questions resulting in **2,000 labeled instances** (1,000 correct `+1`, 1,000 incorrect `-1`).
- **Data Quality Control:** Strict answer length and syntactic parity to prevent length-based heuristic bias. Split into **80% Train (1,600 rows)** and **20% Test (400 rows)** grouped at the question level.

### 2. Semantic Embeddings & Feature Engineering
- **Model:** `ytu-ce-cosmos/turkish-e5-large` via Sentence-Transformers.
- **Vector Space:** Converted text into **1024-dimensional** L2-normalized embeddings using task-specific instruction prefixes (`query:` for questions, `passage:` for answers).
- **Feature Vector:** Concatenated question and answer embeddings with an explicit bias term:
  $$\mathbf{x} = [\mathbf{e}_{\text{question}} \,\Vert{}\, \mathbf{e}_{\text{answer}} \,\Vert{}\, 1]^T \in \mathbb{R}^{2049}$$

### 3. Model Architecture
- **Task:** Binary Classification via Tanh-activated Regression.
- **Formulation:** $\hat{y} = \tanh(w^T x)$, mapped against ground truth targets $y \in \{-1, +1\}$.
- **Loss Function:** Mean Squared Error (MSE), with exact analytical gradients computed and applied manually per optimizer.

---

## 📊 Optimization Benchmarks

| Optimizer | Learning Rate | Convergence Speed | Stability | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **GD** | $\eta = 0.1$ | 🔴 Slow | 🟢 Very High | Full-batch gradient updates; plateaus early and converges slowly on non-convex/flat regions. |
| **SGD** | $\eta = 0.1$ | 🟡 Medium | 🔴 Low | Step-by-step stochastic updates; fast initial descent but high variance and trajectory oscillations. |
| **Adam** | $\eta = 0.01$ | 🟢 Very Fast | 🟢 High | First & second moment estimation with adaptive per-weight step sizes; superior convergence and lowest final MSE. |

---

## 📈 Results & Visualizations

### 1. Loss & Accuracy Dynamics
- **Adam** achieves the steepest and most stable loss descent ($\sim 0.40$ MSE) while delivering the highest sustained test generalization accuracy ($\sim 55\text{--}56\%$).
- **SGD** demonstrates rapid initial drop followed by characteristic stochastic oscillations around local minima.
- **GD** suffers from vanishing progress in batch updates, remaining near the baseline accuracy.

### 2. Optimization Trajectories (t-SNE Analysis)
- Analyzes parameter evolution vectors ($w_{1:t} \in \mathbb{R}^{2049}$) projected down to 2D.
- Demonstrates the direct, momentum-guided trajectory of Adam versus the noisy exploratory zig-zags of SGD.

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── yeni_sorular.csv
│   └── processed/                # Precomputed 1024-dim .npy arrays
├── scripts/
│   └── build_embeddings.py       # Embedding extraction pipeline (e5-large)
├── Regression-Optimization-Analysis.ipynb  # Core models, benchmarks & plots
├── rapor.pdf                     # Detailed mathematical derivation & report
└── README.md
