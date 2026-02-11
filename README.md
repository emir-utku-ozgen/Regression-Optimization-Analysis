# Neural Network Optimization from Scratch (NumPy Implementation) 🚀

This project is a comprehensive implementation of regression-based classification and various optimization algorithms built entirely from scratch using **NumPy**, without relying on high-level frameworks like PyTorch or TensorFlow. 

The study focuses on the entire pipeline: from synthetic data generation using a local LLM to 2D visualization of high-dimensional optimization trajectories using **t-SNE**.

## 🎯 Project Goals

* **Mathematical Depth:** Understanding the core mechanics of AI (manual backpropagation, chain rule, and matrix calculus) by implementing the $y = \tanh(w \cdot x)$ model from scratch.
* **Optimization Benchmarking:** Comparing the performance, convergence speed, and stability of **Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)**, and **Adam**.
* **Loss Landscape Analysis:** Visualizing how different optimizers navigate the high-dimensional weight space by reducing updates to 2D using **t-SNE**.

## 🛠️ Technologies & Methodology

### 1. Synthetic Data Generation
* **Model:** `Turkish-Gemma-9b-T1` via Ollama.
* **Task:** Generated 200 high-quality Question-Answer pairs labeled as incorrect (-1) and correct (+1).

### 2. Semantic Embeddings & Feature Engineering
* **Model:** `ytu-ce-cosmos/turkish-e5-large`.
* **Vectorization:** Converted text into a 768-dimensional semantic space.
* **Input Pipeline:** Concatenated Question + Answer vectors with an added bias term, resulting in a **$(2d + 1)$** dimensional input vector.

### 3. Model Architecture
* **Type:** Binary Classification via Regression.
* **Activation:** Hyperbolic Tangent ($\tanh$) for both model output and weight scaling.
* **Logic:** Implemented manual weight updates based on the derivative of the loss function.

## 📊 Optimization Benchmarks

| Optimizer | Convergence Speed | Stability | Characteristics |
| :--- | :--- | :--- | :--- |
| **GD** | 🔴 Slow | 🟢 Very High | Processes entire dataset. Smooth but slow path. |
| **SGD** | 🟡 Medium | 🔴 Low | High-frequency updates with stochastic noise. |
| **Adam** | 🟢 Very Fast | 🟢 High | Uses Momentum and Adaptive LR. Superior convergence. |

## 📈 Results & Visualizations

### 1. Optimization Trajectories (t-SNE Analysis)
The visualization below shows how different algorithms navigate the high-dimensional weight space ($w_{1:t}$).
* **Adam** takes the most direct path to the global minimum.
* **SGD** shows characteristic oscillations (zig-zags).
* **GD** follows a steady, linear trajectory.

![t-SNE Optimization Trajectories](output/tsne_grafik.png)

### 2. Loss & Accuracy Curves
Comparative analysis of training loss decay and success rates over time for each optimizer.

![Loss and Accuracy Performance](output/süre_loss_grafik.png)

## 📂 Repository Structure
* `Regression-Optimization-Analysis.ipynb`: Main source code & experiment logs.
* `/data/`: Contains `raw/` (CSV) and `processed/` (Embeddings) datasets.
* `/output/`: Exported visualization plots (t-SNE and Loss curves).
* `rapor.pdf`: Detailed technical report on mathematical derivations and findings.

---
*Developed as part of the Computer Engineering curriculum at Yıldız Technical University.*
