# 🚀 Regression & Optimization Algorithm Analysis 

A comprehensive study on the convergence behaviors of various optimization algorithms within a binary classification framework, utilizing high-dimensional NLP embeddings.

## 📝 Project Overview
This project investigates the efficiency of different optimization techniques applied to a regression model. Using synthetic data generated via Large Language Models (LLMs), we analyze how different algorithms navigate the loss landscape to find optimal weight parameters.

### 🛠 Technical Specification
- **Model Architecture:** A binary classifier based on the $y = \tanh(w \cdot x)$ function.
- **Data Source:** 200 high-quality synthetic QA pairs generated using the `Turkish-Gemma-9b-T1` model.
- **Feature Engineering:** Textual data transformed into 768-dimensional semantic vectors using the `turkish-e5-large` embedding model.
- **Dimensionality:** Input features are concatenated, resulting in a $(2d + 1)$ dimensional weight vector.

## 📊 Optimization Benchmarking
We compared three fundamental optimization algorithms across 5 different random weight initializations ($w_0$):

1. **Gradient Descent (GD):** Full-batch updates for stable convergence.
2. **Stochastic Gradient Descent (SGD):** High-frequency updates with inherent noise for local minima escape.
3. **Adam Optimizer:** Adaptive moment estimation for dynamic learning rate adjustment.

### 📉 Performance Analysis
The training process was monitored through various metrics including execution time, update counts, and final accuracy.

![Loss & Accuracy Curves](output/süre_loss_grafik.png)
*Figure 1: Comparison of Training Loss and Success Rates over Time/Epochs.*

### 🗺 Weight Trajectory Visualization (T-SNE)
To understand the optimization path, high-dimensional weight updates ($w_{1:t}$) were reduced to 2D space using **T-SNE**. This visualization reveals the unique "search" pattern of each algorithm.

![T-SNE Trajectories](output/tsne_grafik.png)
*Figure 2: Optimization trajectories in the weight space across different initializations.*

## 📂 Repository Structure
- `Regression-Optimization-Analysis.ipynb`: Complete source code (Data prep, Training, Analysis).
- `/data/`: Contains `raw/` (CSV text files) and `processed/` (NumPy embeddings).
- `/output/`: Visualization exports including the performance plots and T-SNE mappings.
- `rapor.pdf`: Detailed technical report covering the mathematical derivations and empirical findings.

---
*Developed by a Computer Engineering student at Yıldız Technical University.*
