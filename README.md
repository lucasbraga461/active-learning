# 🧠 Active Learning Educational Repository

Welcome to the **Active Learning Educational Repository** — a hands-on project built to help data scientists and ML practitioners **understand, apply, and experiment** with Active Learning techniques in Python.

---

## 📌 What's Inside

This repository contains:

- 📝 An **annotated Jupyter Notebook** (`notebook.ipynb`) walking through a complete Active Learning pipeline  
- 📄 A full-length **article** (*Active Learning in Machine Learning: Techniques and its Pros and Cons*) that explains the theory, techniques, and trade-offs

---

## 🔍 What is Active Learning?

**Active Learning** is a machine learning strategy where the model itself chooses **which data points to label**, aiming to maximize learning efficiency. This is especially useful when labeling is costly, time-consuming, or requires domain expertise.

Key techniques covered in this project:
- **Uncertainty Sampling**
- **Diversity Sampling**
- **Query-by-Committee (QBC)**

You’ll find both **code implementations** and **conceptual explanations** for each method.

---

## 📁 Repository Structure

### `notebook.ipynb`

A step-by-step notebook that:
- Generates a synthetic **binary classification dataset** with 10,000 samples and 9 features
- Starts with just **300 manually labeled samples**
- Iteratively trains a **Logistic Regression** model using:
  - **Uncertainty Sampling** (Iterations 1–3)
  - **k-NN Diversity Sampling** (Iterations 4–5)
  - **Query-by-Committee (QBC)** with model stacking (Iteration 6)
- Uses **Nested Cross-Validation** with `GridSearchCV` for robust evaluation
- Measures performance via **F1-Score**, **Precision**, and **Recall** after each iteration

> ⚠️ *Note*: Since the dataset is synthetic, improvements may not be dramatic. The goal is to demonstrate the **workflow and strategy-switching logic**, not to optimize performance on artificial data.

---

## 🔁 Iteration Breakdown

| Iteration(s) | Sampling Strategy       | Description                                                                 |
|--------------|--------------------------|-----------------------------------------------------------------------------|
| 1–3          | **Uncertainty Sampling** | Selects samples with prediction probabilities near 0.5                     |
| 4–5          | **Diversity Sampling**   | Uses k-NN to sample diverse points from sparse regions                     |
| 6            | **Query-by-Committee**  | Trains multiple models; samples where predictions diverge                  |

Sections in the notebook are clearly labeled (e.g., `## 2`, `## 16`, etc.) for easy navigation.

---

### `article - Active Learning`

A well-researched, reference-backed article that explores:
- Theoretical foundations of Active Learning
- Sampling strategies (uncertainty, diversity, QBC)
- Use cases across domains (NLP, CV, fraud detection)
- Visual diagrams + pros and cons

---

## 🚀 Getting Started

1. **Clone this repo** and open `notebook.ipynb` in Jupyter, VSCode, or Google Colab.
2. Follow along:
   - Generate and explore the data
   - Train baseline models
   - Run the Active Learning loop (with 3 strategies)
   - Evaluate performance across iterations
3. Read the accompanying article to understand the **“why”** behind each method.

---

## 🧰 Requirements

Install the required dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## 🔗 Citation

If you use or adapt this work, please cite the repository as:

> Benevides e Braga, L. (2025). *Active Learning Educational Repository*. GitHub. https://github.com/lucasbraga461/active-learning