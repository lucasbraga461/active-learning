# Active Learning Educational Repository

Welcome to the **Active Learning Educational Repository**! This project is designed to teach data scientists and ML practitioners how to **build, apply, and experiment** with Active Learning techniques in Python.

This repository includes:
- An annotated Jupyter Notebook (`notebook.ipynb`) that walks through a full active learning pipeline.
- An article (*Active Learning in Machine Learning: Techniques and its Pros and Cons*) explaining the theory behind the practice.

---

## 🔍 What is Active Learning?

**Active Learning** is a machine learning paradigm where the model **actively selects the most informative data points** to be labeled, rather than training on a randomly labeled dataset. It is especially useful when labeling data is expensive or time-consuming.

**Key techniques:**
- Uncertainty Sampling
- Diversity Sampling
- Query-by-Committee (QBC)

For a deep dive into each method, pros, and cons, check out the article.

---

## 📚 Repository Structure

### 1. `notebook.ipynb`

This is the core tutorial notebook that **generates a synthetic dataset** and walks you through the full **active learning workflow**.

- A **binary classification dataset** is generated with 10,000 samples and 9 features. The data is synthetic and created entirely for educational purposes.
- The process starts by training a **Logistic Regression** model on just **300 labeled samples**.
- The notebook then walks through **multiple iterations of active learning**, where the goal is to improve the model by selectively adding new labeled data based on different strategies:
  - **Iterations 1–3** use **Uncertainty Sampling**.
  - When uncertainty sampling stops showing meaningful improvement, the notebook **switches to KNN-based Diversity Sampling** in Iterations 4–5.
  - Finally, **Query-by-Committee** is used in Iteration 6.

- After each iteration, the model is retrained and evaluated using **F1-score**, **Precision**, and **Recall** to observe improvements.
- **Nested Cross-Validation** with `GridSearchCV` is used throughout to ensure fair and robust model evaluation.

> ⚠️ **Important Note:** Because the dataset is synthetic and random, you may not see dramatic improvements at every stage. The purpose is to show **how and when** to switch sampling strategies — not to optimize performance on fake data.  
> We encourage you to try these techniques on your own **real-world dataset** to see their full effect.

Each section is carefully commented to help you understand not just **what** is happening, but **why** it's done that way.

---

### 🔁 Iteration Breakdown

#### 🔹 **Iterations 1–3 (Sections 2–16): Uncertainty Sampling**
- Selects unlabeled samples where the model is most uncertain (probability ~ 0.5).
- Labeled samples grow from 300 → 436.
- Covers:
  - `## 2` to `## 16`

#### 🔸 **Iterations 4–5 (Sections 17–27): Diversity Sampling (k-NN based)**
- Uses **k-Nearest Neighbors** to choose diverse samples in dense regions.
- Helps improve data coverage and reduce redundancy.
- Covers:
  - `## 17` to `## 27`

#### 🧠 **Iteration 6 (Sections 28–32): Query by Committee (QBC)**
- Trains multiple models and selects samples where the models disagree most.
- Implements a stacking meta-learner for ensemble predictions.
- Covers:
  - `## 28` to `## 32`

---

### 2. `article - Active Learning`
This is a research-style article that covers:
- Theoretical foundations of active learning
- Visual diagrams and summaries of popular sampling techniques
- Pros and cons of each method

---

## 🚀 How to Use This Repository

1. Clone the repo and open `notebook.ipynb` in Jupyter or VSCode.
2. Follow the walkthrough to:
   - Generate data
   - Train models
   - Apply active learning loop
   - Evaluate performance improvements
   - Run nested cross-validation
3. Refer to `article - Active Learning` to understand the **why** behind each step.

---

## 🔧 Requirements

To run the code, make sure you have the following installed:
```bash
pip install numpy pandas scikit-learn matplotlib
