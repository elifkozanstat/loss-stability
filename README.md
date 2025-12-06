# Submission Notice

This repository accompanies the manuscript:

**"Evaluating Loss-Function Stability in Edge-Based Classification of Imbalanced Images"**

submitted to *MDPI Applied System Innovations*.

The manuscript is **not yet published** and may undergo revisions during peer review.  
A permanent Zenodo DOI will be added upon acceptance.


# loss-stability

This repository provides the complete codebase used to analyze loss-function stability in an edge-based image-classification pipeline under severe class imbalance. The experiments compare:

- **Soft-Fβ surrogate loss (β = 2)**  
- **Weighted Binary Cross-Entropy (BCE)**  

within a controlled preprocessing framework involving grayscale conversion, median filtering, and Sobel edge extraction.

The study evaluates probability-level behavior, threshold sensitivity, F₁ performance, ROC/PR curves, and stability failure modes in an Apis–Bombus bee classification task.

---

## 🧪 Features of This Repository

- **Preprocessing pipeline** (median → Sobel → flatten 40,000-D vectors)  
- **Soft-Fβ (β = 2) differentiable surrogate loss**  
- **Weighted BCE baseline**  
- **Compact neural classifier**  
- **Threshold sensitivity analysis**  
- **Probability distribution diagnostics**  
- **ROC and PR curve generation**  

The code exactly matches the methodology described in the manuscript.

---

##  Repository Structure
loss-stability/
│
├── preprocessing.py              # Edge extraction pipeline (median + Sobel) and DataLoaders
├── soft_fbeta_loss.py            # Soft-Fβ loss and weighted BCE wrapper
├── bce_training.py               # Compact neural classifier and training loop
├── threshold_evaluation.py       # Threshold search, probability diagnostics, ROC/PR curves
│
├── results_notebook.ipynb        # Reproducible experiments for the manuscript
├── requirements.txt              # Environment dependencies
└── README.md                     # Project description and submission notice



