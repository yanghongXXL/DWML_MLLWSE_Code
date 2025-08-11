# DWML_MLLWSE_Code
Source code and reproduction materials for DWML-EFS and MLLWSE — feature selection and stacked ensemble learning methods.
This repository provides the **full implementation** of two proposed multi-label learning methods:  

1. **Dynamic Weighted Multi-Label Ensemble Feature Selection (DWML-EFS)** — integrates information-theoretic approaches to reduce computational costs while improving feature selection efficiency.  
2. **Multi-Label Linearly Weighted Stacked Ensemble (MLLWSE)** — enhances predictive performance via a linearly weighted stacking strategy across multiple base learners.  

Each method includes a **Jupyter Notebook** for step-by-step reproduction of the experimental results reported in our paper.

---

## 📦 Requirements
- Python 3.8+  
- `scikit-learn`  
- `scikit-multilearn`  
- `numpy`  
- `scipy`  
- `pandas`  

---

## 📂 Repository Structure
```
DWML-EFS/  
│   classes/                 # Supporting modules for DWML-EFS
│   results/                 # Output results from experiments
│   data.py                  # Data handling utilities
│   Demo_CHD_BR_RF.ipynb     # Demonstration with CHD dataset
│   evaluation.py            # General evaluation functions
│   evaluationARAM.py        # Evaluation with ARAM model
│   evaluationKNN.py         # Evaluation with KNN model
│   FinalCode.ipynb          # Main reproduction notebook
│   mlmetrics.py             # Multi-label evaluation metrics
│   mlsmote.py               # SMOTE variant for multi-label data

MLLWSE/  
│   Demo_MLLWSEStacking.ipynb  # Main reproduction notebook
│   lasso.py                   # LASSO optimization
│   mlmetrics.py               # Multi-label evaluation metrics
│   read_matfile.py            # Load .mat datasets
│   utils.py                   # Utility functions
```

---

## 📥 Dataset Preparation
Some example datasets are included in `DWML-EFS/results/` and `MLLWSE/`.  
Additional datasets can be downloaded from:  
- [Mulan](http://mulan.sourceforge.net/datasets.html)  
- [KDIS](http://www.uco.es/kdis/mllresources/)  
- [Meka](http://meka.sourceforge.net/)  

**Supported formats:**  
- `.arff` — load using `read_arff.py` (if available)  
- `.mat` — load using:
```bash
python MLLWSE/read_matfile.py
```

---

## 🚀 Running the Code

### 1. DWML-EFS

Run the demonstration notebook:
```bash
jupyter notebook Demo_CHD_BR_RF.ipynb
```

### 2. MLLWSE
Run the main reproduction notebook:
```bash
cd MLLWSE
jupyter notebook Demo_MLLWSEStacking.ipynb
```

---

## 📊 Evaluation Metrics
Implemented in `mlmetrics.py` (both folders):  
- Hamming Loss  
- Accuracy  
- Ranking Loss  
- F1-score  
- Macro-F1  
- Micro-F1  

---

## 📈 Additional Analyses
Some scripts also include:
- **Parameter sensitivity experiments**  
- **Convergence analysis**  
- **Statistical significance tests** (e.g., Friedman test)

---
