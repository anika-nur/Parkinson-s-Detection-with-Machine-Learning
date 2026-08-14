# Parkinson's Detection with Machine Learning

Detecting Parkinson's Disease (PD) from **voice recordings** using supervised machine
learning. The project analyzes 195 sustained-phonation recordings from 31 individuals to
find correlations between vocal parameters and Parkinson's Disease, and trains several
classifiers to distinguish healthy subjects from those with PD — a low-cost, non-invasive
route toward **early screening**.

> 📄 **This work has been peer-reviewed and published for presentation in *Neurology*.**
> See the [Publication](#-publication) section below.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Status](https://img.shields.io/badge/status-published-success)

---

## 🧠 Background

Parkinson's Disease affects the motor system, and one of its early and measurable symptoms
is **dysphonia** — impaired voice production. Because roughly 90% of people with PD show
some vocal impairment, acoustic measurements of sustained vowels are a promising, painless
biomarker for early detection. This project uses biomedical voice measurements to build
and compare classification models for PD.

---

## 📊 Dataset

The project uses the **Oxford Parkinson's Disease Detection Dataset** from the UCI Machine
Learning Repository.

- **Source:** UCI Machine Learning Repository — [DOI: 10.24432/C59C74](https://doi.org/10.24432/C59C74)
- **Instances:** 195 voice recordings
- **Subjects:** 31 people (23 with Parkinson's Disease, 8 healthy)
- **Attributes:** 22 biomedical voice measures per recording + a target label
- **Target:** `status` — `1` = Parkinson's Disease, `0` = healthy


### Notebooks

**`dsProject.ipynb` — Data Science / Exploratory Data Analysis**
The analysis notebook. It loads the raw dataset, checks for missing and duplicate values,
drops the non-predictive `name` column, and renames the raw MDVP columns to readable labels.
It then explores the data: summary statistics and feature ranges, a correlation heatmap of
the vocal measures, distribution plots, and comparisons between the PD group (`status = 1`)
and the healthy group (`status = 0`) — for example, examining the range of average vocal
frequency in PD-detected subjects and whether the noise ratios `HNR` and `NHR` are
proportionally related. This notebook establishes *what the voice features look like* and
which ones separate the two groups.

**`ML.ipynb` — Machine Learning Modeling**
The modeling notebook. Building on the same cleaned data, it splits the set 70/30 into
training and test data, applies `StandardScaler` feature scaling, and then trains and
evaluates five supervised classifiers:

1. **Logistic Regression** — baseline linear classifier
2. **K-Nearest Neighbors** — with an error-rate-vs-K sweep to choose *k*
3. **Support Vector Classifier (SVC)** — with the RBF kernel
4. **Decision Tree Classifier** — tuned depth and leaf constraints
5. **Random Forest Classifier** — ensemble of trees (best performer)

Each model is scored with accuracy, precision, recall, F1-score and a confusion matrix
(using cross-validation) and the results are compared to select the best-performing model.

---


## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Open `dsProject.ipynb` for the exploratory analysis, or `ML.ipynb` to reproduce the model
training and results.

> **Note:** The notebooks were originally developed in Google Colab and mount Google Drive
> to load the data. To run locally, download the dataset from the
> [UCI link](https://doi.org/10.24432/C59C74) and update the file path in the data-loading
> cell.

---

## 🛠️ Built With

- **Python** — pandas, NumPy
- **scikit-learn** — Logistic Regression, KNN, SVC, Decision Tree, Random Forest, preprocessing & metrics
- **matplotlib** & **seaborn** — visualization
- **Jupyter / Google Colab**

---

## 📄 Publication

This research was peer-reviewed and **published in *Neurology***.

🔗 **Read the paper:** https://www.neurology.org/doi/10.1212/WNL.0000000000204767
(DOI: [10.1212/WNL.0000000000204767](https://doi.org/10.1212/WNL.0000000000204767))

If you use this work, please cite the published article.

---

## 📚 Citations

**Dataset**

> Little, M. (2007). *Parkinsons* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C59C74

> Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A. E., & Moroz, I. M. (2007).
> Exploiting nonlinear recurrence and fractal scaling properties for voice disorder
> detection. *BioMedical Engineering OnLine*, 6, 23.

---
