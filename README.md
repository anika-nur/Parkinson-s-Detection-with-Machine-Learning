# Parkinson's Detection with Machine Learning

Detecting Parkinson's Disease (PD) from **voice recordings** using supervised machine
learning. The project analyzes 195 sustained-phonation recordings from 31 individuals to
find correlations between vocal parameters and Parkinson's Disease, and trains several
classifiers to distinguish healthy subjects from those with PD — a low-cost, non-invasive
route toward **early screening**.

> 📄 **This work has been peer-reviewed and published in *Neurology*.**
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

### Feature groups

| Group | Original columns | What it captures |
|-------|------------------|------------------|
| Fundamental frequency | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` | Average, maximum and minimum vocal pitch |
| Jitter (frequency variation) | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` | Cycle-to-cycle variation in pitch |
| Shimmer (amplitude variation) | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` | Cycle-to-cycle variation in loudness |
| Noise ratios | `NHR`, `HNR` | Ratio of noise to tonal components |
| Nonlinear / complexity | `RPDE`, `D2`, `DFA`, `spread1`, `spread2`, `PPE` | Nonlinear dynamical and signal-complexity measures |

> Column names are renamed to more readable labels (e.g. `av_voc_hz`, `max_voc_hz`,
> `frq_percent`, `amp`, …) inside the notebooks.

---

## 🔬 Methodology

1. **Data loading & cleaning** — load the dataset, verify there are no null values, drop the
   non-predictive `name` identifier, and rename columns for readability.
2. **Exploratory data analysis** — inspect feature ranges, distributions, and a correlation
   heatmap; compare vocal measures between PD and healthy groups (e.g. HNR vs. NHR).
3. **Train/test split** — 70% training / 30% testing.
4. **Feature scaling** — `StandardScaler` applied before training the distance- and
   margin-based models.
5. **Model training & evaluation** — train five classifiers and compare them using accuracy,
   precision, recall, F1-score and confusion matrices (with cross-validation).

---

## 🤖 Models & Results

Five supervised classifiers were trained and evaluated on the held-out test set:

| Model | Test Accuracy |
|-------|:-------------:|
| Logistic Regression | 91.5% |
| Decision Tree Classifier | 91.5% |
| K-Nearest Neighbors (k = 1) | 93.2% |
| Support Vector Classifier (SVC) | 93.2% |
| **Random Forest Classifier** ⭐ | **96.6%** |

The **Random Forest Classifier** performed best, reaching **~96.6% test accuracy** with
balanced precision and recall across both classes — making it the strongest candidate for
voice-based PD screening in this study.

---

## 📁 Repository Structure

```
.
├── dsProject.ipynb   # Data exploration & analysis
├── ML.ipynb          # Model training & evaluation
└── README.md
```

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
(using cross-validation), and the results are compared to select the best-performing model.

---

## 🚀 Getting Started

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run

```bash
git clone https://github.com/anika-nur/parkinson-s-detection-with-machine-learning.git
cd parkinson-s-detection-with-machine-learning
jupyter notebook
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

## 📝 License

This project is available for academic and research use. Please cite the publication and
dataset above if you build on this work.
