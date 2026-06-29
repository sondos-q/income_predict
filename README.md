# Adult Income Classification

A machine learning project that predicts whether an individual earns more or less than $50K/year using the [UCI Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult).

---

## Dataset

The dataset (`adult.csv`) contains **48,842 records** and **15 features** collected from US census data.

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age of the individual |
| `workclass` | categorical | Employment type (Private, Gov, Self-emp, etc.) |
| `fnlwgt` | int | Census sampling weight |
| `education` | categorical | Highest education level |
| `education-num` | int | Education level as a number |
| `marital-status` | categorical | Marital status |
| `occupation` | categorical | Job category |
| `relationship` | categorical | Family role |
| `race` | categorical | Race |
| `sex` | binary | Gender (Male/Female) |
| `capital-gain` | int | Capital gains |
| `capital-loss` | int | Capital losses |
| `hours-per-week` | int | Weekly working hours |
| `native-country` | categorical | Country of origin |
| `income` | target | `<=50K` or `>50K` |

---

## Project Structure

```
├── adult.csv               # Raw dataset
├── notebook.ipynb          # Main Jupyter notebook
└── README.md
```

---

## Preprocessing

1. **Duplicate removal** — 29 duplicate rows dropped
2. **Missing values** — `workclass`, `occupation`, and `native-country` had missing values and `?` placeholders, filled with the column mode
3. **Label cleaning** — Inconsistent income labels (`<=50K.`, `>50K.`) standardized
4. **Encoding**
   - `sex` binary-encoded (Male → 1, Female → 0)
   - All other categorical features one-hot encoded via `pd.get_dummies`
5. **Scaling** — Features scaled to [0, 1] with `MinMaxScaler`
6. **Train/test split** — 80% train / 20% test (stratified shuffle)

**Class distribution after cleaning:**

| Label | Count |
|---|---|
| `<=50K` | 37,128 (76%) |
| `>50K` | 11,685 (24%) |

> Note: The dataset is imbalanced. SMOTE oversampling is available in the notebook (commented out) and can be enabled to address this.

---

## Models & Results

| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1 (>50K) |
|---|---|---|---|---|
| **Random Forest** | **85.59%** | 0.73 | 0.61 | 0.66 |
| Logistic Regression | 85.28% | 0.72 | 0.59 | 0.65 |
| Decision Tree | 84.82% | 0.75 | 0.52 | 0.61 |
| KNN (k=5) | 82.56% | 0.64 | 0.56 | 0.60 |

Random Forest achieved the best overall accuracy and F1-score for the minority class (`>50K`).

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

---

## Usage

Open and run `notebook.ipynb` in Jupyter. Sections are ordered as:

1. Data loading & exploration
2. Cleaning & preprocessing
3. Encoding & scaling
4. Model training (Random Forest → KNN → Logistic Regression → Decision Tree)
5. Deep Learning (TensorFlow)

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `tensorflow` | Deep learning model |
