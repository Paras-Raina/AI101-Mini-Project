# AI101 Mini-Project: Diabetes Prediction Using Logistic Regression

An end-to-end clinical machine learning study predicting the likelihood of diabetes in patients based on demographic factors, diagnostic blood markers, and cardiovascular history using regularized Logistic Regression.

## 👥 Group 53 — Team Members & Roles

* **Paras Raina (2026EEB1389)** — PPT Lead
* **Lakkireddy Thulasiram Reddy (2026EEB1371)** — Video Lead
* **Manav Garg (2026MCB1543)** — Documentation Lead
* **Dhonnar Satyam Ramesh (2026AIB1031)** — Execution Lead

## 📌 Executive Summary

Diabetes mellitus is a chronic metabolic disorder marked by persistent hyperglycemia. Early identification of high-risk individuals enables preventive interventions that mitigate severe macrovascular and microvascular complications.

This project analyzes patient diagnostic metrics from clinical records and trains a binary **Logistic Regression** classifier to estimate diabetes risk. The entire experimental workflow—exploratory data analysis, duplicate cleaning, stratified sampling, zero-leakage feature standardization, model training, and diagnostic metric evaluation—is self-contained within an interactive, reproducible Jupyter Notebook.

## 📊 Dataset Overview

* **Source:** [Kaggle Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
* **Raw Dimensions:** 100,000 rows × 9 columns
* **Missing Values:** 0
* **Duplicate Rows Removed:** 3,854
* **Clean Dataset Dimensions:** 96,146 rows × 9 columns
* **Target Column:** `diabetes` (`0` = Non-Diabetic, `1` = Diabetic)
* **Class Distribution (Imbalance):**
  * Negative Class (`0`): **91.18%** (87,664 patients)
  * Positive Class (`1`): **8.82%** (8,482 patients)

### Features & Data Types

| Feature | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `gender` | Categorical | Female, Male, Other | Biological sex of the patient |
| `age` | Numerical (float) | 0.08 – 80.0 years | Patient age |
| `hypertension` | Binary (int) | 0, 1 | History of chronic high blood pressure |
| `heart_disease` | Binary (int) | 0, 1 | History of diagnosed cardiovascular disease |
| `smoking_history` | Categorical | never, current, former, etc. | Patient smoking record |
| `bmi` | Numerical (float) | 10.01 – 95.69 $kg/m^2$ | Body Mass Index |
| `HbA1c_level` | Numerical (float) | 3.5 – 9.0% | Glycated hemoglobin (2–3 month blood sugar index) |
| `blood_glucose_level` | Numerical (int) | 80 – 300 $mg/dL$ | Fasting/random blood sugar level |
| `diabetes` | Binary (int) | 0, 1 | Ground-truth diagnosis (**Target**) |

## 🔬 Methodology & Architecture

1. **Data Hygiene & Preprocessing:**
   * Audited the dataset for null values ($0$ detected across all columns).
   * Identified and dropped 3,854 duplicate entries to eliminate artificial redundancy and evaluation bias.

2. **Exploratory Data Analysis (EDA):**
   * Computed summary statistics (mean, median, standard deviation), covariance matrices, and Pearson correlation coefficients.
   * Visualized inter-feature correlations via Seaborn heatmaps and assessed glycemic distribution variations using box plots.

3. **Feature Selection & Train/Test Split:**
   * Isolated the 6 numerical and binary clinical markers: `['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']`.
   * Formatted an **80/20 train-test partition** using `train_test_split` with a fixed seed (`random_state=42`).
   * Enforced **stratified sampling** (`stratify=y`) to maintain the identical 8.82% positive class ratio across training and testing partitions.
   * Split Sizes:
     * **Training Set:** 76,916 records
     * **Testing Set:** 19,230 records

4. **Feature Standardization:**
   * Scaled input vectors using `StandardScaler` to zero mean and unit variance ($\mu = 0, \sigma = 1$).
   * Fitted strictly on the training partition (`fit_transform`) and projected onto the testing partition (`transform`) to prevent data leakage.

5. **Model Training:**
   * Fitted a binary `LogisticRegression` classifier configured with `max_iter=1000`.

## 📈 Experimental Results

### 1. Learned Model Parameters

Because all numerical predictors were standardized prior to optimization, the absolute magnitude of each coefficient directly reflects its relative contribution to the log-odds of a positive diagnosis:

$$
\log\left(\frac{p}{1 - p}\right) = \beta_0 + \sum_{i=1}^{n} \beta_i X_i
$$

* **Model Intercept ($\beta_0$):** `-5.195`

| Rank | Clinical Feature | Coefficient ($\beta_i$) | Correlation with Diabetes ($r$) |
| :---: | :--- | :---: | :---: |
| 1 | **`HbA1c_level`** | **+2.4887** | 0.4064 |
| 2 | **`blood_glucose_level`** | **+1.3645** | 0.4243 |
| 3 | **`age`** | **+1.0650** | 0.2649 |
| 4 | **`bmi`** | **+0.5903** | 0.2149 |
| 5 | **`hypertension`** | **+0.2147** | 0.1957 |
| 6 | **`heart_disease`** | **+0.1583** | 0.1707 |

### 2. Comprehensive Model Metrics

| Metric | Training Set (76,916 samples) | Testing Set (19,230 samples) |
| :--- | :---: | :---: |
| **Accuracy** | 95.86% | **95.91%** |
| **Classification Error Rate** | 4.14% | **4.09%** |
| **Precision** | 0.8654 | **0.8669** |
| **Recall (Sensitivity)** | 0.6284 | **0.6338** |
| **F1-Score** | 0.7281 | **0.7323** |

### 3. Test Set Confusion Matrix Breakdown

$$
\begin{array}{c|cc}
& \textbf{Predicted Negative (0)} & \textbf{Predicted Positive (1)} \\
\hline
\textbf{Actual Negative (0)} & \text{True Negative (TN): } 17,369 & \text{False Positive (FP): } 165 \\
\textbf{Actual Positive (1)} & \text{False Negative (FN): } 621 & \text{True Positive (TP): } 1,075 \\
\end{array}
$$

* **Total Correct Predictions:** $17,369 + 1,075 = 18,444$ ($95.91\%$)
* **Total Misclassified Samples:** $621 + 165 = 786$ ($4.09\%$)
* **False Negative Dominance:** False negatives account for **79.01%** of all test errors ($621 / 786$).

## 🔍 Analytical & Clinical Discussion

1. **Dominant Diagnostic Drivers:**
   * `HbA1c_level` ($\beta = 2.4887$) and `blood_glucose_level` ($\beta = 1.3645$) emerged as the strongest predictors by a significant margin. This aligns with medical consensus, where glycated hemoglobin and fasting plasma glucose serve as primary diagnostic criteria for diabetes mellitus.

2. **The Accuracy Paradox in Medical Screening:**
   * A naive baseline model classifying every patient as non-diabetic would achieve **91.18% accuracy** simply due to class imbalance.
   * While the model achieves a high overall accuracy (**95.91%**) and strong precision (**86.69%**), the recall of **63.38%** means it fails to identify **621 out of 1,696 diabetic individuals** at the standard $0.5$ classification threshold.
   * In clinical triage, false negatives present higher medical risk than false positives; missed diagnoses delay therapeutic intervention.

## 🚀 Future Enhancements

* **Threshold Tuning & Balanced Weights:** Optimize the classification decision threshold (e.g., lower from $0.5$ to $0.3$) or train with `class_weight='balanced'` to shift the decision boundary toward higher sensitivity/recall.
* **Categorical Feature Encoding:** One-hot encode `smoking_history` and `gender` to assess whether behavioral and demographic signals improve classification margins.
* **Non-Linear Classifiers:** Benchmark against ensemble decision trees (Random Forest, XGBoost) and Support Vector Machines to evaluate complex interactions between BMI and age.

## 📂 Repository File Tree

```text
├── diabetes_prediction_dataset.csv   # Raw clinical dataset (Kaggle)
├── diabetes_prediction.ipynb         # Complete, executable Jupyter Notebook
├── requirements.txt                  # Python dependencies and package versions
└── README.md                         # Project documentation and analysis
```

## 🛠️ Getting Started & Replication

### Prerequisites

* Python 3.9+
* `pip` package manager

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/AI101-Mini-Project.git
   cd AI101-Mini-Project
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # On macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the analysis notebook:**
   ```bash
   jupyter notebook diabetes_prediction.ipynb
   ```