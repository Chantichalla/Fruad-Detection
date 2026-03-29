# Credit Card Fraud Detection System

> End-to-end ML pipeline for real-time fraud detection with explainable AI — built with XGBoost, SHAP, FastAPI, and Streamlit.

---

## 🧠 Problem Statement

Credit card fraud is a classic **severely imbalanced classification problem** — only **0.17% of 284,807 transactions are fraudulent**. A naive model predicting everything as legitimate achieves 99.8% accuracy but catches **zero fraud**. This project correctly addresses that with proper metric selection, imbalance handling, and explainability.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      PIPELINE                           │
└─────────────────────────────────────────────────────────┘

[1] DATA
    └── Kaggle Credit Card Fraud Dataset
        ├── 284,807 transactions
        ├── 492 fraud cases (0.17%)
        └── 30 PCA-transformed features (V1–V28, Time, Amount)

[2] PREPROCESSING
    ├── StandardScaler  → normalize Time & Amount
    ├── Train/Test split → 80/20, stratified
    └── SMOTE           → applied on TRAINING data only (no data leakage)

[3] MODEL COMPARISON (5-fold stratified CV)
    ├── Logistic Regression  → baseline
    ├── Random Forest        → ensemble
    └── XGBoost              → selected (best AUC-ROC + F1 on fraud class)

[4] EXPLAINABILITY
    └── SHAP
        ├── Summary Plot     → global feature importance
        ├── Waterfall Plot   → per-prediction breakdown
        └── Force Plot       → visual explanation per transaction

[5] SERVING
    ├── FastAPI    → POST /predict endpoint
    └── Streamlit  → interactive dashboard with SHAP inline
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC (XGBoost) | ~0.98 |
| F1-Score (fraud class) | ~0.84 |
| Precision (fraud) | ~0.88 |
| Recall (fraud) | ~0.81 |
| Baseline accuracy (all-legit model) | 99.83% — catches zero fraud |

> **Why F1, not accuracy?** At 0.17% fraud rate, accuracy is meaningless. A model predicting everything as legitimate gets 99.83% accuracy. F1-score on the fraud class correctly captures both precision and recall on the minority class.

---

## 🔑 Key Design Decisions

| Decision | Why |
|----------|-----|
| **XGBoost over Random Forest** | Built-in L1/L2 regularization, gradient boosting corrects prior errors sequentially |
| **SMOTE on training data only** | Applying SMOTE before train/test split causes data leakage — inflates metrics artificially |
| **Stratified k-fold CV** | Without stratification, some folds may have zero fraud samples |
| **F1 over accuracy** | Fraud class imbalance makes accuracy misleading — see results above |
| **SHAP for explainability** | Provides legally auditable per-transaction explanations; top drivers: V14, V4, Amount |

---

## 📁 Project Structure

```
fraud-detection/
│
├── data/
│   └── creditcard.csv              ← Kaggle dataset (not committed)
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   ← EDA, model comparison, SHAP analysis
│
├── src/
│   ├── preprocess.py               ← Scaling + SMOTE pipeline
│   ├── train.py                    ← Model training + evaluation
│   └── predict.py                  ← Inference logic
│
├── api/
│   └── main.py                     ← FastAPI app (/predict, /health)
│
├── app/
│   └── streamlit_app.py            ← Streamlit dashboard
│
├── models/
│   └── xgboost_model.pkl           ← Serialized trained model (joblib)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/HarshithReddyML/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

### 4. Train the model
```bash
python src/train.py
```

### 5. Run the FastAPI server
```bash
uvicorn api.main:app --reload
```

### 6. Run the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10 |
| ML | Scikit-learn, XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| API | FastAPI |
| Dashboard | Streamlit |
| Serialization | joblib |

---

## 📋 Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
fastapi
uvicorn
streamlit
joblib
matplotlib
seaborn
```

---

## 📌 Interview Q&A (Know These Cold)

**Q: Why did you choose XGBoost?**
> Gradient boosting sequentially corrects errors from prior trees. XGBoost adds L1/L2 regularization which prevents overfitting on the majority class and handles missing values natively.

**Q: What is SMOTE and why did you apply it only on training data?**
> SMOTE generates synthetic minority class samples by interpolating between existing fraud samples. Applied on test data, it leaks information that wouldn't exist in real inference — inflating all metrics artificially.

**Q: What does SHAP tell you?**
> SHAP assigns each feature a contribution value based on Shapley values from cooperative game theory. V14 and V4 showed the highest negative SHAP values for fraud — meaning large deviations in those PCA components strongly predicted fraud.

---

*Built by Challa Harshith Reddy | github.com/HarshithReddyML*
