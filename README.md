# PLAYHACK ML Track — Athlete Injury & Recovery Duration Prediction
**IIT Guwahati Sports Board X Technical Board Hackathon**

## 📌 Project Overview
This repository contains the end-to-end machine learning pipeline for predicting:
1. **Task A**: Binary Injury Risk Classification (`injured_in_risk_window`: 0 or 1) in Days 31 to 60.
2. **Task B1**: Injury Onset Day Offset (`onset_day_offset`: 1 to 30).
3. **Task B2**: Recovery Duration (`recovery_duration`: 5 to 20 days).

The pipeline strictly adheres to zero-leakage constraints by using only the **30-Day Observation Window (Days 1 to 30)** to extract biometrics, sleep patterns, training session exertion, and workload ratios.

---

## 🏗️ Architecture & Key Components
- **Feature Engineering Engine (`src/feature_engineering.py`)**:
  - Acute-to-Chronic Workload Ratio (ACWR 7-day vs 30-day).
  - Peak single-day step spike delta (+44.8% empirical injury signature).
  - Nocturnal resting heart rate (2 AM to 6 AM) & cardiac strain reserve.
  - Sleep efficiency, sleep regularity index, and cumulative sleep debt.
  - Sport-specific contact risk & bimodal recovery profiling.
- **Model Ensemble (`src/train.py`)**:
  - Triple-Gradient Boosted Ensemble (**LightGBM + XGBoost + HistGradientBoosting**) trained across 5-Fold Stratified Cross-Validation.
  - Threshold optimization for F1 maximization and false negative penalty mitigation.
  - Conditional regression cascade for exact onset day and rehabilitation duration.
- **Inference & Submission Generator (`predict.py`)**:
  - Extracts observation features and produces `sample_submission.csv`.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Feature Extraction
```bash
python3 src/feature_engineering.py
```

### 3. Model Training & Cross-Validation
```bash
python3 src/train.py
```

### 4. End-to-End Inference
```bash
python3 predict.py
```

---

## 📊 Key Cross-Validation Results
- **Task A F1-Score**: `0.6570` (Precision: `90.62%`, AUC: `0.941`)
- **Task B1 Onset MAE (Hits)**: `0.82 Days` (vs Baseline: `7.61 Days`)
- **Task B2 Recovery MAE (Hits)**: `2.90 Days` (vs Baseline: `3.24 Days`)
