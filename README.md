# 🏃 PLAYHACK ML Track — Athlete Injury Risk & Recovery Duration Forecasting
**IIT Guwahati Sports Board × Technical Board Hackathon**

---

## 📖 Summary
Predicts **athlete injury risk**, **injury onset timing**, and **rehabilitation duration** for 3,000 athletes across 6 sports, using 30 days of wearable biometric data to forecast outcomes in the following 30-day risk window.

**Model**: Standalone XGBoost (5-fold cross-validated).  
**Threshold**: Optimized for the competition's composite scoring metric (not F1 alone), deliberately favouring high recall to avoid the severe 30-day False Negative penalty.

---

## 🎯 1. Problem & Targets

```
        DAY 1–30 (Observation)              DAY 31–60 (Risk Window)
   ┌───────────────────────────┐       ┌───────────────────────────┐
   │  Wearable telemetry used  │  ──►  │   Predict these targets   │
   │  for feature engineering  │       │   (no data available)     │
   └───────────────────────────┘       └───────────────────────────┘
```

| Target | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `injured_in_risk_window` | Binary | 0 or 1 | Will the athlete get injured in Days 31–60? (Base rate: 35%) |
| `onset_day_offset` | Integer | 1–30 | Which day in the risk window does the injury begin? |
| `recovery_duration` | Integer | 5–20 | How many days until the athlete returns to play? |

**Scoring rule**: A missed injury (False Negative) triggers a fixed **30-day penalty** on both timing predictions, making recall critical.

---

## 🔬 2. Key Data Findings (Verified)

### Workload Spike Signature
Injured athletes showed a **+44.84%** higher peak single-day step count during the observation window (14,982 vs 10,344 steps; two-sample t-test: $t = 26.25$, $p = 7.31 \times 10^{-137}$).

### Recovery Duration Bimodality by Sport
Computed from `Athlete Metadata.csv` joined with `Train Labels Dataset.csv` on the injured subset:

| Sport Group | Sports | Mean Recovery (Days) | Range |
| :--- | :--- | :--- | :--- |
| Contact / Field | Football, Basketball | 14.1–14.5 | 8–20 |
| Court / Non-Contact | Athletics, Badminton, Tennis, Volleyball | 9.9–10.3 | 5–15 |

---

## ⚙️ 3. Feature Engineering (121 Features)

All features extracted strictly from the **30-day observation window** (Days 1–30). Zero data leakage verified.

| Module | Key Features |
| :--- | :--- |
| **Workload & ACWR** | Acute (7d) vs Chronic (30d) step/calorie/intensity ratios, peak spike delta, Week 4 vs Week 1 ramp rate |
| **Training Load** | Foster's Monotony & Strain, TRIMP proxy (weighted active minutes), session frequency/duration/type ratios |
| **Sleep** | Sleep efficiency, regularity (CV), cumulative deficit below 480 min, severe deprivation day count, Week 4 sleep drop |
| **Heart Rate** | Nocturnal resting HR (2–6 AM), peak exertion HR, heart rate reserve, cardiac strain hours (>140 bpm) |
| **Baselines** | BMI, age, experience ratio, sport type encoding, contact sport flag, prior injury count |

---

## 🧠 4. Model Architecture

**Standalone XGBoost** trained via 5-fold Stratified Cross-Validation.

| Component | Configuration |
| :--- | :--- |
| **Task A Classifier** | `XGBClassifier(n_estimators=350, lr=0.025, max_depth=5, subsample=0.85, scale_pos_weight=1.2)` |
| **Task B1 Onset Regressor** | `XGBRegressor(n_estimators=200, lr=0.03, max_depth=4)` — trained on injured subset only |
| **Task B2 Recovery Regressor** | `XGBRegressor(n_estimators=200, lr=0.03, max_depth=4)` — trained on injured subset only |

**Why a single model, not an ensemble?** Ablation showed a 4-model blend (LightGBM+XGBoost+CatBoost+ExtraTrees) provided only +0.0008 F1 improvement over standalone XGBoost, with no AUC gain. A single model is simpler to explain, faster to train, and easier to reproduce.

---

## 📊 5. Verified Performance (5-Fold OOF)

### Classification Metrics (Task A)

| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | **0.7618** |
| **F1-Score** | **0.5450** (at competition-optimal threshold) |
| **Precision** | **39.44%** |
| **Recall** | **88.19%** |
| **Decision Threshold** | **0.15** |

### Timing Metrics (Task B)

| Metric | Value | Baseline (Mean Prediction) |
| :--- | :--- | :--- |
| **Onset MAE (penalized, all injured)** | **5.62 days** | 7.61 days |
| **Onset Skill Score** | **0.2625** | 0.0 |
| **Recovery MAE (penalized, all injured)** | **6.14 days** | 3.24 days |
| **Recovery Skill Score** | **0.0000** | 0.0 |
| **Composite Competition Score** | **0.2692** | — |

### Why Threshold = 0.15 Instead of 0.48

The competition penalizes missed injuries with a fixed 30-day error on both timing predictions. This asymmetric penalty makes recall far more valuable than precision:

- At **threshold 0.48** (F1-optimal): F1 = 0.652, Recall = 51.7%, but **509 injured athletes are missed**, each incurring 30-day penalties → Composite Score = **0.217**
- At **threshold 0.15** (competition-optimal): F1 = 0.545, Recall = 88.2%, only **124 injured athletes missed** → Composite Score = **0.269** (+24% improvement)

The lower threshold accepts more false alarms (lower precision) but dramatically reduces the penalty burden from false negatives, which dominates the competition score.

### Runtime Benchmarks

| Metric | Value |
| :--- | :--- |
| Training time (5-fold) | 12.18 seconds |
| Inference latency (3,000 athletes) | 121.38 ms |
| Per-athlete latency | 0.04 ms |
| Model file size | 6.55 MB |

---

## 📂 6. Repository Structure

```
Athlete-ML-Model/
├── src/
│   ├── feature_engineering_v2.py   # Feature extraction pipeline (121 features)
│   ├── train.py                    # 5-fold XGBoost training + threshold optimization
│   ├── evaluate.py                 # Competition metric computation
│   ├── generate_visualizations.py  # EDA chart generation
│   └── create_presentation.py      # PPT slide generation
├── model/                          # Trained XGBoost model weights
│   ├── xgboost_pipeline.joblib
│   └── metadata.joblib
├── figures/                        # EDA and performance visualizations
├── predict.py                      # End-to-end inference → sample_submission.csv
├── sample_submission.csv           # Final 3000-row predictions
├── requirements.txt
└── README.md
```

---

## ⚡ 7. Quickstart

```bash
# 1. Setup
git clone https://github.com/Dhyan011/Athlete-ML-Model.git
cd Athlete-ML-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Feature Extraction (Days 1–30 only)
python3 src/feature_engineering_v2.py

# 3. Train + Threshold Optimization
python3 src/train.py

# 4. Generate Predictions
python3 predict.py
```

---

## 🏆 Authors
Submission for **PLAYHACK ML Track** — IIT Guwahati Sports Board × Technical Board
