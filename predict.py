import os
import time
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering_v2 import extract_advanced_features


def generate_predictions(data_dir='dataset(31)',
                         output_csv='sample_submission.csv',
                         models_dir='model'):
    print("=======================================================")
    print("      UNIFIED XGBOOST INFERENCE PIPELINE               ")
    print("=======================================================")

    # 1. Load model & metadata
    models = joblib.load(os.path.join(models_dir, 'xgboost_pipeline.joblib'))
    meta = joblib.load(os.path.join(models_dir, 'metadata.joblib'))

    feature_cols = meta['feature_cols']
    best_thresh = meta['best_thresh']

    # 2. Extract features (observation window only)
    print(">>> Extracting features from observation window...")
    df = extract_advanced_features(data_dir=data_dir, is_train=False)
    X = df[feature_cols].copy()
    athlete_ids = df['athlete_id'].values

    n_folds = len(models['classifiers'])

    # 3. Task A: Injury Classification (average across 5-fold models)
    print(">>> Task A: Injury Classification...")
    t0 = time.time()
    probs = np.zeros(len(X))
    for clf in models['classifiers']:
        probs += clf.predict_proba(X)[:, 1] / n_folds
    binary_preds = (probs >= best_thresh).astype(int)

    # 4. Task B1: Onset Day Offset
    print(">>> Task B1: Onset Day Offset...")
    onset = np.zeros(len(X))
    for reg in models['onset_regressors']:
        onset += reg.predict(X) / n_folds
    onset_int = np.clip(np.round(onset), 1, 30).astype(int)

    # 5. Task B2: Recovery Duration
    print(">>> Task B2: Recovery Duration...")
    rec = np.zeros(len(X))
    for reg in models['rec_regressors']:
        rec += reg.predict(X) / n_folds
    rec_int = np.clip(np.round(rec), 5, 20).astype(int)

    t_inference = (time.time() - t0) * 1000

    # 6. Assemble submission
    submission = pd.DataFrame({
        'athlete_id': athlete_ids,
        'injured_in_risk_window': binary_preds,
        'onset_day_offset': onset_int,
        'recovery_duration': rec_int
    })

    submission.to_csv(output_csv, index=False)

    print(f"\n>>> Generated {output_csv}")
    print(f"    Shape: {submission.shape}")
    print(f"    Threshold used: {best_thresh}")
    print(f"    Predicted injured: {binary_preds.sum()} / {len(binary_preds)}")
    print(f"    Inference time: {t_inference:.2f} ms ({t_inference/len(X):.4f} ms/athlete)")
    print(f"\nFirst 5 rows:")
    print(submission.head())

    # Verify submission format
    assert len(submission) == 3000, f"Expected 3000 rows, got {len(submission)}"
    assert list(submission.columns) == ['athlete_id', 'injured_in_risk_window', 'onset_day_offset', 'recovery_duration']
    assert submission.isnull().sum().sum() == 0, "Null values found"
    assert submission['injured_in_risk_window'].isin([0, 1]).all()
    assert (submission['onset_day_offset'] >= 1).all() and (submission['onset_day_offset'] <= 30).all()
    assert (submission['recovery_duration'] >= 5).all() and (submission['recovery_duration'] <= 20).all()
    print("\n>>> All submission format constraints VERIFIED.")

    return submission


if __name__ == '__main__':
    generate_predictions()
