import os
import time
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import extract_features


def generate_predictions(data_dir='dataset(31)',
                         output_csv='sample_submission.csv',
                         models_dir='model'):
    models = joblib.load(os.path.join(models_dir, 'xgboost_pipeline.joblib'))
    meta = joblib.load(os.path.join(models_dir, 'metadata.joblib'))

    feature_cols = meta['feature_cols']
    best_thresh = meta['best_thresh']

    df = extract_features(data_dir=data_dir, is_train=False)
    for c in df.select_dtypes(include='bool').columns:
        df[c] = df[c].astype(int)

    X = df[feature_cols].copy()
    athlete_ids = df['athlete_id'].values

    n_folds = len(models['classifiers'])

    t0 = time.time()
    probs = np.zeros(len(X))
    for clf in models['classifiers']:
        probs += clf.predict_proba(X)[:, 1] / n_folds

    binary_preds = (probs >= best_thresh).astype(int)

    onset = np.zeros(len(X))
    for reg in models['onset_regressors']:
        onset += reg.predict(X) / n_folds
    onset_int = np.clip(np.round(onset), 1, 30).astype(int)

    rec = np.zeros(len(X))
    for reg in models['rec_regressors']:
        rec += reg.predict(X) / n_folds
    rec_int = np.clip(np.round(rec), 5, 20).astype(int)

    t_inference = (time.time() - t0) * 1000

    submission = pd.DataFrame({
        'athlete_id': athlete_ids,
        'injured_in_risk_window': binary_preds,
        'onset_day_offset': onset_int,
        'recovery_duration': rec_int
    })

    submission.to_csv(output_csv, index=False)

    print(f"Generated {output_csv} (shape: {submission.shape})")
    print(f"Threshold: {best_thresh} | Predicted injured: {binary_preds.sum()}/{len(binary_preds)}")
    print(f"Inference time: {t_inference:.2f} ms ({t_inference/len(X):.4f} ms/row)")

    assert len(submission) == len(athlete_ids), f"Expected {len(athlete_ids)} rows, got {len(submission)}"
    assert list(submission.columns) == ['athlete_id', 'injured_in_risk_window', 'onset_day_offset', 'recovery_duration']
    assert submission.isnull().sum().sum() == 0, "Null values found"
    assert submission['injured_in_risk_window'].isin([0, 1]).all()
    assert (submission['onset_day_offset'] >= 1).all() and (submission['onset_day_offset'] <= 30).all()
    assert (submission['recovery_duration'] >= 5).all() and (submission['recovery_duration'] <= 20).all()

    return submission


if __name__ == '__main__':
    generate_predictions()
