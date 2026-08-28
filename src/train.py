import os
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from evaluate import compute_competition_metrics


def run_training(features_path='processed_data/master_features.csv', models_dir='model'):
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(features_path)

    target_cols = ['injured_in_risk_window', 'onset_day_offset', 'recovery_duration']
    id_cols = ['athlete_id', 'sport', 'gender', 'dominant_side', 'position', 'team_id']
    feature_cols = [c for c in df.columns if c not in target_cols and c not in id_cols]

    X = df[feature_cols].copy()
    for c in X.select_dtypes(include='bool').columns:
        X[c] = X[c].astype(int)

    y_cls = df['injured_in_risk_window'].values
    y_onset = df['onset_day_offset'].values
    y_rec = df['recovery_duration'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(df))
    oof_onset = np.full(len(df), np.nan)
    oof_rec = np.full(len(df), np.nan)

    trained_classifiers = []
    trained_onset_regressors = []
    trained_rec_regressors = []

    clf_params = {
        'n_estimators': 100,
        'learning_rate': 0.0415,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.6212,
        'colsample_bytree': 0.6840,
        'colsample_bynode': 0.6505,
        'gamma': 2.1641,
        'reg_alpha': 1.7637,
        'reg_lambda': 0.0162,
        'scale_pos_weight': 1.4577,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }

    reg_params = {
        'n_estimators': 250,
        'learning_rate': 0.03,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'eval_metric': 'mae',
        'n_jobs': -1
    }

    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cls)):
        X_tr, y_tr = X.iloc[train_idx], y_cls[train_idx]
        X_val, y_val = X.iloc[val_idx], y_cls[val_idx]

        clf = xgb.XGBClassifier(**clf_params, random_state=42 + fold)
        clf.fit(X_tr, y_tr)
        trained_classifiers.append(clf)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

        inj_mask = y_tr == 1
        onset_reg = xgb.XGBRegressor(**reg_params, random_state=42 + fold)
        onset_reg.fit(X_tr[inj_mask], y_onset[train_idx][inj_mask])
        trained_onset_regressors.append(onset_reg)
        oof_onset[val_idx] = np.clip(np.round(onset_reg.predict(X_val)), 1, 30)

        rec_reg = xgb.XGBRegressor(**reg_params, random_state=42 + fold)
        rec_reg.fit(X_tr[inj_mask], y_rec[train_idx][inj_mask])
        trained_rec_regressors.append(rec_reg)
        oof_rec[val_idx] = np.clip(np.round(rec_reg.predict(X_val)), 5, 20)

    t_train = time.time() - t_start
    auc = roc_auc_score(y_cls, oof_probs)

    best_composite = -1
    best_thresh = 0.5
    best_metrics = None

    for t in np.arange(0.10, 0.65, 0.01):
        pred_bin = (oof_probs >= t).astype(int)
        m = compute_competition_metrics(
            y_true_cls=y_cls,
            y_pred_cls=pred_bin,
            onset_true=y_onset,
            onset_pred=oof_onset,
            recovery_true=y_rec,
            recovery_pred=oof_rec
        )
        if m['composite_score'] > best_composite:
            best_composite = m['composite_score']
            best_thresh = round(t, 2)
            best_metrics = m

    hits = (y_cls == 1) & (oof_probs >= best_thresh)
    onset_mae_hits = np.mean(np.abs(oof_onset[hits] - y_onset[hits]))
    rec_mae_hits = np.mean(np.abs(oof_rec[hits] - y_rec[hits]))

    print(f"5-Fold CV completed in {t_train:.1f}s")
    print(f"OOF ROC-AUC:      {auc:.4f}")
    print(f"Optimal Thresh:   {best_thresh}")
    print(f"Composite Score:  {best_composite:.4f}")
    print(f"Recall @ {best_thresh}:    {best_metrics['task_a_recall']:.4f}")
    print(f"Precision @ {best_thresh}: {best_metrics['task_a_precision']:.4f}")
    print(f"Onset MAE (hits): {onset_mae_hits:.2f} days")
    print(f"Rec MAE (hits):   {rec_mae_hits:.2f} days")

    joblib.dump({
        'classifiers': trained_classifiers,
        'onset_regressors': trained_onset_regressors,
        'rec_regressors': trained_rec_regressors
    }, os.path.join(models_dir, 'xgboost_pipeline.joblib'))

    joblib.dump({
        'feature_cols': feature_cols,
        'best_thresh': best_thresh,
        'metrics': best_metrics,
        'auc': auc,
        'onset_mae_hits': onset_mae_hits,
        'rec_mae_hits': rec_mae_hits,
        'training_time_seconds': t_train
    }, os.path.join(models_dir, 'metadata.joblib'))

    oof_df = pd.DataFrame({
        'athlete_id': df['athlete_id'],
        'true_injured': y_cls,
        'pred_prob': oof_probs,
        'pred_injured': (oof_probs >= best_thresh).astype(int),
        'true_onset': y_onset,
        'pred_onset': oof_onset,
        'true_recovery': y_rec,
        'pred_recovery': oof_rec
    })
    oof_df.to_csv('processed_data/oof_predictions.csv', index=False)

    return best_metrics, auc


if __name__ == '__main__':
    run_training()
