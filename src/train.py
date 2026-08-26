import os
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from evaluate import compute_competition_metrics

def run_training(features_path='processed_data/master_features_v2.csv', models_dir='model'):
    os.makedirs(models_dir, exist_ok=True)

    print("=======================================================")
    print("   UNIFIED SINGLE-MODEL PIPELINE: STANDALONE XGBOOST   ")
    print("=======================================================")

    df = pd.read_csv(features_path)

    target_cols = ['injured_in_risk_window', 'onset_day_offset', 'recovery_duration']
    id_cols = ['athlete_id', 'sport', 'gender', 'dominant_side', 'position', 'team_id']
    feature_cols = [c for c in df.columns if c not in target_cols and c not in id_cols]
    print(f"Feature count: {len(feature_cols)}")

    X = df[feature_cols].copy()
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

    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cls)):
        print(f"\n--- Fold {fold + 1} / 5 ---")
        X_tr, y_tr = X.iloc[train_idx], y_cls[train_idx]
        X_val, y_val = X.iloc[val_idx], y_cls[val_idx]

        # Task A: XGBoost Classifier
        clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            scale_pos_weight=1.3,
            random_state=42 + fold,
            eval_metric='logloss'
        )
        clf.fit(X_tr, y_tr)
        trained_classifiers.append(clf)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

        # Task B1: Onset Day Regressor (trained on injured subset only)
        inj_mask = y_tr == 1
        onset_reg = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            random_state=42 + fold,
            eval_metric='mae'
        )
        onset_reg.fit(X_tr[inj_mask], y_onset[train_idx][inj_mask])
        trained_onset_regressors.append(onset_reg)
        oof_onset[val_idx] = np.clip(np.round(onset_reg.predict(X_val)), 1, 30)

        # Task B2: Recovery Duration Regressor (trained on injured subset only)
        rec_reg = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            random_state=42 + fold,
            eval_metric='mae'
        )
        rec_reg.fit(X_tr[inj_mask], y_rec[train_idx][inj_mask])
        trained_rec_regressors.append(rec_reg)
        oof_rec[val_idx] = np.clip(np.round(rec_reg.predict(X_val)), 5, 20)

    t_train = time.time() - t_start
    print(f"\nTraining time (5-fold): {t_train:.2f} seconds")

    # =========================================================
    # JOINT THRESHOLD OPTIMIZATION (Fix 3)
    # =========================================================
    print("\n=======================================================")
    print("  JOINT THRESHOLD OPTIMIZATION: COMPETITION SCORING    ")
    print("=======================================================")
    print(f"{'Thresh':>7s} | {'F1':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'Onset MAE':>10s} | {'Rec MAE':>10s} | {'Onset Skill':>11s} | {'Rec Skill':>11s} | {'Composite':>10s}")
    print("-" * 110)

    best_composite = -1
    best_thresh = 0.5
    best_metrics = None
    results_table = []

    for t in np.arange(0.15, 0.56, 0.01):
        pred_bin = (oof_probs >= t).astype(int)
        m = compute_competition_metrics(
            y_true_cls=y_cls,
            y_pred_cls=pred_bin,
            onset_true=y_onset,
            onset_pred=oof_onset,
            recovery_true=y_rec,
            recovery_pred=oof_rec
        )
        row = {
            'threshold': round(t, 2),
            'f1': m['task_a_f1'],
            'precision': m['task_a_precision'],
            'recall': m['task_a_recall'],
            'onset_mae': m['mae_onset_model'],
            'rec_mae': m['mae_rec_model'],
            'onset_skill': m['task_b_onset_skill'],
            'rec_skill': m['task_b_recovery_skill'],
            'composite': m['composite_score']
        }
        results_table.append(row)
        print(f"{t:7.2f} | {m['task_a_f1']:6.4f} | {m['task_a_precision']:6.4f} | {m['task_a_recall']:6.4f} | {m['mae_onset_model']:10.2f} | {m['mae_rec_model']:10.2f} | {m['task_b_onset_skill']:11.4f} | {m['task_b_recovery_skill']:11.4f} | {m['composite_score']:10.4f}")

        if m['composite_score'] > best_composite:
            best_composite = m['composite_score']
            best_thresh = round(t, 2)
            best_metrics = m

    # Also find F1-only best for comparison
    f1_best_t = max(results_table, key=lambda r: r['f1'])['threshold']

    print(f"\n>>> Competition-Score-Optimal Threshold: {best_thresh}")
    print(f">>> F1-Only-Optimal Threshold:           {f1_best_t}")
    print(f">>> Difference:                          {abs(best_thresh - f1_best_t):.2f}")
    print(f"\nAt competition-optimal threshold {best_thresh}:")
    print(f"  F1:              {best_metrics['task_a_f1']:.4f}")
    print(f"  Precision:       {best_metrics['task_a_precision']:.4f}")
    print(f"  Recall:          {best_metrics['task_a_recall']:.4f}")
    print(f"  Onset MAE:       {best_metrics['mae_onset_model']:.2f} (Baseline: {best_metrics['mae_onset_base']:.2f})")
    print(f"  Recovery MAE:    {best_metrics['mae_rec_model']:.2f} (Baseline: {best_metrics['mae_rec_base']:.2f})")
    print(f"  Onset Skill:     {best_metrics['task_b_onset_skill']:.4f}")
    print(f"  Recovery Skill:  {best_metrics['task_b_recovery_skill']:.4f}")
    print(f"  Composite Score: {best_metrics['composite_score']:.4f}")

    # Compute verified AUC
    auc = roc_auc_score(y_cls, oof_probs)
    print(f"\n  ROC-AUC (OOF):   {auc:.4f}")

    # Hits-only MAE
    hits = (y_cls == 1) & (oof_probs >= best_thresh)
    onset_mae_hits = np.mean(np.abs(oof_onset[hits] - y_onset[hits]))
    rec_mae_hits = np.mean(np.abs(oof_rec[hits] - y_rec[hits]))
    print(f"  Onset MAE (Hits only):    {onset_mae_hits:.4f}")
    print(f"  Recovery MAE (Hits only): {rec_mae_hits:.4f}")

    # Save model artifacts
    print("\n>>> Saving model artifacts...")
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

    # Save OOF predictions
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
    print("Saved processed_data/oof_predictions.csv")

    return best_metrics, auc


if __name__ == '__main__':
    run_training()
