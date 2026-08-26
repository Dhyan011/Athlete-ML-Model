import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from evaluate import compute_competition_metrics

def run_advanced_training(features_path='processed_data/master_features_v2.csv', models_dir='models_v2'):
    os.makedirs(models_dir, exist_ok=True)
    
    print(">>> Loading processed v2 feature dataset...")
    df = pd.read_csv(features_path)
    
    target_cols = ['injured_in_risk_window', 'onset_day_offset', 'recovery_duration']
    id_cols = ['athlete_id', 'sport', 'gender', 'dominant_side', 'position', 'team_id']
    
    feature_cols = [c for c in df.columns if c not in target_cols and c not in id_cols]
    print(f"Total Feature Count: {len(feature_cols)}")
    
    X = df[feature_cols].copy()
    y_cls = df['injured_in_risk_window'].values
    y_onset = df['onset_day_offset'].values
    y_rec = df['recovery_duration'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_cls_probs = np.zeros(len(df))
    oof_onset_preds = np.full(len(df), np.nan)
    oof_rec_preds = np.full(len(df), np.nan)
    
    trained_models = {
        'task_a_lgb': [],
        'task_a_xgb': [],
        'task_a_cat': [],
        'task_a_et': [],
        'task_b1_onset': [],
        'task_b2_rec': []
    }
    
    print("\n=======================================================")
    print("      STARTING ADVANCED 5-FOLD ENSEMBLE TRAINING       ")
    print("=======================================================")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cls)):
        print(f"\n--- FOLD {fold + 1} / 5 ---")
        X_train, y_train_cls = X.iloc[train_idx], y_cls[train_idx]
        X_val, y_val_cls = X.iloc[val_idx], y_cls[val_idx]
        
        # 1. LightGBM Classifier (with tuned leaf depth & scale_pos_weight)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.8,
            scale_pos_weight=1.2,
            random_state=42 + fold,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train_cls)
        
        # 2. XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.8,
            scale_pos_weight=1.2,
            random_state=42 + fold,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train_cls)
        
        # 3. CatBoost Classifier
        cat_model = CatBoostClassifier(
            iterations=400,
            learning_rate=0.03,
            depth=5,
            random_seed=42 + fold,
            verbose=0
        )
        cat_model.fit(X_train, y_train_cls)
        
        # 4. ExtraTrees Classifier
        et_model = ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=4,
            random_state=42 + fold,
            n_jobs=-1
        )
        et_model.fit(X_train, y_train_cls)
        
        trained_models['task_a_lgb'].append(lgb_model)
        trained_models['task_a_xgb'].append(xgb_model)
        trained_models['task_a_cat'].append(cat_model)
        trained_models['task_a_et'].append(et_model)
        
        # Out-of-Fold Weighted Probability Blend
        p_lgb = lgb_model.predict_proba(X_val)[:, 1]
        p_xgb = xgb_model.predict_proba(X_val)[:, 1]
        p_cat = cat_model.predict_proba(X_val)[:, 1]
        p_et = et_model.predict_proba(X_val)[:, 1]
        
        fold_probs = 0.35 * p_lgb + 0.35 * p_xgb + 0.20 * p_cat + 0.10 * p_et
        oof_cls_probs[val_idx] = fold_probs
        
        # --- TASK B: TIMING REGRESSION CASCADE ---
        train_inj_mask = (y_train_cls == 1)
        X_train_inj = X_train[train_inj_mask]
        y_train_onset_inj = y_onset[train_idx][train_inj_mask]
        y_train_rec_inj = y_rec[train_idx][train_inj_mask]
        
        # Task B1: Onset Day Regressor (CatBoost + LightGBM MAE objective)
        onset_cat = CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            loss_function='MAE',
            random_seed=42 + fold,
            verbose=0
        )
        onset_cat.fit(X_train_inj, y_train_onset_inj)
        trained_models['task_b1_onset'].append(onset_cat)
        
        val_onset_pred = np.clip(np.round(onset_cat.predict(X_val)), 1, 30)
        oof_onset_preds[val_idx] = val_onset_pred
        
        # Task B2: Recovery Duration Regressor
        rec_cat = CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            loss_function='MAE',
            random_seed=42 + fold,
            verbose=0
        )
        rec_cat.fit(X_train_inj, y_train_rec_inj)
        trained_models['task_b2_rec'].append(rec_cat)
        
        val_rec_pred = np.clip(np.round(rec_cat.predict(X_val)), 5, 20)
        oof_rec_preds[val_idx] = val_rec_pred
        
        print(f"Fold {fold + 1} validation completed.")
        
    # --- THRESHOLD OPTIMIZATION ---
    print("\n=======================================================")
    print("      OPTIMIZING CLASSIFICATION DECISION THRESHOLDS    ")
    print("=======================================================")
    
    best_thresh = 0.5
    best_f1 = 0.0
    best_metrics = None
    
    thresholds = np.linspace(0.20, 0.70, 101)
    for t in thresholds:
        y_pred_bin = (oof_cls_probs >= t).astype(int)
        metrics = compute_competition_metrics(
            y_true_cls=y_cls,
            y_pred_cls=y_pred_bin,
            onset_true=y_onset,
            onset_pred=oof_onset_preds,
            recovery_true=y_rec,
            recovery_pred=oof_rec_preds
        )
        if metrics['task_a_f1'] > best_f1:
            best_f1 = metrics['task_a_f1']
            best_thresh = t
            best_metrics = metrics
            
    print(f"Optimal Threshold: {best_thresh:.3f}")
    print(f"Task A F1-Score: {best_metrics['task_a_f1']:.4f} (Precision: {best_metrics['task_a_precision']:.4f}, Recall: {best_metrics['task_a_recall']:.4f})")
    print(f"Task B1 Onset MAE (Hits): {np.mean(np.abs(oof_onset_preds[(y_cls==1) & (oof_cls_probs>=best_thresh)] - y_onset[(y_cls==1) & (oof_cls_probs>=best_thresh)])):.2f} Days")
    print(f"Task B2 Recovery MAE (Hits): {np.mean(np.abs(oof_rec_preds[(y_cls==1) & (oof_cls_probs>=best_thresh)] - y_rec[(y_cls==1) & (oof_cls_probs>=best_thresh)])):.2f} Days")
    print(f"Composite Score: {best_metrics['composite_score']:.4f}")
    
    # Save Model Artifacts
    print("\n>>> Saving Trained Model Ensemble to models_v2/ ...")
    joblib.dump(trained_models, os.path.join(models_dir, 'trained_ensemble.joblib'))
    joblib.dump({
        'feature_cols': feature_cols,
        'best_thresh': best_thresh,
        'metrics': best_metrics
    }, os.path.join(models_dir, 'pipeline_metadata.joblib'))
    
    oof_df = pd.DataFrame({
        'athlete_id': df['athlete_id'],
        'true_injured': y_cls,
        'pred_prob': oof_cls_probs,
        'pred_injured': (oof_cls_probs >= best_thresh).astype(int),
        'true_onset': y_onset,
        'pred_onset': oof_onset_preds,
        'true_recovery': y_rec,
        'pred_recovery': oof_rec_preds
    })
    oof_df.to_csv('processed_data/oof_predictions_v2.csv', index=False)
    print("Saved processed_data/oof_predictions_v2.csv successfully!")
    
    return best_metrics

if __name__ == '__main__':
    run_advanced_training()
