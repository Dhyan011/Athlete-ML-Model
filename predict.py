import os
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import extract_features

def generate_predictions(data_dir='dataset(31)', 
                         output_csv='sample_submission.csv',
                         models_dir='models'):
    print("=======================================================")
    print("      PLAYHACK ML INFERENCE & SUBMISSION PIPELINE      ")
    print("=======================================================")
    
    # 1. Load trained models & metadata
    models_dict = joblib.load(os.path.join(models_dir, 'trained_ensemble.joblib'))
    meta_dict = joblib.load(os.path.join(models_dir, 'pipeline_metadata.joblib'))
    
    feature_cols = meta_dict['feature_cols']
    best_thresh = meta_dict['best_thresh']
    
    # 2. Extract features from Observation Window (Days 1 to 30)
    print(">>> Extracting features from observation window...")
    df_features = extract_features(data_dir=data_dir, is_train=False)
    
    X = df_features[feature_cols].copy()
    athlete_ids = df_features['athlete_id'].values
    
    # 3. Model Inference - Task A (Injury Classification)
    print(">>> Generating Task A Predictions (Injury Classification)...")
    prob_preds = np.zeros(len(X))
    
    n_folds = len(models_dict['task_a_lgb'])
    for fold in range(n_folds):
        p_lgb = models_dict['task_a_lgb'][fold].predict_proba(X)[:, 1]
        p_xgb = models_dict['task_a_xgb'][fold].predict_proba(X)[:, 1]
        p_hgb = models_dict['task_a_hgb'][fold].predict_proba(X)[:, 1]
        prob_preds += (0.4 * p_lgb + 0.4 * p_xgb + 0.2 * p_hgb) / n_folds
        
    binary_preds = (prob_preds >= best_thresh).astype(int)
    
    # 4. Model Inference - Task B1 (Onset Day Offset)
    print(">>> Generating Task B1 Predictions (Onset Day Offset)...")
    onset_preds = np.zeros(len(X))
    for fold in range(n_folds):
        onset_preds += models_dict['task_b1_onset'][fold].predict(X) / n_folds
    onset_preds_int = np.clip(np.round(onset_preds), 1, 30).astype(int)
    
    # 5. Model Inference - Task B2 (Recovery Duration)
    print(">>> Generating Task B2 Predictions (Recovery Duration)...")
    rec_preds = np.zeros(len(X))
    for fold in range(n_folds):
        rec_preds += models_dict['task_b2_rec'][fold].predict(X) / n_folds
    rec_preds_int = np.clip(np.round(rec_preds), 5, 20).astype(int)
    
    # 6. Assemble Submission DataFrame matching competition format exactly
    submission_df = pd.DataFrame({
        'athlete_id': athlete_ids,
        'injured_in_risk_window': binary_preds,
        'onset_day_offset': onset_preds_int,
        'recovery_duration': rec_preds_int
    })
    
    submission_df.to_csv(output_csv, index=False)
    print(f"\n>>> Successfully generated submission file: {output_csv}")
    print(f"Shape: {submission_df.shape}")
    print("First 10 submission rows:")
    print(submission_df.head(10))
    print("\nPredicted Injury Distribution:")
    print(submission_df['injured_in_risk_window'].value_counts())
    
    return submission_df

if __name__ == '__main__':
    generate_predictions()
