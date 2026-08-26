import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def compute_competition_metrics(y_true_cls, y_pred_cls, 
                                onset_true, onset_pred,
                                recovery_true, recovery_pred,
                                penalty=30.0):
    """
    Computes official competition metrics:
    - Task A: F1-Score in [0, 1]
    - Task B: Skill Scores for onset_day_offset and recovery_duration
      - Hits (y_pred=1, y_true=1): Absolute Error |pred - true|
      - Misses (y_pred=0, y_true=1): Fixed penalty = 30
      - Baseline: Mean of true targets
      - Skill Score = max(0, 1 - MAE_model / MAE_baseline)
    """
    # 1. Classification Metrics (Task A)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    
    # 2. Timing Metrics (Task B - evaluated only on truly injured athletes y_true_cls == 1)
    injured_mask = (y_true_cls == 1)
    n_injured = np.sum(injured_mask)
    
    if n_injured == 0:
        return {'f1': f1, 'precision': precision, 'recall': recall, 'onset_skill': 0.0, 'recovery_skill': 0.0, 'overall_score': f1}
    
    # Extract injured subsets
    true_onset_inj = onset_true[injured_mask]
    pred_onset_inj = onset_pred[injured_mask]
    
    true_rec_inj = recovery_true[injured_mask]
    pred_rec_inj = recovery_pred[injured_mask]
    
    pred_cls_inj = y_pred_cls[injured_mask]
    
    # Compute penalized errors for Onset Day
    onset_errors = np.where(
        pred_cls_inj == 1,
        np.abs(pred_onset_inj - true_onset_inj),
        penalty
    )
    mae_onset_model = np.mean(onset_errors)
    
    # Baseline onset error (predicting mean onset for all)
    mean_onset_base = np.mean(true_onset_inj)
    mae_onset_base = np.mean(np.abs(mean_onset_base - true_onset_inj))
    onset_skill = max(0.0, 1.0 - (mae_onset_model / (mae_onset_base + 1e-8)))
    
    # Compute penalized errors for Recovery Duration
    rec_errors = np.where(
        pred_cls_inj == 1,
        np.abs(pred_rec_inj - true_rec_inj),
        penalty
    )
    mae_rec_model = np.mean(rec_errors)
    
    # Baseline recovery error (predicting mean recovery for all)
    mean_rec_base = np.mean(true_rec_inj)
    mae_rec_base = np.mean(np.abs(mean_rec_base - true_rec_inj))
    recovery_skill = max(0.0, 1.0 - (mae_rec_model / (mae_rec_base + 1e-8)))
    
    # Composite Score (Harmonic / Arithmetic mean of F1 and Skill Scores)
    composite_score = (f1 + onset_skill + recovery_skill) / 3.0
    
    return {
        'task_a_f1': f1,
        'task_a_precision': precision,
        'task_a_recall': recall,
        'mae_onset_model': mae_onset_model,
        'mae_onset_base': mae_onset_base,
        'task_b_onset_skill': onset_skill,
        'mae_rec_model': mae_rec_model,
        'mae_rec_base': mae_rec_base,
        'task_b_recovery_skill': recovery_skill,
        'composite_score': composite_score
    }
