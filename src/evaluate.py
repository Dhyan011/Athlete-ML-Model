import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_competition_metrics(y_true_cls, y_pred_cls,
                                onset_true, onset_pred,
                                recovery_true, recovery_pred,
                                penalty=30.0):
    """
    Computes competition scoring metrics:
    - Task A: F1, Precision, Recall
    - Task B: Penalized MAE and skill score relative to mean baseline
    - Composite Score: Mean of F1, Onset Skill, and Recovery Skill
    """
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)

    injured_mask = (y_true_cls == 1)
    n_injured = np.sum(injured_mask)

    if n_injured == 0:
        return {
            'task_a_f1': f1, 'task_a_precision': precision, 'task_a_recall': recall,
            'mae_onset_model': 0.0, 'mae_onset_base': 0.0, 'task_b_onset_skill': 0.0,
            'mae_rec_model': 0.0, 'mae_rec_base': 0.0, 'task_b_recovery_skill': 0.0,
            'composite_score': f1
        }

    true_onset_inj = onset_true[injured_mask]
    pred_onset_inj = onset_pred[injured_mask]
    true_rec_inj = recovery_true[injured_mask]
    pred_rec_inj = recovery_pred[injured_mask]
    pred_cls_inj = y_pred_cls[injured_mask]

    onset_errors = np.where(pred_cls_inj == 1, np.abs(pred_onset_inj - true_onset_inj), penalty)
    mae_onset_model = np.mean(onset_errors)
    mean_onset_base = np.mean(true_onset_inj)
    mae_onset_base = np.mean(np.abs(mean_onset_base - true_onset_inj))
    onset_skill = max(0.0, 1.0 - (mae_onset_model / (mae_onset_base + 1e-8)))

    rec_errors = np.where(pred_cls_inj == 1, np.abs(pred_rec_inj - true_rec_inj), penalty)
    mae_rec_model = np.mean(rec_errors)
    mean_rec_base = np.mean(true_rec_inj)
    mae_rec_base = np.mean(np.abs(mean_rec_base - true_rec_inj))
    recovery_skill = max(0.0, 1.0 - (mae_rec_model / (mae_rec_base + 1e-8)))

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
