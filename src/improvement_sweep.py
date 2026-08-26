"""
Rigorous improvement sweep with honest ablation reporting.
Every change is tested against the same 5-fold StratifiedKFold(random_state=42).
"""
import os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve
sys.path.insert(0, 'src')
from evaluate import compute_competition_metrics

DATA_PATH = 'processed_data/master_features_v2.csv'
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    target_cols = ['injured_in_risk_window', 'onset_day_offset', 'recovery_duration']
    id_cols = ['athlete_id', 'sport', 'gender', 'dominant_side', 'position', 'team_id']
    feature_cols = [c for c in df.columns if c not in target_cols and c not in id_cols]
    X = df[feature_cols].copy()
    y = df['injured_in_risk_window'].values
    return X, y, feature_cols, df

def evaluate_xgb(X, y, params, label=""):
    oof_probs = np.zeros(len(X))
    fold_aucs = []
    for fold, (tr, va) in enumerate(SKF.split(X, y)):
        clf = xgb.XGBClassifier(**params, random_state=42+fold, eval_metric='logloss')
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[va])[:, 1]
        oof_probs[va] = p
        fold_aucs.append(roc_auc_score(y[va], p))

    auc = roc_auc_score(y, oof_probs)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.60, 0.005):
        f = f1_score(y, (oof_probs >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t

    brier = brier_score_loss(y, oof_probs)
    return {
        'label': label,
        'auc': auc,
        'f1': best_f1,
        'f1_thresh': best_t,
        'brier': brier,
        'fold_aucs': fold_aucs,
        'oof_probs': oof_probs
    }

# ====================================================================
print("=" * 70)
print("STEP 1: BASELINE (current shipped XGBoost)")
print("=" * 70)

X, y, feat_cols, df = load_data()
baseline_params = dict(
    n_estimators=350, learning_rate=0.025, max_depth=5,
    subsample=0.85, colsample_bytree=0.8, scale_pos_weight=1.2
)
baseline = evaluate_xgb(X, y, baseline_params, "Baseline (current)")
print(f"AUC: {baseline['auc']:.4f}  F1: {baseline['f1']:.4f} @{baseline['f1_thresh']:.3f}  Brier: {baseline['brier']:.4f}")
print(f"Per-fold AUC: {[f'{a:.4f}' for a in baseline['fold_aucs']]}")

# ====================================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING VERIFICATION")
print("=" * 70)

# 2a. Verify ACWR computation
daily = pd.read_csv('dataset(31)/Daily Activity Merged.csv')
daily['ActivityDate'] = pd.to_datetime(daily['ActivityDate'])
dates = sorted(daily['ActivityDate'].unique())
obs_dates = dates[:30]
last7 = obs_dates[-7:]

athlete_1_obs = daily[(daily['Id'] == 1) & (daily['ActivityDate'].isin(obs_dates))]
athlete_1_last7 = daily[(daily['Id'] == 1) & (daily['ActivityDate'].isin(last7))]
chronic_mean = athlete_1_obs['TotalSteps'].mean()
acute_mean = athlete_1_last7['TotalSteps'].mean()
correct_acwr = acute_mean / chronic_mean
stored_acwr = df[df['athlete_id'] == 1]['acwr_steps'].values[0]

print(f"Athlete 1 ACWR verification:")
print(f"  Chronic (30d) mean steps: {chronic_mean:.2f}")
print(f"  Acute (7d) mean steps:    {acute_mean:.2f}")
print(f"  Correct ACWR:             {correct_acwr:.6f}")
print(f"  Stored ACWR:              {stored_acwr:.6f}")
print(f"  Match: {abs(correct_acwr - stored_acwr) < 0.001}")

# 2b. Check prior_season_injury_count distribution
meta = pd.read_csv('dataset(31)/Athlete Metadata.csv')
labels = pd.read_csv('dataset(31)/Train Labels Dataset.csv')
merged = meta.merge(labels, on='athlete_id')
print(f"\nPrior injury count vs injury rate:")
for cnt in sorted(merged['prior_season_injury_count'].unique()):
    subset = merged[merged['prior_season_injury_count'] == cnt]
    rate = subset['injured_in_risk_window'].mean()
    print(f"  prior_count={cnt}: n={len(subset)}, injury_rate={rate:.3f}")

# ====================================================================
print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING IMPROVEMENTS")
print("=" * 70)

# 3a. Add binary has_prior_injury + interaction
X_improved = X.copy()
X_improved['has_prior_injury'] = (X_improved['prior_season_injury_count'] > 0).astype(int)
X_improved['prior_x_age'] = X_improved['prior_season_injury_count'] * X_improved['age']
X_improved['prior_x_contact'] = X_improved['prior_season_injury_count'] * X_improved['is_contact_sport']

# 3b. Consecutive high-load days (proxy from daily data)
# Count days where TotalSteps > athlete's own median
daily_obs = daily[daily['ActivityDate'].isin(obs_dates)].copy()
athlete_medians = daily_obs.groupby('Id')['TotalSteps'].median().reset_index()
athlete_medians.columns = ['Id', 'median_steps']
daily_obs = daily_obs.merge(athlete_medians, on='Id')
daily_obs['above_median'] = (daily_obs['TotalSteps'] > daily_obs['median_steps'] * 1.3).astype(int)
daily_obs = daily_obs.sort_values(['Id', 'ActivityDate'])

# Max consecutive above-median days
def max_consecutive(series):
    max_streak, current = 0, 0
    for v in series:
        if v == 1: current += 1
        else: current = 0
        max_streak = max(max_streak, current)
    return max_streak

streaks = daily_obs.groupby('Id')['above_median'].apply(max_consecutive).reset_index()
streaks.columns = ['athlete_id', 'max_overload_streak']
streaks_dict = dict(zip(streaks['athlete_id'], streaks['max_overload_streak']))
X_improved['max_overload_streak'] = df['athlete_id'].map(streaks_dict).fillna(0)

result_feat = evaluate_xgb(X_improved, y, baseline_params, "Feature improvements")
print(f"Feature improvements: AUC={result_feat['auc']:.4f} (delta: {result_feat['auc']-baseline['auc']:+.4f})  F1={result_feat['f1']:.4f}")

# ====================================================================
print("\n" + "=" * 70)
print("STEP 4: HYPERPARAMETER TUNING (grid search)")
print("=" * 70)

results = []
param_grid = [
    dict(n_estimators=350, learning_rate=0.025, max_depth=5, subsample=0.85, colsample_bytree=0.8, min_child_weight=1, scale_pos_weight=1.2),
    dict(n_estimators=500, learning_rate=0.02, max_depth=5, subsample=0.85, colsample_bytree=0.8, min_child_weight=1, scale_pos_weight=1.2),
    dict(n_estimators=350, learning_rate=0.025, max_depth=6, subsample=0.85, colsample_bytree=0.8, min_child_weight=1, scale_pos_weight=1.2),
    dict(n_estimators=350, learning_rate=0.025, max_depth=4, subsample=0.85, colsample_bytree=0.8, min_child_weight=1, scale_pos_weight=1.2),
    dict(n_estimators=500, learning_rate=0.015, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, scale_pos_weight=1.2),
    dict(n_estimators=500, learning_rate=0.015, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=5, scale_pos_weight=1.2),
    dict(n_estimators=600, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, scale_pos_weight=1.2),
    dict(n_estimators=600, learning_rate=0.01, max_depth=6, subsample=0.85, colsample_bytree=0.75, min_child_weight=5, scale_pos_weight=1.2),
    dict(n_estimators=400, learning_rate=0.02, max_depth=5, subsample=0.9, colsample_bytree=0.85, min_child_weight=3, scale_pos_weight=1.2),
    dict(n_estimators=500, learning_rate=0.02, max_depth=5, subsample=0.85, colsample_bytree=0.75, min_child_weight=5, scale_pos_weight=1.3),
]

# Test on improved features
for i, params in enumerate(param_grid):
    r = evaluate_xgb(X_improved, y, params, f"Grid {i}")
    results.append(r)
    print(f"Grid {i}: depth={params['max_depth']} lr={params['learning_rate']} n={params['n_estimators']} mcw={params['min_child_weight']} spw={params.get('scale_pos_weight',1)} | AUC={r['auc']:.4f} F1={r['f1']:.4f}")

best_grid = max(results, key=lambda r: r['auc'])
print(f"\nBest grid config: {best_grid['label']} with AUC={best_grid['auc']:.4f}")

# ====================================================================
print("\n" + "=" * 70)
print("STEP 5: CLASS IMBALANCE EXPERIMENTS")
print("=" * 70)

best_grid_idx = results.index(best_grid)
best_params = param_grid[best_grid_idx].copy()

for spw in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
    p = best_params.copy()
    p['scale_pos_weight'] = spw
    r = evaluate_xgb(X_improved, y, p, f"SPW={spw}")
    print(f"  scale_pos_weight={spw:.1f}: AUC={r['auc']:.4f} F1={r['f1']:.4f} Brier={r['brier']:.4f}")

# ====================================================================
print("\n" + "=" * 70)
print("STEP 6: CALIBRATION CHECK")
print("=" * 70)

# Use baseline OOF probs
probs = baseline['oof_probs']
frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
print("Calibration (reliability diagram):")
print(f"{'Bin Mean Pred':>15s} | {'Actual Pos Frac':>15s} | {'Deviation':>10s}")
for mp, fp in zip(mean_pred, frac_pos):
    print(f"{mp:15.3f} | {fp:15.3f} | {fp - mp:+10.3f}")
print(f"\nBrier Score: {baseline['brier']:.4f}")

# ====================================================================
print("\n" + "=" * 70)
print("STEP 7: FINAL ABLATION TABLE")
print("=" * 70)

# Determine best overall config
# Re-run best grid params with best SPW
final_configs = []

# A: Pure baseline (original shipped)
final_configs.append(('A. Baseline (shipped)', X, baseline_params, baseline))

# B: Feature improvements only
final_configs.append(('B. +Feature fixes', X_improved, baseline_params, result_feat))

# C: Best HP only (on original features)
best_hp = param_grid[best_grid_idx].copy()
r_hp = evaluate_xgb(X, y, best_hp, "HP-only")
final_configs.append(('C. +HP tuning only', X, best_hp, r_hp))

# D: Feature improvements + best HP
r_both = evaluate_xgb(X_improved, y, best_hp, "Feat+HP")
final_configs.append(('D. +Features +HP', X_improved, best_hp, r_both))

# E: Features + HP + best SPW (if different)
# Find the best SPW
best_spw_auc = 0
best_spw_val = 1.2
for spw in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
    p = best_hp.copy()
    p['scale_pos_weight'] = spw
    r = evaluate_xgb(X_improved, y, p, f"SPW={spw}")
    if r['auc'] > best_spw_auc:
        best_spw_auc = r['auc']
        best_spw_val = spw
        best_spw_result = r

best_hp_final = best_hp.copy()
best_hp_final['scale_pos_weight'] = best_spw_val
r_final = evaluate_xgb(X_improved, y, best_hp_final, "Final")
final_configs.append((f'E. +Features +HP +SPW={best_spw_val}', X_improved, best_hp_final, r_final))

print(f"\n{'Config':<35s} | {'AUC':>7s} | {'F1':>7s} | {'Brier':>7s} | {'Per-fold AUC'}")
print("-" * 100)
for label, _, _, r in final_configs:
    folds_str = ', '.join([f'{a:.4f}' for a in r['fold_aucs']])
    print(f"{label:<35s} | {r['auc']:7.4f} | {r['f1']:7.4f} | {r['brier']:7.4f} | [{folds_str}]")

# Print the final best config
best_final = max(final_configs, key=lambda x: x[3]['auc'])
print(f"\n>>> Best configuration: {best_final[0]}")
print(f">>> Final AUC: {best_final[3]['auc']:.4f}")
print(f">>> Final F1:  {best_final[3]['f1']:.4f}")
print(f">>> Final best params: {best_hp_final}")
