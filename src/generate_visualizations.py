import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

os.makedirs('figures', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 12, 'figure.dpi': 300})

# Load datasets
oof = pd.read_csv('processed_data/oof_predictions.csv')
features = pd.read_csv('processed_data/master_features.csv')
meta = pd.read_csv('dataset(31)/Athlete Metadata.csv')
labels = pd.read_csv('dataset(31)/Train Labels Dataset.csv')

# 1. Sport vs Injury Rate & Recovery Duration
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sport_stats = meta.merge(labels, on='athlete_id')
sport_inj = sport_stats.groupby('sport')['injured_in_risk_window'].mean().reset_index()
sport_inj['injury_rate_pct'] = sport_inj['injured_in_risk_window'] * 100

sns.barplot(data=sport_inj, x='sport', y='injury_rate_pct', ax=ax1, palette='Blues_r')
ax1.set_title('A: Injury Incidence Rate by Sport (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Injury Rate (%)')
ax1.set_xlabel('Sport Category')
ax1.set_ylim(0, 50)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                 ha='center', va='center', fontsize=11, fontweight='semibold')

# Bimodal Recovery Duration
inj_df = sport_stats[sport_stats['injured_in_risk_window'] == 1]
sns.boxplot(data=inj_df, x='sport', y='recovery_duration', ax=ax2, palette='Set2')
ax2.set_title('B: Recovery Duration Bimodality by Sport', fontsize=14, fontweight='bold')
ax2.set_ylabel('Recovery Duration (Days)')
ax2.set_xlabel('Sport Category')

plt.tight_layout()
plt.savefig('figures/01_sport_and_recovery_analysis.png')
plt.close()
print("Saved figures/01_sport_and_recovery_analysis.png")

# 2. Acute vs Chronic Workload Spike Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
features_inj = features.copy()
features_inj['Injury_Status'] = features_inj['injured_in_risk_window'].map({0: 'Non-Injured', 1: 'Injured'})

sns.kdeplot(data=features_inj, x='chronic_TotalSteps_max', hue='Injury_Status', common_norm=False, fill=True, ax=ax1, palette=['#2b5c8f', '#d95f02'])
ax1.set_title('A: Peak Single-Day Workload Spike (Steps)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Maximum Single-Day Steps in Observation Window')

sns.boxplot(data=features_inj, x='Injury_Status', y='acwr_steps', ax=ax2, palette=['#2b5c8f', '#d95f02'])
ax2.set_title('B: Acute-to-Chronic Workload Ratio (ACWR)', fontsize=14, fontweight='bold')
ax2.set_ylabel('ACWR (Last 7 Days / 30 Days Mean)')
ax2.axhline(1.5, color='red', linestyle='--', label='High Risk Danger Zone (>1.5)')
ax2.legend()

plt.tight_layout()
plt.savefig('figures/02_workload_spikes_and_acwr.png')
plt.close()
print("Saved figures/02_workload_spikes_and_acwr.png")

# 3. Model Classification Evaluation (ROC & Confusion Matrix)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

fpr, tpr, _ = roc_curve(oof['true_injured'], oof['pred_prob'])
roc_auc = auc(fpr, tpr)

ax1.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Ensemble ROC (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
ax1.set_title('A: Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')

cm = confusion_matrix(oof['true_injured'], oof['pred_injured'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
ax2.set_title('B: Out-of-Fold Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_ylabel('True Condition')
ax2.set_xlabel('Predicted Condition')

plt.tight_layout()
plt.savefig('figures/03_model_classification_performance.png')
plt.close()
print("Saved figures/03_model_classification_performance.png")

# 4. Feature Importance Plot
models_dict = joblib.load('models/trained_ensemble.joblib')
meta_dict = joblib.load('models/pipeline_metadata.joblib')
feature_cols = meta_dict['feature_cols']

importances = np.mean([m.feature_importances_ for m in models_dict['task_a_lgb']], axis=0)
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values(by='importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, y='feature', x='importance', palette='viridis')
plt.title('Top 15 Most Predictive Features for Athlete Injury (LightGBM)', fontsize=14, fontweight='bold')
plt.xlabel('Mean Feature Importance Gain')
plt.ylabel('Engineered Feature')
plt.tight_layout()
plt.savefig('figures/04_feature_importance.png')
plt.close()
print("Saved figures/04_feature_importance.png")

# 5. Timing Target Regression Accuracy (Hits)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
hits = (oof['true_injured'] == 1) & (oof['pred_injured'] == 1)

ax1.scatter(oof.loc[hits, 'true_onset'], oof.loc[hits, 'pred_onset'], alpha=0.6, color='#1f77b4', edgecolors='none')
ax1.plot([1, 30], [1, 30], 'r--', lw=2)
ax1.set_title('A: Onset Day Offset Prediction (Hits)', fontsize=14, fontweight='bold')
ax1.set_xlabel('True Onset Day (1 to 30)')
ax1.set_ylabel('Predicted Onset Day')

ax2.scatter(oof.loc[hits, 'true_recovery'], oof.loc[hits, 'pred_recovery'], alpha=0.6, color='#2ca02c', edgecolors='none')
ax2.plot([5, 20], [5, 20], 'r--', lw=2)
ax2.set_title('B: Recovery Duration Prediction (Hits)', fontsize=14, fontweight='bold')
ax2.set_xlabel('True Recovery Days (5 to 20)')
ax2.set_ylabel('Predicted Recovery Days')

plt.tight_layout()
plt.savefig('figures/05_timing_predictions_evaluation.png')
plt.close()
print("Saved figures/05_timing_predictions_evaluation.png")
