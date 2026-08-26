import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def extract_features(data_dir='dataset(31)', is_train=True):
    """
    Extracts all sports science, biometric, activity, sleep, and session features
    strictly using the Observation Window (Days 1 to 30: 2026-01-05 to 2026-02-03).
    """
    print(">>> Loading Metadata...")
    meta_df = pd.read_csv(os.path.join(data_dir, 'Athlete Metadata.csv'))
    
    # 1. Demographic & Baseline Biometrics
    meta_df['bmi_baseline'] = meta_df['weight_kg_baseline'] / ((meta_df['height_cm'] / 100.0) ** 2)
    meta_df['experience_ratio'] = meta_df['years_playing'] / meta_df['age']
    meta_df['is_contact_sport'] = meta_df['sport'].isin(['Football', 'Basketball']).astype(int)
    meta_df['is_court_sport'] = meta_df['sport'].isin(['Tennis', 'Badminton', 'Volleyball']).astype(int)
    
    # 2. Daily Activity (Observation Window: Days 1-30)
    print(">>> Processing Daily Activity (Observation Window)...")
    daily_df = pd.read_csv(os.path.join(data_dir, 'Daily Activity Merged.csv'))
    daily_df['ActivityDate'] = pd.to_datetime(daily_df['ActivityDate'])
    
    # Determine the first 30 dates
    unique_dates = sorted(daily_df['ActivityDate'].unique())
    obs_dates = unique_dates[:30]
    last_7_dates = obs_dates[-7:]
    
    daily_obs = daily_df[daily_df['ActivityDate'].isin(obs_dates)].copy()
    daily_last7 = daily_df[daily_df['ActivityDate'].isin(last_7_dates)].copy()
    
    # Full 30-day chronic activity stats
    daily_chronic = daily_obs.groupby('Id').agg({
        'TotalSteps': ['mean', 'std', 'max', 'min', 'sum'],
        'TotalDistance': ['mean', 'max', 'sum'],
        'VeryActiveMinutes': ['mean', 'sum', 'max'],
        'FairlyActiveMinutes': ['mean', 'sum'],
        'LightlyActiveMinutes': ['mean', 'sum'],
        'SedentaryMinutes': ['mean', 'std'],
        'Calories': ['mean', 'std', 'max', 'sum']
    })
    daily_chronic.columns = ['chronic_' + '_'.join(c) for c in daily_chronic.columns]
    daily_chronic.reset_index(inplace=True)
    daily_chronic.rename(columns={'Id': 'athlete_id'}, inplace=True)
    
    # Acute 7-day activity stats (Days 24-30)
    daily_acute = daily_last7.groupby('Id').agg({
        'TotalSteps': ['mean', 'sum', 'max'],
        'VeryActiveMinutes': ['mean', 'sum'],
        'Calories': ['mean', 'sum']
    })
    daily_acute.columns = ['acute_' + '_'.join(c) for c in daily_acute.columns]
    daily_acute.reset_index(inplace=True)
    daily_acute.rename(columns={'Id': 'athlete_id'}, inplace=True)
    
    # Merge Daily Stats & Compute ACWR (Acute-to-Chronic Workload Ratio)
    daily_feats = daily_chronic.merge(daily_acute, on='athlete_id')
    daily_feats['acwr_steps'] = daily_feats['acute_TotalSteps_mean'] / (daily_feats['chronic_TotalSteps_mean'] + 1e-5)
    daily_feats['acwr_very_active'] = daily_feats['acute_VeryActiveMinutes_mean'] / (daily_feats['chronic_VeryActiveMinutes_mean'] + 1e-5)
    daily_feats['acwr_calories'] = daily_feats['acute_Calories_mean'] / (daily_feats['chronic_Calories_mean'] + 1e-5)
    
    # Peak-to-chronic step spike ratio
    daily_feats['step_spike_ratio'] = daily_feats['chronic_TotalSteps_max'] / (daily_feats['chronic_TotalSteps_mean'] + 1e-5)
    daily_feats['step_spike_delta'] = daily_feats['chronic_TotalSteps_max'] - daily_feats['chronic_TotalSteps_mean']
    
    # Low activity rest days count (steps < 3500)
    rest_days = daily_obs[daily_obs['TotalSteps'] < 3500].groupby('Id')['ActivityDate'].count().reset_index()
    rest_days.columns = ['athlete_id', 'rest_days_count']
    daily_feats = daily_feats.merge(rest_days, on='athlete_id', how='left').fillna({'rest_days_count': 0})
    
    # 3. Sleep Metrics (Observation Window)
    print(">>> Processing Sleep Biometrics...")
    sleep_df = pd.read_csv(os.path.join(data_dir, 'Sleep Day Merged Dataset.csv'))
    sleep_df['SleepDay'] = pd.to_datetime(sleep_df['SleepDay'])
    sleep_obs = sleep_df[sleep_df['SleepDay'].isin(obs_dates)].copy()
    sleep_last7 = sleep_df[sleep_df['SleepDay'].isin(last_7_dates)].copy()
    
    sleep_obs['sleep_efficiency'] = sleep_obs['TotalMinutesAsleep'] / (sleep_obs['TotalTimeInBed'] + 1e-5)
    sleep_obs['sleep_deficit_480'] = np.maximum(0, 480 - sleep_obs['TotalMinutesAsleep'])
    
    sleep_chronic = sleep_obs.groupby('Id').agg({
        'TotalMinutesAsleep': ['mean', 'min', 'max', 'std'],
        'TotalTimeInBed': ['mean', 'std'],
        'sleep_efficiency': ['mean', 'min', 'std'],
        'sleep_deficit_480': ['mean', 'sum']
    })
    sleep_chronic.columns = ['sleep_' + '_'.join(c) for c in sleep_chronic.columns]
    sleep_chronic.reset_index(inplace=True)
    sleep_chronic.rename(columns={'Id': 'athlete_id'}, inplace=True)
    
    # Sleep Regularity (Coefficient of Variation)
    sleep_chronic['sleep_cv'] = sleep_chronic['sleep_TotalMinutesAsleep_std'] / (sleep_chronic['sleep_TotalMinutesAsleep_mean'] + 1e-5)
    
    # Acute sleep (last 7 days)
    sleep_acute = sleep_last7.groupby('Id').agg({
        'TotalMinutesAsleep': 'mean'
    }).reset_index().rename(columns={'TotalMinutesAsleep': 'acute_sleep_mean', 'Id': 'athlete_id'})
    
    sleep_feats = sleep_chronic.merge(sleep_acute, on='athlete_id')
    sleep_feats['sleep_acwr'] = sleep_feats['acute_sleep_mean'] / (sleep_feats['sleep_TotalMinutesAsleep_mean'] + 1e-5)
    
    # 4. Training Sessions (Observation Window)
    print(">>> Processing Training Sessions...")
    sess_df = pd.read_csv(os.path.join(data_dir, 'Training Sessions Dataset.csv'))
    sess_df['date'] = pd.to_datetime(sess_df['date'])
    sess_obs = sess_df[sess_df['date'].isin(obs_dates)].copy()
    sess_obs['duration_hours'] = sess_obs['end_hour'] - sess_obs['start_hour']
    
    sess_agg = sess_obs.groupby('athlete_id').agg(
        total_training_sessions=('session_id', 'count'),
        total_training_hours=('duration_hours', 'sum'),
        mean_session_duration=('duration_hours', 'mean'),
        practice_count=('sport_session_type', lambda x: (x == 'practice').sum()),
        gym_count=('sport_session_type', lambda x: (x == 'gym').sum()),
        scrimmage_count=('sport_session_type', lambda x: (x == 'scrimmage').sum()),
    ).reset_index()
    
    sess_agg['scrimmage_ratio'] = sess_agg['scrimmage_count'] / (sess_agg['total_training_sessions'] + 1e-5)
    sess_agg['gym_ratio'] = sess_agg['gym_count'] / (sess_agg['total_training_sessions'] + 1e-5)
    
    # 5. Weight Log Info
    print(">>> Processing Weight Log Info...")
    weight_df = pd.read_csv(os.path.join(data_dir, 'Weight Log Info Merged.csv'))
    weight_df['Date'] = pd.to_datetime(weight_df['Date'])
    weight_obs = weight_df[weight_df['Date'].isin(obs_dates)].copy()
    
    if len(weight_obs) > 0:
        weight_agg = weight_obs.groupby('Id').agg({
            'WeightKg': ['mean', 'min', 'max', 'count'],
            'BMI': ['mean', 'max']
        })
        weight_agg.columns = ['weight_' + '_'.join(c) for c in weight_agg.columns]
        weight_agg.reset_index(inplace=True)
        weight_agg.rename(columns={'Id': 'athlete_id'}, inplace=True)
    else:
        weight_agg = pd.DataFrame({'athlete_id': meta_df['athlete_id'].unique()})
    
    # 6. Hourly High-Frequency Biometrics (Optimized Processing)
    print(">>> Processing Hourly Heart Rate Biometrics...")
    hr_path = os.path.join(data_dir, 'Hourly Heart Rate Dataset.csv')
    
    obs_date_strs = set([d.strftime('%Y-%m-%d') for d in obs_dates])
    
    hr_chunks = []
    for chunk in pd.read_csv(hr_path, chunksize=500000):
        chunk['DateStr'] = chunk['ActivityHour'].str[:10]
        obs_chunk = chunk[chunk['DateStr'].isin(obs_date_strs)]
        if len(obs_chunk) > 0:
            obs_chunk['Hour'] = obs_chunk['ActivityHour'].str[11:13].astype(int)
            hr_chunks.append(obs_chunk)
            
    hr_obs = pd.concat(hr_chunks, ignore_index=True)
    
    # Nocturnal Resting Heart Rate (2 AM to 6 AM)
    nocturnal_hr = hr_obs[hr_obs['Hour'].isin([2, 3, 4, 5, 6])].groupby('Id').agg(
        resting_hr_mean=('MinHeartRate', 'mean'),
        resting_hr_min=('MinHeartRate', 'min'),
        resting_hr_std=('MinHeartRate', 'std')
    ).reset_index().rename(columns={'Id': 'athlete_id'})
    
    # Peak Exertion Heart Rate
    peak_hr = hr_obs.groupby('Id').agg(
        peak_hr_max=('MaxHeartRate', 'max'),
        avg_hr_mean=('AvgHeartRate', 'mean'),
        cardiac_strain_hours=('MaxHeartRate', lambda x: (x >= 140).sum())
    ).reset_index().rename(columns={'Id': 'athlete_id'})
    
    hr_feats = nocturnal_hr.merge(peak_hr, on='athlete_id')
    hr_feats['heart_rate_reserve'] = hr_feats['peak_hr_max'] - hr_feats['resting_hr_mean']
    
    # 7. Merge All Features
    print(">>> Assembling Master Feature Matrix...")
    features = meta_df.merge(daily_feats, on='athlete_id', how='left')
    features = features.merge(sleep_feats, on='athlete_id', how='left')
    features = features.merge(sess_agg, on='athlete_id', how='left')
    features = features.merge(weight_agg, on='athlete_id', how='left')
    features = features.merge(hr_feats, on='athlete_id', how='left')
    
    # Fill remaining NaNs
    features.fillna(0, inplace=True)
    
    # One-hot / categorical encodings
    sport_dummies = pd.get_dummies(features['sport'], prefix='sport', drop_first=False)
    gender_dummies = pd.get_dummies(features['gender'], prefix='gender', drop_first=True)
    dominant_dummies = pd.get_dummies(features['dominant_side'], prefix='dominant', drop_first=True)
    position_dummies = pd.get_dummies(features['position'], prefix='pos', drop_first=True)
    
    features = pd.concat([features, sport_dummies, gender_dummies, dominant_dummies, position_dummies], axis=1)
    
    if is_train and os.path.exists(os.path.join(data_dir, 'Train Labels Dataset.csv')):
        labels_df = pd.read_csv(os.path.join(data_dir, 'Train Labels Dataset.csv'))
        features = features.merge(labels_df, on='athlete_id', how='left')
        
    print(f">>> Feature Extraction Complete! Shape: {features.shape}")
    return features

if __name__ == '__main__':
    df = extract_features()
    os.makedirs('processed_data', exist_ok=True)
    df.to_csv('processed_data/master_features.csv', index=False)
    print("Saved processed_data/master_features.csv successfully!")
