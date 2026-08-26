import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def extract_advanced_features(data_dir='dataset(31)', is_train=True):
    print("=======================================================")
    print("     EXTRACTING ADVANCED SPORTS SCIENCE BIOMETRICS     ")
    print("=======================================================")
    
    # 1. Athlete Metadata
    print(">>> 1. Processing Athlete Profiles & Physical Baselines...")
    meta_df = pd.read_csv(os.path.join(data_dir, 'Athlete Metadata.csv'))
    
    meta_df['bmi_baseline'] = meta_df['weight_kg_baseline'] / ((meta_df['height_cm'] / 100.0) ** 2)
    meta_df['experience_ratio'] = meta_df['years_playing'] / (meta_df['age'] + 1e-5)
    meta_df['age_prior_injury_interaction'] = meta_df['age'] * (meta_df['prior_season_injury_count'] + 1)
    meta_df['is_contact_sport'] = meta_df['sport'].isin(['Football', 'Basketball']).astype(int)
    meta_df['is_racket_sport'] = meta_df['sport'].isin(['Tennis', 'Badminton']).astype(int)
    meta_df['is_high_impact_sport'] = meta_df['sport'].isin(['Football', 'Basketball', 'Athletics']).astype(int)
    meta_df['has_prior_injury'] = (meta_df['prior_season_injury_count'] > 0).astype(int)
    meta_df['prior_x_age'] = meta_df['prior_season_injury_count'] * meta_df['age']
    meta_df['prior_x_contact'] = meta_df['prior_season_injury_count'] * meta_df['is_contact_sport']
    
    # 2. Daily Activity (Observation Window: Days 1 to 30)
    print(">>> 2. Processing Daily Workload Dynamics (Days 1 to 30)...")
    daily_df = pd.read_csv(os.path.join(data_dir, 'Daily Activity Merged.csv'))
    daily_df['ActivityDate'] = pd.to_datetime(daily_df['ActivityDate'])
    
    unique_dates = sorted(daily_df['ActivityDate'].unique())
    obs_dates = unique_dates[:30]
    week1_dates = obs_dates[:7]
    week4_dates = obs_dates[-7:]
    
    daily_obs = daily_df[daily_df['ActivityDate'].isin(obs_dates)].copy()
    daily_obs.sort_values(by=['Id', 'ActivityDate'], inplace=True)
    
    # Daily Load Proxy (TRIMP approximation = VeryActive*3 + FairlyActive*2 + LightlyActive*1)
    daily_obs['daily_trimp'] = (daily_obs['VeryActiveMinutes'] * 3.0 + 
                                daily_obs['FairlyActiveMinutes'] * 2.0 + 
                                daily_obs['LightlyActiveMinutes'] * 1.0)
    
    # Compute Foster's Monotony and Strain per Athlete
    # Monotony = Mean Daily Load / Std Daily Load
    # Strain = Weekly Load * Monotony
    daily_stats = daily_obs.groupby('Id').agg({
        'TotalSteps': ['mean', 'std', 'max', 'min', 'sum'],
        'TotalDistance': ['mean', 'max', 'sum'],
        'VeryActiveMinutes': ['mean', 'sum', 'max', 'std'],
        'FairlyActiveMinutes': ['mean', 'sum'],
        'LightlyActiveMinutes': ['mean', 'sum'],
        'SedentaryMinutes': ['mean', 'std'],
        'Calories': ['mean', 'std', 'max', 'sum'],
        'daily_trimp': ['mean', 'std', 'sum', 'max']
    })
    daily_stats.columns = ['daily_' + '_'.join(c) for c in daily_stats.columns]
    daily_stats.reset_index(inplace=True)
    daily_stats.rename(columns={'Id': 'athlete_id'}, inplace=True)
    
    # Foster's Training Strain & Monotony
    daily_stats['foster_monotony'] = daily_stats['daily_daily_trimp_mean'] / (daily_stats['daily_daily_trimp_std'] + 1e-5)
    daily_stats['foster_strain'] = daily_stats['daily_daily_trimp_sum'] * daily_stats['foster_monotony']
    
    # Acute (Week 4: Days 24-30) vs Chronic (Full 30 days) Ratios
    week4_df = daily_obs[daily_obs['ActivityDate'].isin(week4_dates)]
    week1_df = daily_obs[daily_obs['ActivityDate'].isin(week1_dates)]
    
    w4_stats = week4_df.groupby('Id').agg({
        'TotalSteps': ['mean', 'max', 'sum'],
        'VeryActiveMinutes': ['mean', 'sum'],
        'Calories': ['mean', 'sum'],
        'daily_trimp': ['mean', 'sum']
    })
    w4_stats.columns = ['w4_' + '_'.join(c) for c in w4_stats.columns]
    w4_stats.reset_index(inplace=True)
    w4_stats.rename(columns={'Id': 'athlete_id'}, inplace=True)
    
    w1_stats = week1_df.groupby('Id').agg({
        'TotalSteps': 'mean',
        'VeryActiveMinutes': 'mean',
        'daily_trimp': 'mean'
    }).reset_index().rename(columns={
        'Id': 'athlete_id',
        'TotalSteps': 'w1_steps_mean',
        'VeryActiveMinutes': 'w1_very_active_mean',
        'daily_trimp': 'w1_trimp_mean'
    })
    
    daily_feats = daily_stats.merge(w4_stats, on='athlete_id').merge(w1_stats, on='athlete_id')
    
    # ACWR Ratios
    daily_feats['acwr_steps'] = daily_feats['w4_TotalSteps_mean'] / (daily_feats['daily_TotalSteps_mean'] + 1e-5)
    daily_feats['acwr_very_active'] = daily_feats['w4_VeryActiveMinutes_mean'] / (daily_feats['daily_VeryActiveMinutes_mean'] + 1e-5)
    daily_feats['acwr_trimp'] = daily_feats['w4_daily_trimp_mean'] / (daily_feats['daily_daily_trimp_mean'] + 1e-5)
    daily_feats['workload_ramp_w4_vs_w1'] = daily_feats['w4_TotalSteps_mean'] / (daily_feats['w1_steps_mean'] + 1e-5)
    
    # Peak Workload Spike Deltas
    daily_feats['step_spike_ratio'] = daily_feats['daily_TotalSteps_max'] / (daily_feats['daily_TotalSteps_mean'] + 1e-5)
    daily_feats['step_spike_delta'] = daily_feats['daily_TotalSteps_max'] - daily_feats['daily_TotalSteps_mean']
    daily_feats['trimp_spike_delta'] = daily_feats['daily_daily_trimp_max'] - daily_feats['daily_daily_trimp_mean']
    
    # Overload Streaks (Consecutive days > 1.3x median)
    athlete_medians = daily_obs.groupby('Id')['TotalSteps'].median().reset_index().rename(columns={'TotalSteps': 'median_steps'})
    daily_obs_streak = daily_obs.merge(athlete_medians, on='Id')
    daily_obs_streak['above_median'] = (daily_obs_streak['TotalSteps'] > daily_obs_streak['median_steps'] * 1.3).astype(int)
    
    def _calc_max_streak(series):
        max_s, curr = 0, 0
        for v in series:
            if v == 1: curr += 1
            else: curr = 0
            max_s = max(max_s, curr)
        return max_s
        
    streak_s = daily_obs_streak.groupby('Id')['above_median'].apply(_calc_max_streak).reset_index()
    streak_s.columns = ['athlete_id', 'max_overload_streak']
    daily_feats = daily_feats.merge(streak_s, on='athlete_id', how='left')
    
    # 3. Sleep Architecture & Cumulative Sleep Debt
    print(">>> 3. Processing Sleep Architecture & Deficits...")
    sleep_df = pd.read_csv(os.path.join(data_dir, 'Sleep Day Merged Dataset.csv'))
    sleep_df['SleepDay'] = pd.to_datetime(sleep_df['SleepDay'])
    sleep_obs = sleep_df[sleep_df['SleepDay'].isin(obs_dates)].copy()
    sleep_obs.sort_values(by=['Id', 'SleepDay'], inplace=True)
    
    sleep_obs['sleep_efficiency'] = sleep_obs['TotalMinutesAsleep'] / (sleep_obs['TotalTimeInBed'] + 1e-5)
    sleep_obs['sleep_deficit_480'] = np.maximum(0, 480 - sleep_obs['TotalMinutesAsleep'])
    sleep_obs['severe_sleep_deprived'] = (sleep_obs['TotalMinutesAsleep'] < 360).astype(int)
    
    sleep_stats = sleep_obs.groupby('Id').agg({
        'TotalMinutesAsleep': ['mean', 'min', 'max', 'std'],
        'TotalTimeInBed': ['mean', 'std'],
        'sleep_efficiency': ['mean', 'min', 'std'],
        'sleep_deficit_480': ['mean', 'sum'],
        'severe_sleep_deprived': ['sum']
    })
    sleep_stats.columns = ['sleep_' + '_'.join(c) for c in sleep_stats.columns]
    sleep_stats.reset_index(inplace=True)
    sleep_stats.rename(columns={'Id': 'athlete_id'}, inplace=True)
    
    # Sleep Regularity (CV)
    sleep_stats['sleep_regularity_cv'] = sleep_stats['sleep_TotalMinutesAsleep_std'] / (sleep_stats['sleep_TotalMinutesAsleep_mean'] + 1e-5)
    
    # Week 4 Sleep
    sleep_w4 = sleep_obs[sleep_obs['SleepDay'].isin(week4_dates)].groupby('Id').agg({
        'TotalMinutesAsleep': 'mean',
        'sleep_efficiency': 'mean'
    }).reset_index().rename(columns={'Id': 'athlete_id', 'TotalMinutesAsleep': 'w4_sleep_mean', 'sleep_efficiency': 'w4_sleep_efficiency'})
    
    sleep_feats = sleep_stats.merge(sleep_w4, on='athlete_id')
    sleep_feats['sleep_acwr'] = sleep_feats['w4_sleep_mean'] / (sleep_feats['sleep_TotalMinutesAsleep_mean'] + 1e-5)
    
    # 4. Training Sessions & Intensity Modality
    print(">>> 4. Processing Training Sessions & Practice Loads...")
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
    sess_agg['weekly_session_frequency'] = sess_agg['total_training_sessions'] / 4.28
    
    # 5. Weight & BMI Drift
    print(">>> 5. Processing Weight & Body Composition Drift...")
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
        
    # 6. High-Frequency Hourly Heart Rate Telemetry (Resting & Strain)
    print(">>> 6. Processing High-Frequency Hourly Heart Rate Streams...")
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
    
    # Peak Cardiovascular Strain & Cardiac Exertion
    peak_hr = hr_obs.groupby('Id').agg(
        peak_hr_max=('MaxHeartRate', 'max'),
        avg_hr_mean=('AvgHeartRate', 'mean'),
        cardiac_strain_hours=('MaxHeartRate', lambda x: (x >= 140).sum()),
        extreme_cardiac_hours=('MaxHeartRate', lambda x: (x >= 160).sum())
    ).reset_index().rename(columns={'Id': 'athlete_id'})
    
    hr_feats = nocturnal_hr.merge(peak_hr, on='athlete_id')
    hr_feats['heart_rate_reserve'] = hr_feats['peak_hr_max'] - hr_feats['resting_hr_mean']
    hr_feats['cardiac_exertion_ratio'] = hr_feats['peak_hr_max'] / (hr_feats['resting_hr_mean'] + 1e-5)
    
    # 7. Merge All Features
    print(">>> 7. Assembling Master Feature Matrix...")
    features = meta_df.merge(daily_feats, on='athlete_id', how='left')
    features = features.merge(sleep_feats, on='athlete_id', how='left')
    features = features.merge(sess_agg, on='athlete_id', how='left')
    features = features.merge(weight_agg, on='athlete_id', how='left')
    features = features.merge(hr_feats, on='athlete_id', how='left')
    
    features.fillna(0, inplace=True)
    
    # Encodings
    sport_dummies = pd.get_dummies(features['sport'], prefix='sport', drop_first=False)
    gender_dummies = pd.get_dummies(features['gender'], prefix='gender', drop_first=True)
    dominant_dummies = pd.get_dummies(features['dominant_side'], prefix='dominant', drop_first=True)
    position_dummies = pd.get_dummies(features['position'], prefix='pos', drop_first=True)
    
    features = pd.concat([features, sport_dummies, gender_dummies, dominant_dummies, position_dummies], axis=1)
    
    if is_train and os.path.exists(os.path.join(data_dir, 'Train Labels Dataset.csv')):
        labels_df = pd.read_csv(os.path.join(data_dir, 'Train Labels Dataset.csv'))
        features = features.merge(labels_df, on='athlete_id', how='left')
        
    print(f">>> Advanced Feature Matrix Complete! Shape: {features.shape}")
    return features

if __name__ == '__main__':
    df = extract_advanced_features()
    df.to_csv('processed_data/master_features_v2.csv', index=False)
    print("Saved processed_data/master_features_v2.csv successfully!")
