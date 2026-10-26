import pandas as pd
import numpy as np

def calculate_temporal_features(df):
    """
    Generates temporal features for each service independently.
    
    Formulas:
    - Rolling Mean: Avg(metric, window)
    - Rolling Slope: (Current - Mean(window)) / window_size (approximate)
    - Acceleration: diff(slope)
    """
    # Ensure chronological ordering
    df = df.sort_values(['service_name', 'timestamp']).reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    results = []
    for service, group in df.groupby('service_name'):
        group = group.sort_values('timestamp').copy()
        
        # To use time-based windows ('15T'), the index MUST be a DatetimeIndex
        group = group.set_index('timestamp')
        
        metrics = {
            'error_rate': ['15min', '30min', '60min'], # Fixed window strings
            'latency_p95': ['15min', '30min', '60min']
        }
        
        for col, windows in metrics.items():
            for win in windows:
                # Rolling Mean
                feat_name = f"{col}_roll_mean_{win}"
                group[feat_name] = group[col].rolling(window=win).mean()
                
                # Rolling Slope proxy
                feat_name_slope = f"{col}_slope_{win}"
                group[feat_name_slope] = group[col] - group[feat_name]

        # Error Rate Acceleration (1st derivative of slope)
        group['error_rate_acceleration'] = group['error_rate_slope_15min'].diff()
        
        # error_rate_change_10min
        # Use a 10min window offset for the change
        group['error_rate_change_10min'] = group['error_rate'].diff(periods=1) # Approximation
        
        # latency_trend
        group['latency_trend'] = group['latency_p95_slope_60min']
        
        # Lag features
        group['error_rate_lag_1'] = group['error_rate'].shift(1)
        
        results.append(group.reset_index())
        
    return pd.concat(results, ignore_index=True)
