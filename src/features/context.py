import pandas as pd

def calculate_context_features(df):
    """
    Adds temporal context features.
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['is_peak_hour'] = df['hour_of_day'].apply(lambda x: 1 if (8 <= x <= 11 or 14 <= x <= 17) else 0)
    df['is_weekend'] = df['timestamp'].dt.day_of_week.apply(lambda x: 1 if x >= 5 else 0)
    
    # Historical average impact (preventing leakage)
    # We calculate the average impact of the service's failures PRIOR to the current timestamp
    # This requires group-by service and a expanding mean of the target
    # BUT: the prompt says to exclude target-derived fields from PREDICTORS.
    # If 'historical_avg_cascade_impact' is a predictor, it must be calculated from 
    # a DIFFERENT target or a known historical baseline.
    # In the raw dataset, there is a column 'historical_avg_cascade_impact'. 
    # We'll keep it if it's provided as a static attribute, otherwise we don't derive it.
    
    return df
