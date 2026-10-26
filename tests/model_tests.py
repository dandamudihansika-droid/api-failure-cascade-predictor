import pytest
import pandas as pd
import numpy as np
from src.models.train import load_and_prepare_data, chronological_split

def test_chronological_split():
    # Create dummy data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=100, freq='min'),
        'service_name': ['S1'] * 100,
        'feature1': np.random.rand(100),
        'cascade_level': np.random.randint(0, 2, 100)
    })
    
    train, val, test = chronological_split(df, ['feature1'], 'cascade_level')
    
    assert train.index[0] == 0
    assert train.index[-1] == 69
    assert val.index[0] == 70
    assert val.index[-1] == 84
    assert test.index[0] == 85
    assert test.index[-1] == 99

def test_target_exclusion():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=10, freq='min'),
        'service_name': ['S1'] * 10,
        'feature1': np.random.rand(10),
        'cascade_risk_score': np.random.rand(10),
        'cascade_level': np.random.randint(0, 2, 10),
        'blast_radius': np.random.randint(0, 10, 10)
    })
    temp_file = 'test_temp.parquet'
    try:
        df.to_parquet(temp_file)
        _, features, _ = load_and_prepare_data(temp_file)
        for col in ['cascade_risk_score', 'cascade_level', 'blast_radius']:
            assert col not in features
    finally:
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
