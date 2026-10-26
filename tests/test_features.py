import pytest
import pandas as pd
import numpy as np
from src.features.temporal import calculate_temporal_features
from src.features.graph_features import calculate_graph_features
from src.features.context import calculate_context_features
import networkx as nx

@pytest.fixture
def sample_df():
    # Create a deterministic small dataset
    timestamps = pd.date_range('2026-01-01', periods=10, freq='15min')
    services = ['S1', 'S2']
    data = []
    for s in services:
        for t in timestamps:
            data.append({
                'timestamp': t,
                'service_name': s,
                'error_rate': np.random.random(),
                'latency_p95': np.random.random() * 100,
                'downstream_dependencies': 0
            })
    return pd.DataFrame(data)

def test_temporal_no_leakage():
    # Create a simple 2-row DF
    df_mini = pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-01-01 00:00', '2026-01-01 00:15']),
        'service_name': ['S1', 'S1'],
        'error_rate': [0.1, 0.9],
        'latency_p95': [10, 90]
    })
    res = calculate_temporal_features(df_mini)
    # Row 0 rolling mean 15min should be 0.1
    # If it were 0.5, it would be leakage.
    assert res.iloc[0]['error_rate_roll_mean_15min'] == 0.1

def test_service_separation(sample_df):
    processed = calculate_temporal_features(sample_df)
    # Ensure S1's rolling mean doesn't include S2's data
    s1_vals = sample_df[sample_df['service_name'] == 'S1']['error_rate'].values
    s1_processed = processed[processed['service_name'] == 'S1']['error_rate_roll_mean_15min'].values
    
    assert s1_processed[0] == s1_vals[0]

def test_graph_features():
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C')])
    df = pd.DataFrame({'service_name': ['A', 'B', 'C']})
    res = calculate_graph_features(df, G)
    assert res.loc[res['service_name'] == 'A', 'graph_out_degree'].values[0] == 1
    assert res.loc[res['service_name'] == 'C', 'graph_transitive_callers'].values[0] == 2

def test_context_features(sample_df):
    res = calculate_context_features(sample_df)
    assert 'hour_of_day' in res.columns
    assert 'is_weekend' in res.columns
