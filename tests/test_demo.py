import pytest
import os
import pandas as pd
import numpy as np
import joblib
from src.decision.alerts import AlertEngine
from src.decision.demo import (
    SEVERITY_ORDER,
    assign_ground_truth_risk,
    generate_synthetic_stress_data,
    evaluate_stress_predictions
)
from src.data.graph import load_dependency_graph

def test_risk_level_mapping_and_ordering():
    assert SEVERITY_ORDER['Low'] == 0
    assert SEVERITY_ORDER['Medium'] == 1
    assert SEVERITY_ORDER['High'] == 2
    assert SEVERITY_ORDER['Critical'] == 3
    assert SEVERITY_ORDER['Low'] < SEVERITY_ORDER['Medium'] < SEVERITY_ORDER['High'] < SEVERITY_ORDER['Critical']

def test_alert_score_scale_normalization():
    engine = AlertEngine(threshold=0.7, window_size=2)
    # Score on 0-100 scale: 80.0 should normalize to 0.80 and exceed 0.7
    res1 = engine.process_observation('S1', 80.0)
    assert res1['alert'] is False # Window size is 2
    res2 = engine.process_observation('S1', 75.0)
    assert res2['alert'] is True # Second consecutive interval >= 0.70

    # Low score on 0-100 scale: 15.0 should normalize to 0.15 and NOT alert
    res3 = engine.process_observation('S1', 15.0)
    assert res3['alert'] is False

def test_ground_truth_risk_rules():
    assert assign_ground_truth_risk({'error_rate': 0.1, 'latency_p95': 50}) == 'Low'
    assert assign_ground_truth_risk({'error_rate': 2.0, 'latency_p95': 100}) == 'Medium'
    assert assign_ground_truth_risk({'error_rate': 5.0, 'latency_p95': 200}) == 'High'
    assert assign_ground_truth_risk({'error_rate': 12.0, 'latency_p95': 300}) == 'Critical'
    assert assign_ground_truth_risk({'error_rate': 0.1, 'latency_p95': 950}) == 'Critical'

def test_scenario_isolation_and_schema_compatibility():
    raw_csv = 'data/raw/api_failure_cascade_dataset.csv'
    proc_parquet = 'data/processed/features.parquet'
    
    csv_mtime_before = os.path.getmtime(raw_csv)
    parquet_mtime_before = os.path.getmtime(proc_parquet)

    df_base = pd.read_parquet(proc_parquet)
    graph = load_dependency_graph(raw_csv)
    model = joblib.load('models/cascade_predictor.joblib')
    encoder = joblib.load('models/label_encoder.joblib')

    stress_df, metrics = generate_synthetic_stress_data(
        df_base=df_base,
        graph=graph,
        model=model,
        encoder=encoder,
        stressed_services=['AuthService', 'PaymentService'],
        intensity='Severe',
        stress_steps=3
    )

    # 1. Verify schema compatibility: all model features must be present
    for feat in model.feature_names_in_:
        assert feat in stress_df.columns
        assert stress_df[feat].isna().sum() == 0

    # 2. Verify model predictions exist and are non-empty
    assert 'pred_cascade_level' in stress_df.columns
    assert 'risk_probability' in stress_df.columns
    assert len(stress_df) > 0
    assert 'macro_f1' in metrics
    assert 'report' in metrics
    assert 'confusion_matrix' in metrics

    # 3. Verify scenario isolation: original files were NOT touched or modified
    assert os.path.getmtime(raw_csv) == csv_mtime_before
    assert os.path.getmtime(proc_parquet) == parquet_mtime_before
