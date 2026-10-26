"""
Unit tests validating Priority 1, 2, and 3 audit fixes:
- Risk calculation correctness and model feature compatibility
- Unit validation: error rate as percentage points vs ratios
- Alert counting consistency and chronological ordering
- Graph node and edge counts, and visualization-simulator synchronization
"""
import pytest
import pandas as pd
import numpy as np
import joblib
from src.decision.alerts import AlertEngine
from src.decision.simulator import CascadeSimulator
from src.data.graph import load_dependency_graph
from app.components.data_loader import load_full_dataset, load_model_artifacts, load_graph

def test_model_feature_compatibility_and_predictions():
    """Confirms model feature names, count, order, and label mapping match training."""
    model, encoder = load_model_artifacts()
    df = load_full_dataset()
    
    assert model is not None
    assert encoder is not None
    assert len(model.feature_names_in_) == 35
    assert list(encoder.classes_) == ['Critical', 'High', 'Low', 'Medium']
    
    # Verify model prediction on latest rows matches expected features
    latest = df.sort_values('timestamp').groupby('service_name').tail(1)
    features = list(model.feature_names_in_)
    
    preds = model.predict(latest[features])
    probs = model.predict_proba(latest[features])
    
    assert len(preds) == len(latest)
    assert probs.shape == (len(latest), 4)
    # Sum of probabilities per row must equal 1.0
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

def test_error_rate_units_and_magnitude():
    """Confirms error_rate is percentage points (0.0% to 32.0%), not fractional ratios."""
    df_raw = pd.read_csv('data/raw/api_failure_cascade_dataset.csv')
    
    # If error_rate were a ratio [0, 1], the max could not be > 1.0
    assert df_raw['error_rate'].max() > 1.0
    assert df_raw['error_rate'].max() <= 100.0
    
    # 75th percentile is under 1.0%, indicating healthy normal baseline
    assert df_raw['error_rate'].quantile(0.75) < 1.0

def test_alert_consistency_and_chronological_ordering():
    """Confirms AlertEngine requires 2 consecutive intervals and evaluates in time order."""
    engine = AlertEngine(threshold=0.7, window_size=2)
    
    # Observation 1: high risk (0.85) -> No alert yet (window size is 2)
    ev1 = engine.process_observation('TestSvc', 0.85)
    assert ev1['alert'] is False
    
    # Observation 2: high risk (0.92) -> Active sustained alert!
    ev2 = engine.process_observation('TestSvc', 0.92)
    assert ev2['alert'] is True
    assert ev2['severity'] == 'Critical'
    
    # Observation 3: low risk (0.10) -> Alert resets
    ev3 = engine.process_observation('TestSvc', 0.10)
    assert ev3['alert'] is False

def test_graph_node_and_edge_counts_and_simulator_sync():
    """Confirms graph has 14 nodes, 4 internal edges, Caller->Callee direction, and simulator sync."""
    raw_path = 'data/raw/api_failure_cascade_dataset.csv'
    graph = load_dependency_graph(raw_path)
    
    assert len(graph.nodes()) == 14
    assert len(graph.edges()) == 4
    
    # Edge directions: Caller -> Callee
    assert graph.has_edge('PaymentService', 'AuthService')
    assert graph.has_edge('PaymentService', 'InventoryService')
    assert graph.has_edge('SearchService', 'InventoryService')
    assert graph.has_edge('SearchService', 'AnalyticsService')
    
    # Verify CascadeSimulator uses the identical graph
    sim = CascadeSimulator(graph_path=raw_path)
    assert set(sim.graph.nodes()) == set(graph.nodes())
    assert set(sim.graph.edges()) == set(graph.edges())
