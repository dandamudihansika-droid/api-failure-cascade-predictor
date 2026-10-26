import pytest
import numpy as np
from src.decision.simulator import CascadeSimulator
from src.decision.mitigation import MitigationRecommender
from src.decision.alerts import AlertEngine

def test_simulator_determinism():
    sim = CascadeSimulator()
    # Test that same input gives same output
    res1 = sim.whatif('AuthService', 'High')
    res2 = sim.whatif('AuthService', 'High')
    assert res1 == res2

def test_simulator_severity_bounds():
    sim = CascadeSimulator()
    # Invalid severity
    res = sim.whatif('AuthService', 'Extreme')
    assert "error" in res

def test_mitigation_rules():
    rec = MitigationRecommender()
    
    # Test Retry Storm rule
    metrics = {'error_rate': 0.4, 'latency_p95': 100, 'cpu_usage': 20}
    suggestions = rec.recommend('ServiceA', metrics, 'High')
    # Should recommend disabling retries
    assert any('Disable Automatic Retries' in s['action'] for s in suggestions)
    
    # Test Timeout rule
    metrics_lat = {'error_rate': 0.01, 'latency_p95': 1000, 'cpu_usage': 20}
    suggestions_lat = rec.recommend('ServiceA', metrics_lat, 'Low')
    assert any('Reduce Request Timeout' in s['action'] for s in suggestions_lat)

def test_alert_consecutive_intervals():
    engine = AlertEngine(threshold=0.5, window_size=2)
    
    # Interval 1: High risk
    res1 = engine.process_observation('S1', 0.6)
    assert res1['alert'] is False
    
    # Interval 2: High risk -> Alert!
    res2 = engine.process_observation('S1', 0.7)
    assert res2['alert'] is True
    
    # Interval 3: Low risk -> Reset
    res3 = engine.process_observation('S1', 0.2)
    assert res3['alert'] is False

def test_alert_missing_data():
    engine = AlertEngine(threshold=0.5, window_size=2)
    # Only one observation for S2
    res = engine.process_observation('S2', 0.9)
    assert res['alert'] is False
