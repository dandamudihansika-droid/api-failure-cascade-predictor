"""
Interactive Demo Engine for API Failure Cascade Predictor.
Provides scenario generation, synthetic stress testing, feature recomputation,
and evaluation without modifying original dataset artifacts.
"""
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.features.temporal import calculate_temporal_features
from src.features.graph_features import calculate_graph_features
from src.features.context import calculate_context_features

SEVERITY_ORDER = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
SEVERITY_COLORS = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#fd7e14', 'Critical': '#dc3545'}

def assign_ground_truth_risk(row):
    """
    Documented ground-truth cascade level definition based on operational telemetry:
    - Critical: error_rate >= 8.0% OR latency_p95 >= 900ms
    - High: error_rate >= 4.0% OR latency_p95 >= 500ms
    - Medium: error_rate >= 1.5% OR latency_p95 >= 300ms
    - Low: Otherwise
    """
    err = float(row.get('error_rate', 0))
    lat = float(row.get('latency_p95', 0))
    
    if err >= 8.0 or lat >= 900.0:
        return 'Critical'
    elif err >= 4.0 or lat >= 500.0:
        return 'High'
    elif err >= 1.5 or lat >= 300.0:
        return 'Medium'
    return 'Low'

def generate_synthetic_stress_data(
    df_base, 
    graph, 
    model, 
    encoder, 
    stressed_services=None, 
    intensity="Severe", 
    stress_steps=4
):
    """
    Generates a separate, temporary synthetic stress dataset.
    Recomputes temporal, graph, and contextual features through the pipeline,
    and runs the trained model. Never overwrites raw or processed data files.
    """
    if stressed_services is None:
        stressed_services = ['PaymentService', 'AuthService']

    intensity_configs = {
        "Moderate": {
            "error_rate": 4.5,
            "latency_p95": 520.0,
            "latency_p99": 1050.0,
            "timeout_rate": 4.0,
            "cpu_usage": 75.0,
            "memory_usage": 70.0
        },
        "Severe": {
            "error_rate": 14.5,
            "latency_p95": 1150.0,
            "latency_p99": 2300.0,
            "timeout_rate": 8.5,
            "cpu_usage": 91.0,
            "memory_usage": 85.0
        },
        "Extreme": {
            "error_rate": 26.0,
            "latency_p95": 1800.0,
            "latency_p99": 3600.0,
            "timeout_rate": 18.0,
            "cpu_usage": 98.0,
            "memory_usage": 94.0
        }
    }
    cfg = intensity_configs.get(intensity, intensity_configs["Severe"])

    # Extract base columns for recomputation
    base_cols = [
        'timestamp', 'service_name', 'error_rate', 'latency_p95', 'latency_p99',
        'request_volume', 'timeout_rate', 'cpu_usage', 'memory_usage',
        'downstream_dependencies', 'critical_downstream_count', 'betweenness_centrality',
        'is_peak_hour', 'is_weekend', 'historical_avg_cascade_impact'
    ]
    available_base = [c for c in base_cols if c in df_base.columns]
    
    # Use recent history (e.g. last 10 intervals per service)
    recent_df = df_base.sort_values('timestamp').groupby('service_name').tail(10).copy()
    stress_raw = recent_df[available_base].copy().reset_index(drop=True)

    # Inject synthetic stress for designated services on the final steps
    for svc in stressed_services:
        svc_idx = stress_raw[stress_raw['service_name'] == svc].tail(stress_steps).index
        for param, val in cfg.items():
            if param in stress_raw.columns:
                stress_raw.loc[svc_idx, param] = val

    # Recompute feature pipeline
    temp = calculate_temporal_features(stress_raw)
    temp = calculate_graph_features(temp, graph)
    temp = calculate_context_features(temp)

    # Assign synthetic ground truth
    temp['ground_truth_level'] = temp.apply(assign_ground_truth_risk, axis=1)

    # Run trained model predictions
    features = list(model.feature_names_in_)
    for f in features:
        if f not in temp.columns:
            temp[f] = 0.0
    temp[features] = temp[features].fillna(0)

    preds = model.predict(temp[features])
    probs = model.predict_proba(temp[features])
    temp['pred_cascade_level'] = encoder.inverse_transform(preds)

    # Class probabilities
    for i, cls_name in enumerate(encoder.classes_):
        temp[f'prob_{cls_name}'] = probs[:, i]

    # Overall probability of cascade risk (Non-Low probability)
    low_idx = list(encoder.classes_).index('Low') if 'Low' in encoder.classes_ else -1
    if low_idx >= 0:
        temp['risk_probability'] = 1.0 - probs[:, low_idx]
    else:
        temp['risk_probability'] = np.max(probs, axis=1)

    # Evaluate predictions against ground-truth
    eval_metrics = evaluate_stress_predictions(temp['ground_truth_level'], temp['pred_cascade_level'])

    return temp, eval_metrics

def evaluate_stress_predictions(y_true, y_pred):
    """
    Computes confusion matrix, per-class report, and macro-F1 for synthetic test.
    """
    classes = ['Critical', 'High', 'Low', 'Medium']
    # Filter classes present in either true or pred
    present_classes = sorted(list(set(y_true).union(set(y_pred))))

    report = classification_report(
        y_true, 
        y_pred, 
        labels=present_classes, 
        output_dict=True, 
        zero_division=0
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=present_classes)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'macro_f1': macro_f1,
        'report': report,
        'confusion_matrix': conf_mat.tolist(),
        'classes': present_classes
    }
