import streamlit as st
import pandas as pd
import joblib
import os
import json
from src.data.graph import load_dependency_graph

@st.cache_data
def load_full_dataset(custom_path=None):
    """Loads processed features dataset."""
    path = custom_path or 'data/processed/features.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_resource
def load_model_artifacts():
    """Loads trained Random Forest model and label encoder."""
    model_path = 'models/cascade_predictor.joblib'
    encoder_path = 'models/label_encoder.joblib'
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None
    return joblib.load(model_path), joblib.load(encoder_path)

@st.cache_resource
def load_graph(custom_path=None):
    """Loads microservice dependency graph."""
    path = custom_path or 'data/raw/api_failure_cascade_dataset.csv'
    if not os.path.exists(path):
        return None
    return load_dependency_graph(path)

@st.cache_data
def load_evaluation_report(custom_path=None):
    """Loads model evaluation report metrics."""
    path = custom_path or 'reports/evaluation_report.json'
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def clear_all_caches():
    """Clears Streamlit data and resource caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
