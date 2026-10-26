import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from app.components.styles import (
    THEME, 
    inject_custom_css, 
    get_plotly_layout, 
    render_kpi_card, 
    render_badge,
    render_sre_table_html
)
from app.components.data_loader import (
    load_full_dataset, 
    load_model_artifacts, 
    load_graph, 
    load_evaluation_report,
    clear_all_caches
)
from src.decision.simulator import CascadeSimulator
from src.decision.mitigation import MitigationRecommender
from src.decision.alerts import AlertEngine
from src.decision.demo import (
    SEVERITY_ORDER, 
    SEVERITY_COLORS, 
    generate_synthetic_stress_data,
    assign_ground_truth_risk
)

# -----------------------------------------------------------------------------
# APP CONFIG & INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NEXUS SRE | API Cascade Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Session State Initialization
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "scenario" not in st.session_state:
    st.session_state.scenario = "A. Live & Historical Telemetry"
if "telemetry_window" not in st.session_state:
    st.session_state.telemetry_window = "Live Baseline (Healthy: 2026-10-25 23:55)"
if "stress_services" not in st.session_state:
    st.session_state.stress_services = ["PaymentService", "AuthService"]
if "stress_intensity" not in st.session_state:
    st.session_state.stress_intensity = "Severe"
if "stress_steps" not in st.session_state:
    st.session_state.stress_steps = 4
if "sim_service" not in st.session_state:
    st.session_state.sim_service = "PaymentService"
if "sim_severity" not in st.session_state:
    st.session_state.sim_severity = "High"
if "selected_project" not in st.session_state:
    st.session_state.selected_project = "Production E-Commerce Cluster"

# Load Core Artifacts
df_raw = load_full_dataset()
model, encoder = load_model_artifacts()
graph = load_graph()
report = load_evaluation_report()

leakage_cols = ['cascade_risk_score', 'cascade_level', 'blast_radius']
id_cols = ['timestamp', 'service_name']
model_features = list(model.feature_names_in_) if model is not None else [
    c for c in df_raw.columns if c not in leakage_cols and c not in id_cols
]

# -----------------------------------------------------------------------------
# PERSISTENT SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. Product Logo and Identity
    st.markdown(
        f"""
        <div style='display: flex; align-items: center; gap: 10px; padding: 6px 0 16px 0;'>
            <div style='background: linear-gradient(135deg, {THEME["cerulean"]}, {THEME["bright_blue"]}); 
                        width: 36px; height: 36px; border-radius: 8px; display: flex; 
                        align-items: center; justify-content: center; font-size: 1.25rem; font-weight: 800; color: #05070B;
                        box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);'>
                ⚡
            </div>
            <div>
                <div style='font-size: 1.05rem; font-weight: 800; letter-spacing: 0.05em; color: {THEME["text_main"]};'>NEXUS RADAR</div>
                <div style='font-size: 0.70rem; color: {THEME["text_muted"]}; text-transform: uppercase; letter-spacing: 0.08em;'>SRE Observability</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Project Selector
    st.caption("Active Cluster Mesh")
    selected_proj = st.selectbox(
        "Cluster",
        [
            "Production E-Commerce Cluster",
            "Payment & Checkout Core Mesh",
            "Ingress & API Gateway Zone"
        ],
        index=["Production E-Commerce Cluster", "Payment & Checkout Core Mesh", "Ingress & API Gateway Zone"].index(st.session_state.selected_project),
        label_visibility="collapsed"
    )
    st.session_state.selected_project = selected_proj

    # 3. Data-Source Status Pill
    status_pill_color = THEME["low"]
    status_text = "LIVE BASELINE"
    status_desc = "Healthy Quiescent Telemetry"
    
    if st.session_state.scenario == "A. Live & Historical Telemetry":
        if "Critical" in st.session_state.telemetry_window or "High" in st.session_state.telemetry_window:
            status_pill_color = THEME["critical"]
            status_text = "INCIDENT REPLAY"
            status_desc = st.session_state.telemetry_window.split(":")[0]
        else:
            status_pill_color = THEME["low"]
            status_text = "LIVE BASELINE"
            status_desc = "Healthy Quiescent Telemetry"
    elif st.session_state.scenario == "B. Simulated Incident":
        status_pill_color = THEME["high"]
        status_text = "SIMULATED INCIDENT"
        status_desc = f"{st.session_state.sim_service} ({st.session_state.sim_severity})"
    elif st.session_state.scenario == "C. Synthetic Stress Test":
        status_pill_color = THEME["critical"]
        status_text = "STRESS INJECTED"
        status_desc = f"{st.session_state.stress_intensity} Dynamic Load"

    st.markdown(
        f"""
        <div style='background: rgba(14, 24, 38, 0.85); border: 1px solid #162436; border-radius: 8px; padding: 10px 14px; margin: 4px 0 14px 0;'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <span style='font-size: 0.70rem; text-transform: uppercase; font-weight: 700; color: {THEME["text_muted"]}; letter-spacing: 0.06em;'>Data Status</span>
                <span style='display: inline-flex; align-items: center; gap: 5px; font-size: 0.70rem; font-weight: 700; color: {status_pill_color}; font-family: monospace;'>
                    <span style='width: 7px; height: 7px; border-radius: 50%; background-color: {status_pill_color}; box-shadow: 0 0 8px {status_pill_color};'></span>
                    {status_text}
                </span>
            </div>
            <div style='font-size: 0.75rem; color: {THEME["text_muted"]}; margin-top: 4px;'>{status_desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 1px; background: rgba(56, 189, 248, 0.1); margin: 6px 0 12px 0;'></div>", unsafe_allow_html=True)
    st.caption("Navigation")

    # 4. Sidebar Navigation
    nav_options = [
        "Overview",
        "Service Topology",
        "Cascade Simulator",
        "Service Explorer",
        "Model Performance",
        "Settings & Data Import"
    ]
    
    current_idx = nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0
    selected_nav = st.radio(
        "nav_selector",
        nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )
    if selected_nav != st.session_state.page:
        st.session_state.page = selected_nav
        st.rerun()

    st.markdown("<div style='height: 1px; background: rgba(56, 189, 248, 0.1); margin: 12px 0 12px 0;'></div>", unsafe_allow_html=True)
    
    # 5. Interactive Demo Mode Drawer
    with st.expander("🎮 Demo Mode & Operational Scenarios", expanded=True):
        scenario_choice = st.selectbox(
            "Operational Scenario",
            [
                "A. Live & Historical Telemetry",
                "B. Simulated Incident",
                "C. Synthetic Stress Test"
            ],
            index=["A. Live & Historical Telemetry", "B. Simulated Incident", "C. Synthetic Stress Test"].index(st.session_state.scenario)
        )
        if scenario_choice != st.session_state.scenario:
            st.session_state.scenario = scenario_choice
            st.rerun()

        if st.session_state.scenario == "A. Live & Historical Telemetry":
            window_choices = [
                "Live Baseline (Healthy: 2026-10-25 23:55)",
                "Historical Outage: UserService Critical (2026-10-23 02:35)",
                "Historical Outage: UserService High (2026-10-18 07:20)",
                "Historical Outage: PaymentService High (2026-09-26 08:00)"
            ]
            win_choice = st.selectbox(
                "Telemetry Snapshot Window",
                window_choices,
                index=window_choices.index(st.session_state.telemetry_window) if st.session_state.telemetry_window in window_choices else 0
            )
            if win_choice != st.session_state.telemetry_window:
                st.session_state.telemetry_window = win_choice
                st.rerun()

        elif st.session_state.scenario == "B. Simulated Incident":
            if graph is not None:
                sim_s = st.selectbox("Root Service", sorted(list(graph.nodes())), index=sorted(list(graph.nodes())).index(st.session_state.sim_service) if st.session_state.sim_service in graph.nodes() else 0)
                st.session_state.sim_service = sim_s
            sim_sev = st.selectbox("Simulated Severity", ["Low", "Medium", "High", "Critical"], index=["Low", "Medium", "High", "Critical"].index(st.session_state.sim_severity))
            st.session_state.sim_severity = sim_sev

        elif st.session_state.scenario == "C. Synthetic Stress Test":
            all_s = sorted(df_raw["service_name"].unique()) if not df_raw.empty else ["PaymentService", "AuthService"]
            st.session_state.stress_services = st.multiselect("Stressed Services", all_s, default=st.session_state.stress_services)
            st.session_state.stress_intensity = st.selectbox("Intensity", ["Moderate", "Severe", "Extreme"], index=["Moderate", "Severe", "Extreme"].index(st.session_state.stress_intensity))
            st.session_state.stress_steps = st.slider("Intervals", min_value=2, max_value=6, value=st.session_state.stress_steps)

        if st.button("Reset Demo State", use_container_width=True):
            st.session_state.scenario = "A. Live & Historical Telemetry"
            st.session_state.telemetry_window = "Live Baseline (Healthy: 2026-10-25 23:55)"
            st.session_state.stress_services = ["PaymentService", "AuthService"]
            st.session_state.stress_intensity = "Severe"
            st.session_state.stress_steps = 4
            st.session_state.sim_service = "PaymentService"
            st.session_state.sim_severity = "High"
            st.rerun()

# -----------------------------------------------------------------------------
# TELEMETRY & INFERENCE RESOLUTION (PRIORITY 1 & 2 FIXES)
# -----------------------------------------------------------------------------
def resolve_current_telemetry():
    """
    Resolves active telemetry, model predictions, alert states, and threshold flags.
    Ensures model input features, order, and label mappings match training exactly.
    """
    if df_raw.empty:
        return pd.DataFrame(), "No Data", {}, []

    df_active = df_raw.copy()
    if st.session_state.selected_project == "Payment & Checkout Core Mesh":
        core_services = ["PaymentService", "AuthService", "InventoryService", "OrderService", "CartService"]
        df_active = df_active[df_active["service_name"].isin(core_services)].copy()
    elif st.session_state.selected_project == "Ingress & API Gateway Zone":
        ingress_services = ["SearchService", "ProductService", "RecommendationService", "ReviewService", "UserService"]
        df_active = df_active[df_active["service_name"].isin(ingress_services)].copy()

    if st.session_state.scenario == "A. Live & Historical Telemetry":
        if "UserService Critical (2026-10-23 02:35)" in st.session_state.telemetry_window:
            target_ts = pd.to_datetime("2026-10-23 02:35:00")
            snapshot = df_active[df_active["timestamp"] <= target_ts].sort_values("timestamp").groupby("service_name").tail(1).copy()
            desc = "Historical Telemetry: UserService Critical Incident (2026-10-23 02:35)"
        elif "UserService High (2026-10-18 07:20)" in st.session_state.telemetry_window:
            target_ts = pd.to_datetime("2026-10-18 07:20:00")
            snapshot = df_active[df_active["timestamp"] <= target_ts].sort_values("timestamp").groupby("service_name").tail(1).copy()
            desc = "Historical Telemetry: UserService High Incident (2026-10-18 07:20)"
        elif "PaymentService High (2026-09-26 08:00)" in st.session_state.telemetry_window:
            target_ts = pd.to_datetime("2026-09-26 08:00:00")
            snapshot = df_active[df_active["timestamp"] <= target_ts].sort_values("timestamp").groupby("service_name").tail(1).copy()
            desc = "Historical Telemetry: PaymentService High Load (2026-09-26 08:00)"
        else:
            snapshot = df_active.sort_values("timestamp").groupby("service_name").tail(1).copy()
            desc = "Live Baseline Telemetry (Normal Quiescent Baseline: 2026-10-25 23:55)"

        # Priority 1: Exact model features matching training
        X_inputs = snapshot[model_features].fillna(0)
        preds = model.predict(X_inputs)
        probs = model.predict_proba(X_inputs)

        snapshot["pred_cascade_level"] = encoder.inverse_transform(preds)
        for i, c in enumerate(encoder.classes_):
            snapshot[f"prob_{c}"] = probs[:, i]
        low_idx = list(encoder.classes_).index("Low") if "Low" in encoder.classes_ else -1
        # Model failure probability: P(Non-Low) = P(Critical) + P(High) + P(Medium)
        snapshot["risk_probability"] = (1.0 - probs[:, low_idx]) if low_idx >= 0 else np.max(probs, axis=1)
        # Model confidence in the predicted class
        snapshot["pred_confidence"] = np.max(probs, axis=1)

        # Priority 1: Continuous Telemetry Risk Score from dataset (0-100 scale)
        if "cascade_risk_score" in snapshot.columns:
            snapshot["telemetry_risk_score"] = snapshot["cascade_risk_score"]
        else:
            snapshot["telemetry_risk_score"] = snapshot["risk_probability"] * 100.0

        # Priority 1: SRE Operational Threshold Violations (independent of model)
        def check_violations(row):
            violations = []
            if float(row.get("error_rate", 0)) > 2.0:
                violations.append(f"Err > 2% ({row['error_rate']:.2f}%)")
            if float(row.get("latency_p95", 0)) > 500.0:
                violations.append(f"Lat > 500ms ({row['latency_p95']:.1f}ms)")
            if float(row.get("cpu_usage", 0)) > 80.0:
                violations.append(f"CPU > 80% ({row['cpu_usage']:.1f}%)")
            return ", ".join(violations) if violations else "Nominal"

        snapshot["threshold_flags"] = snapshot.apply(check_violations, axis=1)

        # Priority 2: Consistent AlertEngine Evaluation
        # Feeds the chronological history for each service up to the snapshot timestamp
        active_alerts = []
        for svc, group in df_active[df_active["timestamp"] <= snapshot["timestamp"].max()].groupby("service_name"):
            eng = AlertEngine(threshold=0.7, window_size=2)
            # Evaluate the last 3 chronological intervals for sustained high risk
            recent_scores = group.sort_values("timestamp").tail(3)
            # Use normalized risk score (score / 100.0) or model failure probability
            last_alert = None
            for _, r in recent_scores.iterrows():
                # Score on 0.0-1.0 scale
                s_val = float(r.get("cascade_risk_score", 0)) / 100.0
                res = eng.process_observation(svc, s_val)
                if res.get("alert"):
                    last_alert = res
            if last_alert and svc in snapshot["service_name"].values:
                active_alerts.append(last_alert)

        return snapshot, desc, {}, active_alerts

    elif st.session_state.scenario == "B. Simulated Incident":
        snapshot = df_active.sort_values("timestamp").groupby("service_name").tail(1).copy()
        X_inputs = snapshot[model_features].fillna(0)
        preds = model.predict(X_inputs)
        probs = model.predict_proba(X_inputs)

        snapshot["pred_cascade_level"] = encoder.inverse_transform(preds)
        for i, c in enumerate(encoder.classes_):
            snapshot[f"prob_{c}"] = probs[:, i]
        low_idx = list(encoder.classes_).index("Low") if "Low" in encoder.classes_ else -1
        snapshot["risk_probability"] = (1.0 - probs[:, low_idx]) if low_idx >= 0 else np.max(probs, axis=1)
        snapshot["pred_confidence"] = np.max(probs, axis=1)
        snapshot["telemetry_risk_score"] = snapshot.get("cascade_risk_score", snapshot["risk_probability"] * 100.0)
        snapshot["threshold_flags"] = "Nominal"

        sim = CascadeSimulator()
        sim_res = sim.whatif(st.session_state.sim_service, st.session_state.sim_severity)

        # Apply simulated failure and propagation to the active snapshot metrics
        root_s = st.session_state.sim_service
        root_sev = st.session_state.sim_severity
        sev_risk_scores = {"Critical": 92.5, "High": 78.0, "Medium": 54.0, "Low": 8.0}
        sev_probs = {"Critical": 0.95, "High": 0.82, "Medium": 0.55, "Low": 0.05}

        # Update root service
        root_mask = snapshot["service_name"] == root_s
        if root_mask.any():
            snapshot.loc[root_mask, "pred_cascade_level"] = root_sev
            snapshot.loc[root_mask, "telemetry_risk_score"] = sev_risk_scores.get(root_sev, 75.0)
            snapshot.loc[root_mask, "risk_probability"] = sev_probs.get(root_sev, 0.80)
            snapshot.loc[root_mask, "threshold_flags"] = f"Simulated Injected Failure: {root_sev}"

        # Update impacted downstream/upstream services according to propagation
        active_alerts = []
        if root_sev in ["Critical", "High"]:
            active_alerts.append({
                "service": root_s,
                "severity": root_sev,
                "message": f"Simulated root failure injection active on {root_s} ({root_sev})"
            })

        for imp in sim_res.get("impacted_services", []):
            imp_svc = imp.get("service")
            imp_prob = imp.get("est_failure_prob", 0.0)
            imp_mask = snapshot["service_name"] == imp_svc
            if imp_mask.any() and imp_svc != root_s:
                if imp_prob >= 0.70:
                    imp_lvl = "Critical" if root_sev == "Critical" else "High"
                elif imp_prob >= 0.40:
                    imp_lvl = "Medium"
                else:
                    imp_lvl = "Low"
                snapshot.loc[imp_mask, "pred_cascade_level"] = imp_lvl
                snapshot.loc[imp_mask, "telemetry_risk_score"] = imp_prob * 100.0
                snapshot.loc[imp_mask, "risk_probability"] = imp_prob
                snapshot.loc[imp_mask, "threshold_flags"] = f"Cascade Impact (Hop {imp.get('distance', 1)})"
                if imp_lvl in ["Critical", "High"]:
                    active_alerts.append({
                        "service": imp_svc,
                        "severity": imp_lvl,
                        "message": f"Simulated cascade propagation from {root_s} ({imp_lvl})"
                    })

        return snapshot, f"Simulated Incident ({root_s} - {root_sev})", sim_res, active_alerts

    elif st.session_state.scenario == "C. Synthetic Stress Test":
        stress_df, eval_metrics = generate_synthetic_stress_data(
            df_base=df_active,
            graph=graph,
            model=model,
            encoder=encoder,
            stressed_services=st.session_state.stress_services,
            intensity=st.session_state.stress_intensity,
            stress_steps=st.session_state.stress_steps
        )
        snapshot = stress_df.sort_values("timestamp").groupby("service_name").tail(1).copy()
        snapshot["pred_confidence"] = snapshot[[f"prob_{c}" for c in encoder.classes_]].max(axis=1)
        snapshot["telemetry_risk_score"] = snapshot["risk_probability"] * 100.0

        def check_stress_violations(row):
            violations = []
            if float(row.get("error_rate", 0)) > 2.0:
                violations.append(f"Err > 2% ({row['error_rate']:.2f}%)")
            if float(row.get("latency_p95", 0)) > 500.0:
                violations.append(f"Lat > 500ms ({row['latency_p95']:.1f}ms)")
            return ", ".join(violations) if violations else "Nominal"

        snapshot["threshold_flags"] = snapshot.apply(check_stress_violations, axis=1)

        # Alerts for stressed services
        active_alerts = []
        for svc in st.session_state.stress_services:
            svc_rows = stress_df[stress_df["service_name"] == svc]
            if len(svc_rows) >= 2:
                eng = AlertEngine(threshold=0.7, window_size=2)
                for score in svc_rows["risk_probability"].tail(3):
                    res = eng.process_observation(svc, float(score))
                    if res.get("alert"):
                        active_alerts.append(res)
                        break

        return snapshot, f"Synthetic Stress Test ({st.session_state.stress_intensity} Dynamic Load)", eval_metrics, active_alerts

active_snapshot, scenario_title, scenario_extra, active_alerts = resolve_current_telemetry()

# -----------------------------------------------------------------------------
# TOP COMPACT HEADER & OPERATIONAL MODE SELECTOR
# -----------------------------------------------------------------------------
mode_label_map = {
    "Live Baseline (Healthy: 2026-10-25 23:55)": "🟢 Normal Operation (Nominal Baseline)",
    "Historical Outage: UserService Critical (2026-10-23 02:35)": "🔴 Historical Incident Replay (UserService Critical)",
    "Historical Outage: PaymentService High (2026-09-26 08:00)": "⚠️ High Load Replay (PaymentService High)"
}

header_html = f"""
<div class='app-header' style='display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; padding: 14px 20px; background: {THEME["panel_bg"]}; border: 1px solid {THEME["panel_border"]}; border-radius: 10px; margin-bottom: 16px;'>
    <div style='display: flex; align-items: center; gap: 16px;'>
        <div>
            <div style='font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: {THEME["text_muted"]}; font-weight: 700;'>Cluster Mesh</div>
            <div style='font-size: 1.15rem; font-weight: 800; color: {THEME["text_main"]};'>{st.session_state.selected_project}</div>
        </div>
        <div style='height: 28px; width: 1px; background: rgba(56, 189, 248, 0.2);'></div>
        <div style='display: flex; gap: 8px;'>
            <span class='badge badge-info'>REGION: US-EAST-1</span>
            <span class='badge' style='background: rgba(14, 24, 38, 0.9); border: 1px solid rgba(56, 189, 248, 0.35); color: {THEME["bright_blue"]};'>
                {scenario_title}
            </span>
        </div>
    </div>
    <div>
        <span style='font-size: 0.72rem; color: {THEME["text_muted"]}; font-family: monospace;'>Telemetry Stream:</span>
        <span style='font-size: 0.75rem; font-weight: 700; color: {THEME["bright_blue"]}; font-family: monospace; margin-left: 6px;'>
            14/14 Microservices Active
        </span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 1: OVERVIEW (AUDITED SRE CONTROL PLANE)
# -----------------------------------------------------------------------------
if st.session_state.page == "Overview":
    # Prominent Operational Snapshot & Mode Selector (Directly on Page)
    st.markdown("<div style='margin-bottom: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: " + THEME["text_muted"] + ";'>🕹️ Operational Telemetry Mode & Demonstration</div>", unsafe_allow_html=True)
    
    mode_options = [
        "🟢 Normal Operation",
        "🔴 Historical Incident Replay",
        "⚠️ High Load Replay",
        "🧪 Custom What-If Simulator"
    ]
    
    current_mode_idx = 0
    if st.session_state.scenario == "A. Live & Historical Telemetry":
        if "Critical" in st.session_state.telemetry_window:
            current_mode_idx = 1
        elif "PaymentService" in st.session_state.telemetry_window:
            current_mode_idx = 2
        else:
            current_mode_idx = 0
    elif st.session_state.scenario == "B. Simulated Incident":
        current_mode_idx = 3

    chosen_mode = st.radio(
        "Operational Mode",
        mode_options,
        index=current_mode_idx,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if chosen_mode != mode_options[current_mode_idx]:
        if chosen_mode == "🟢 Normal Operation":
            st.session_state.scenario = "A. Live & Historical Telemetry"
            st.session_state.telemetry_window = "Live Baseline (Healthy: 2026-10-25 23:55)"
        elif chosen_mode == "🔴 Historical Incident Replay":
            st.session_state.scenario = "A. Live & Historical Telemetry"
            st.session_state.telemetry_window = "Historical Outage: UserService Critical (2026-10-23 02:35)"
        elif chosen_mode == "⚠️ High Load Replay":
            st.session_state.scenario = "A. Live & Historical Telemetry"
            st.session_state.telemetry_window = "Historical Outage: PaymentService High (2026-09-26 08:00)"
        elif chosen_mode == "🧪 Custom What-If Simulator":
            st.session_state.scenario = "B. Simulated Incident"
            st.session_state.sim_service = "PaymentService"
            st.session_state.sim_severity = "Critical"
        st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    if active_snapshot.empty:
        st.warning("⚠️ No operational telemetry available.")
    else:
        # Documented Risk Rule: Count services with predicted cascade level != 'Low'
        active_snapshot["pred_rank"] = active_snapshot["pred_cascade_level"].map(SEVERITY_ORDER).fillna(0)
        at_risk_df = active_snapshot[active_snapshot["pred_rank"] > 0]
        at_risk_count = len(at_risk_df)
        
        highest_row = active_snapshot.sort_values("pred_rank", ascending=False).iloc[0]
        highest_svc = highest_row["service_name"]
        highest_lvl = highest_row["pred_cascade_level"]

        # Clarified Subtitle: Explicit breakdown of elevated services vs nominal
        if at_risk_count == 0:
            risk_subtitle = "All 14 Nodes Nominal (Low Risk)"
        else:
            elevated_parts = []
            for lvl in ["Critical", "High", "Medium"]:
                cnt = len(active_snapshot[active_snapshot["pred_cascade_level"] == lvl])
                if cnt > 0:
                    elevated_parts.append(f"{cnt} {lvl}")
            risk_subtitle = f"Elevated: {', '.join(elevated_parts)}" if elevated_parts else f"{at_risk_count} Elevated Services"

        # 1. Four Aligned KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            render_kpi_card(
                title="Monitored Services",
                value=f"{len(active_snapshot)}",
                subtitle="All nodes streaming telemetry",
                border_color=THEME["bright_blue"]
            )
        with kpi_col2:
            render_kpi_card(
                title="Services at Risk",
                value=f"{at_risk_count}",
                subtitle=risk_subtitle,
                border_color=THEME["critical"] if at_risk_count > 0 else THEME["low"]
            )
        with kpi_col3:
            render_kpi_card(
                title="Highest Risk Service",
                value=highest_svc,
                subtitle=f"ML Pred: {highest_lvl} Risk",
                border_color=SEVERITY_COLORS.get(highest_lvl, THEME["bright_blue"])
            )
        with kpi_col4:
            render_kpi_card(
                title="Active Sustained Alerts",
                value=f"{len(active_alerts)}",
                subtitle="Hysteresis (2 Windows)",
                border_color=THEME["critical"] if len(active_alerts) > 0 else THEME["low"]
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # 2. Prominent Simulator Action Callout
        sim_col1, sim_col2 = st.columns([3.8, 1.2])
        with sim_col1:
            st.markdown(
                f"""
                <div class='sre-panel' style='background: linear-gradient(90deg, rgba(21, 159, 232, 0.12) 0%, rgba(11, 17, 26, 0.95) 100%); border-left: 3px solid {THEME["bright_blue"]}; padding: 10px 16px; margin-bottom: 0px;'>
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <div style='background: rgba(56, 189, 248, 0.2); width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1.15rem;'>🧪</div>
                        <div>
                            <div style='font-size: 0.90rem; font-weight: 700; color: {THEME["text_main"]};'>Cascade Propagation Simulator</div>
                            <div style='font-size: 0.77rem; color: {THEME["text_muted"]};'>Simulate upstream blast radius and test automated mitigations before outages propagate.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with sim_col2:
            st.markdown("<div style='height: 3px;'></div>", unsafe_allow_html=True)
            if st.button("Launch Simulator →", key="btn_overview_sim_launch", type="primary", use_container_width=True):
                st.session_state.page = "Cascade Simulator"
                st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # 3. Two-Column Layout: Time-Series + Ranked SRE Services Table
        col_ts, col_ranked = st.columns([1, 1])

        with col_ts:
            st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.90rem; font-weight: 700; color: {THEME['text_main']}; margin-bottom: 8px;'>📈 Cluster Telemetry Time-Series (Error Rate % & p95 Latency ms)</div>", unsafe_allow_html=True)
            
            history_slice = df_raw.sort_values("timestamp").tail(70).copy()
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=history_slice["timestamp"],
                y=history_slice["error_rate"],
                name="Error Rate (%)",
                line=dict(color=THEME["bright_blue"], width=2),
                mode="lines"
            ))
            fig_ts.add_trace(go.Scatter(
                x=history_slice["timestamp"],
                y=history_slice["latency_p95"] / 100.0,
                name="p95 Latency (×100 ms)",
                line=dict(color=THEME["high"], width=2, dash="dot"),
                mode="lines"
            ))
            fig_ts.update_layout(get_plotly_layout(height=320))
            st.plotly_chart(fig_ts, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ranked:
            st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.90rem; font-weight: 700; color: {THEME['text_main']}; margin-bottom: 4px;'>⚠️ Ranked Monitored Services</div>", unsafe_allow_html=True)
            st.caption("Units: Error Rate is % pts (0.45% = 0.0045). Telemetry Risk Index is continuous [0, 100].")

            ranked_df = active_snapshot.sort_values(["pred_rank", "telemetry_risk_score"], ascending=False)[[
                "service_name", "pred_cascade_level", "risk_probability", "error_rate", "latency_p95", "telemetry_risk_score", "threshold_flags"
            ]].copy()
            
            ranked_df = ranked_df.rename(columns={
                "service_name": "Service",
                "pred_cascade_level": "ML Pred",
                "risk_probability": "P(Cascade)",
                "error_rate": "Error %",
                "latency_p95": "p95 ms",
                "telemetry_risk_score": "Risk Index",
                "threshold_flags": "SRE Flags"
            })
            ranked_df["Error %"] = ranked_df["Error %"].apply(lambda x: f"{x:.2f}%")
            ranked_df["p95 ms"] = ranked_df["p95 ms"].apply(lambda x: f"{x:.1f}")
            ranked_df["Risk Index"] = ranked_df["Risk Index"].apply(lambda x: f"{x:.2f}")
            ranked_df["P(Cascade)"] = ranked_df["P(Cascade)"].apply(lambda x: f"{x:.3f}")

            # Render custom sleek HTML SRE table (no pandas index!)
            st.markdown(render_sre_table_html(ranked_df), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. Two-Column Layout: Dependency Graph Preview + Alerts Panel
        col_graph_prev, col_timeline = st.columns([1, 1])

        with col_graph_prev:
            st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.90rem; font-weight: 700; color: {THEME['text_main']}; margin-bottom: 4px;'>🕸️ Topology Preview (14 Monitored Nodes | 4 Configured Edges)</div>", unsafe_allow_html=True)
            st.caption("Documented Architecture: `Caller -> Callee`. Node color indicates current risk status.")
            
            if graph is not None:
                pos = nx.spring_layout(graph, seed=42)
                edge_x, edge_y = [], []
                for edge in graph.edges():
                    x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color="#22354D"), hoverinfo="none", mode="lines")

                node_x, node_y, node_colors, node_labels = [], [], [], []
                for node in graph.nodes():
                    x, y = pos[node]
                    node_x.append(x); node_y.append(y)
                    node_labels.append(node)
                    match = active_snapshot[active_snapshot["service_name"] == node]
                    node_lvl = match["pred_cascade_level"].iloc[0] if not match.empty else "Low"
                    node_colors.append(SEVERITY_COLORS.get(node_lvl, THEME["low"]))

                node_trace = go.Scatter(
                    x=node_x, y=node_y, 
                    mode="markers+text", 
                    text=node_labels, 
                    textposition="top center", 
                    marker=dict(size=18, color=node_colors, line=dict(width=2, color="#05070B")), 
                    hoverinfo="text"
                )
                fig_prev = go.Figure(data=[edge_trace, node_trace], layout=get_plotly_layout(height=280, margin=dict(l=10, r=10, t=10, b=10)))
                fig_prev.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                st.plotly_chart(fig_prev, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_timeline:
            st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.90rem; font-weight: 700; color: {THEME['text_main']}; margin-bottom: 4px;'>🔔 Active Sustained Alerts ({len(active_alerts)} Active)</div>", unsafe_allow_html=True)
            st.caption("Condition: Normalized risk >= 0.70 for 2 consecutive chronological observation windows.")
            
            if active_alerts:
                for alert in active_alerts:
                    st.markdown(
                        f"""
                        <div style='background: rgba(239, 68, 68, 0.10); border-left: 3px solid {THEME["critical"]}; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;'>
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='font-weight: 700; font-size: 0.85rem; color: {THEME["text_main"]};'>{alert["service"]}</span>
                                <span class='badge badge-critical'>{alert["severity"]}</span>
                            </div>
                            <div style='font-size: 0.76rem; color: {THEME["text_muted"]}; margin-top: 4px;'>{alert["message"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 50px 10px; color: {THEME["text_muted"]};'>
                        <div style='font-size: 2rem; margin-bottom: 6px;'>🛡️</div>
                        <div style='font-weight: 700; font-size: 0.88rem; color: {THEME["text_main"]};'>Zero Active Outages</div>
                        <div style='font-size: 0.76rem; margin-top: 4px;'>Active telemetry within healthy bounds. Hysteresis alert engine reports 0 sustained alerts.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # 5. Service Inspector & Root-Cause Detail View
        st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.95rem; font-weight: 700; color: {THEME['text_main']}; margin-bottom: 6px;'>🔍 Service Health Inspector & Root-Cause Analysis</div>", unsafe_allow_html=True)
        st.caption("Select any monitored microservice to inspect telemetry drivers, dependency blast radius, and automated mitigation playbooks.")
        
        all_services = sorted(active_snapshot["service_name"].unique())
        default_inspect = highest_svc if highest_svc in all_services else all_services[0]
        inspect_svc = st.selectbox("Select Service to inspect:", all_services, index=all_services.index(default_inspect))
        
        svc_row = active_snapshot[active_snapshot["service_name"] == inspect_svc].iloc[0]
        svc_lvl = svc_row["pred_cascade_level"]
        svc_prob = float(svc_row["risk_probability"])
        svc_risk = float(svc_row.get("telemetry_risk_score", 0.0))
        svc_err = float(svc_row.get("error_rate", 0.0))
        svc_p95 = float(svc_row.get("latency_p95", 0.0))
        svc_cpu = float(svc_row.get("cpu_usage", 0.0))
        svc_mem = float(svc_row.get("memory_usage", 0.0))

        # Recommender
        recommender = MitigationRecommender()
        mit_list = recommender.recommend(inspect_svc, svc_row.to_dict(), svc_lvl)
        mit_plan = mit_list[0] if mit_list else {"action": "Monitor", "reason": "Telemetry within nominal thresholds.", "priority": "Low"}

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Predicted Severity", svc_lvl, delta=f"P(Cascade): {svc_prob:.3f}", delta_color="inverse" if svc_lvl != "Low" else "normal")
        sc2.metric("Continuous Risk Index", f"{svc_risk:.1f} / 100", delta=f"Err: {svc_err:.2f}%", delta_color="inverse" if svc_err > 2.0 else "normal")
        sc3.metric("p95 Latency", f"{svc_p95:.1f} ms", delta="Nominal" if svc_p95 < 300 else "High Latency", delta_color="inverse" if svc_p95 >= 300 else "normal")
        sc4.metric("Resource Saturation", f"CPU: {svc_cpu:.0f}%", delta=f"Mem: {svc_mem:.0f}%")

        st.markdown(
            f"""
            <div style='background: #0E1825; border: 1px solid #1A2B3F; border-radius: 8px; padding: 12px 16px; margin-top: 10px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 0.85rem; font-weight: 700; color: {THEME["bright_blue"]};'>🛠️ Recommended SRE Playbook: {mit_plan.get("action", "Maintain Standard Health Checks")}</span>
                    <span class='badge' style='background: rgba(56, 189, 248, 0.15); color: {THEME["bright_blue"]};'>Priority: {mit_plan.get("priority", "Low")}</span>
                </div>
                <div style='font-size: 0.78rem; color: {THEME["text_muted"]}; margin-top: 6px;'>{mit_plan.get("reason", "Telemetry within healthy operating thresholds.")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VIEW 2: SERVICE TOPOLOGY (AUDITED PRIORITY 3 FIXES)
# -----------------------------------------------------------------------------
elif st.session_state.page == "Service Topology":
    st.title("🕸️ Service Dependency Topology")
    st.markdown("Interactive directional caller-callee mapping. Failures propagate upwards from dependencies to calling clients.")

    if graph is None:
        st.error("Graph artifact unavailable.")
    else:
        st.info("📌 **Documented Dependency Map:** Directed edges represent `Caller -> Callee` dependencies. Downstream services provide APIs to upstream callers. Edges are not invented from dependency counts.")

        col_g_meta1, col_g_meta2, col_g_meta3 = st.columns(3)
        col_g_meta1.metric("Monitored Service Nodes", len(graph.nodes()))
        col_g_meta2.metric("Configured Architecture Edges", len(graph.edges()))
        col_g_meta3.metric("Autonomous Component Nodes", len(graph.nodes()) - len(set([e[0] for e in graph.edges()] + [e[1] for e in graph.edges()])))

        pos = nx.spring_layout(graph, seed=42)
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, 
            line=dict(width=2.0, color="#22354D"), 
            hoverinfo="none", 
            mode="lines"
        )
        
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)
            
            match = active_snapshot[active_snapshot["service_name"] == node]
            node_risk = match["pred_cascade_level"].iloc[0] if not match.empty else "Low"
            
            node_text.append(f"<b>{node}</b><br>Upstream Callers: {in_deg}<br>Downstream Deps: {out_deg}<br>ML Pred: {node_risk}")
            node_color.append(SEVERITY_COLORS.get(node_risk, THEME["low"]))
            node_size.append(18 + (in_deg + out_deg) * 3)

        node_trace = go.Scatter(
            x=node_x, y=node_y, 
            mode="markers+text", 
            text=[node for node in graph.nodes()], 
            textposition="top center", 
            marker=dict(size=node_size, color=node_color, line=dict(width=2, color="#05070B")), 
            hoverinfo="text",
            hovertext=node_text
        )
        
        fig = go.Figure(data=[edge_trace, node_trace], layout=get_plotly_layout(height=480))
        fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
        st.subheader("🔍 Service Inspector & Dependency Analysis")
        selected_node = st.selectbox("Select Microservice to Inspect", sorted(list(graph.nodes())), key="topo_inspect_select")
        
        if selected_node:
            deps = list(graph.successors(selected_node))
            callers = list(graph.predecessors(selected_node))

            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.markdown(f"**Microservice:** `{selected_node}`")
                st.write(f"Direct Upstream Callers: **{len(callers)}**")
                st.write(f"Direct Downstream Dependencies: **{len(deps)}**")

            with col_i2:
                st.markdown("**Downstream Dependencies (Callees):**")
                if deps:
                    for d in deps:
                        st.write(f"- 📦 `{d}`")
                else:
                    st.write("*(None — Leaf Service)*")

            with col_i3:
                st.markdown("**Upstream Callers (Dependents):**")
                if callers:
                    for c in callers:
                        st.write(f"- 🚀 `{c}`")
                else:
                    st.write("*(None — Entry Gateway)*")

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 3: CASCADE SIMULATOR
# -----------------------------------------------------------------------------
elif st.session_state.page == "Cascade Simulator":
    st.title("🧪 What-if Cascade Propagation Simulator")
    st.markdown("Simulate potential failure propagation and assess blast radius across the service dependency graph.")

    st.warning("⚠️ **Heuristic Simulation Notice:** Propagation probabilities are heuristic estimates based on graph distance decay, NOT calibrated causal probabilities.")

    if graph is None:
        st.error("Graph artifact unavailable.")
    else:
        st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
        c_sim1, c_sim2, c_sim3 = st.columns([2, 2, 1])
        with c_sim1:
            sim_node = st.selectbox("Root Failure Service", sorted(list(graph.nodes())), index=sorted(list(graph.nodes())).index(st.session_state.sim_service) if st.session_state.sim_service in graph.nodes() else 0, key="sim_page_node")
        with c_sim2:
            sim_lvl = st.selectbox("Failure Severity", ["Low", "Medium", "High", "Critical"], index=["Low", "Medium", "High", "Critical"].index(st.session_state.sim_severity), key="sim_page_lvl")
        with c_sim3:
            st.write("")
            st.write("")
            run_sim_btn = st.button("Run Simulation", type="primary", use_container_width=True)

        simulator = CascadeSimulator()
        sim_res = simulator.whatif(sim_node, sim_lvl)

        if "error" in sim_res:
            st.error(sim_res["error"])
        else:
            impacted = sim_res.get("impacted_services", [])
            blast_radius = len(impacted)
            max_depth = sim_res.get("cascade_depth", 0)

            c_k1, c_k2, c_k3, c_k4 = st.columns(4)
            c_k1.metric("Root Failure Service", sim_node)
            c_k2.metric("Injected Severity", sim_lvl)
            c_k3.metric("Blast Radius (Services)", blast_radius)
            c_k4.metric("Cascade Depth", max_depth)

            st.markdown("---")
            if impacted:
                col_chart, col_tbl = st.columns([3, 2])
                with col_chart:
                    st.subheader("📉 Estimated Failure Probability by Service")
                    df_imp = pd.DataFrame(impacted)
                    fig_bar = px.bar(
                        df_imp, 
                        x="service", 
                        y="est_failure_prob", 
                        color="distance", 
                        labels={"service": "Service", "est_failure_prob": "Est. Failure Probability", "distance": "Hop Distance"},
                        color_continuous_scale="Reds",
                        range_y=[0, 1.0]
                    )
                    fig_bar.update_layout(get_plotly_layout(height=320))
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_tbl:
                    st.subheader("📋 Impact Assessment Details")
                    st.dataframe(
                        df_imp.rename(columns={
                            "service": "Service",
                            "distance": "Distance",
                            "est_failure_prob": "Est. Probability",
                            "source": "Method"
                        }),
                        use_container_width=True
                    )

            st.subheader(f"🛡️ SRE Recommended Mitigations for {sim_node}")
            recommender = MitigationRecommender()
            severity_metrics = {
                "Low": {"error_rate": 0.5, "latency_p95": 200, "cpu_usage": 50},
                "Medium": {"error_rate": 4.5, "latency_p95": 400, "cpu_usage": 70},
                "High": {"error_rate": 8.0, "latency_p95": 750, "cpu_usage": 85},
                "Critical": {"error_rate": 15.0, "latency_p95": 1200, "cpu_usage": 95}
            }
            recs = recommender.recommend(sim_node, severity_metrics[sim_lvl], sim_lvl)
            for r in recs:
                st.info(f"**[{r['priority']}] {r['action']}**: {r['reason']}")

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 4: SERVICE EXPLORER (SERVICE DETAIL)
# -----------------------------------------------------------------------------
elif st.session_state.page == "Service Explorer":
    st.title("🔍 Service Explorer & Live Inference")
    st.markdown("Detailed telemetry, live ML model predictions, and rule-based mitigation actions for a specific microservice.")

    if df_raw.empty:
        st.warning("No telemetry available.")
    else:
        services = sorted(df_raw["service_name"].unique())
        selected_svc = st.selectbox("Choose Service to Explore", services, key="svc_explore_select")
        svc_df = df_raw[df_raw["service_name"] == selected_svc].sort_values("timestamp")

        if svc_df.empty:
            st.warning(f"No records found for {selected_svc}.")
        else:
            latest_obs = svc_df.iloc[-1]

            # Metric KPI row
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric("Current Error Rate", f"{latest_obs.get('error_rate', 0):.2f}%")
            c_s2.metric("p95 Latency", f"{latest_obs.get('latency_p95', 0):.1f} ms")
            c_s3.metric("CPU Utilization", f"{latest_obs.get('cpu_usage', 0):.1f}%")
            c_s4.metric("Timeout Rate", f"{latest_obs.get('timeout_rate', 0):.2f}%")

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            col_pred, col_recom = st.columns(2)

            with col_pred:
                st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
                st.subheader("🤖 Model Cascade Risk Prediction")
                if model is not None and encoder is not None:
                    input_data = pd.DataFrame([latest_obs[model_features]]).fillna(0)
                    pred_code = model.predict(input_data)[0]
                    pred_label = encoder.inverse_transform([pred_code])[0]
                    pred_probs = model.predict_proba(input_data)[0]

                    st.markdown(f"Predicted Cascade Level: {render_badge(pred_label, pred_label)}", unsafe_allow_html=True)
                    
                    prob_df = pd.DataFrame({
                        "Risk Level": encoder.classes_,
                        "Probability": pred_probs
                    })
                    fig_prob = px.bar(
                        prob_df, 
                        x="Risk Level", 
                        y="Probability", 
                        color="Risk Level",
                        color_discrete_map=SEVERITY_COLORS,
                        range_y=[0, 1.0]
                    )
                    fig_prob.update_layout(get_plotly_layout(height=240))
                    st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.info("Model artifacts not available for live prediction.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_recom:
                st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
                st.subheader("🛡️ Automated Mitigation Rules")
                recommender = MitigationRecommender()
                current_metrics = {
                    "error_rate": float(latest_obs.get("error_rate", 0)),
                    "latency_p95": float(latest_obs.get("latency_p95", 0)),
                    "cpu_usage": float(latest_obs.get("cpu_usage", 0))
                }
                current_level = latest_obs.get("cascade_level", "Low")
                recs = recommender.recommend(selected_svc, current_metrics, current_level)
                for r in recs:
                    st.info(f"**[{r['priority']}] {r['action']}**: {r['reason']}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
            st.subheader("📈 Historical Metric Trends")
            col_ts1, col_ts2 = st.columns(2)
            with col_ts1:
                fig_err = px.line(svc_df.tail(60), x="timestamp", y="error_rate", title="Error Rate (%)")
                fig_err.update_layout(get_plotly_layout(height=260))
                st.plotly_chart(fig_err, use_container_width=True)
            with col_ts2:
                fig_lat = px.line(svc_df.tail(60), x="timestamp", y="latency_p95", title="p95 Latency (ms)")
                fig_lat.update_layout(get_plotly_layout(height=260))
                st.plotly_chart(fig_lat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 5: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif st.session_state.page == "Model Performance":
    st.title("📈 Model Performance & Evaluation Report")
    st.markdown("Evaluation metrics derived from a strict chronological 70% Train / 15% Val / 15% Test split.")

    if report is None:
        st.warning("Evaluation report not found at reports/evaluation_report.json.")
    else:
        base_f1 = report.get("baseline_metrics", {}).get("macro_f1", 0)
        chal_f1 = report.get("challenger_metrics", {}).get("macro_f1", 0)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Macro-F1", f"{base_f1:.4f}")
        c2.metric("Challenger Macro-F1", f"{chal_f1:.4f}")
        c3.metric("Production Champion", "Challenger (500 Trees, Depth 15)")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        col_rep, col_feat = st.columns(2)

        with col_rep:
            st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
            st.subheader("📋 Challenger Classification Report")
            chal_report = report.get("challenger_metrics", {}).get("report", {})
            class_rows = []
            for cls in ["Critical", "High", "Medium", "Low"]:
                if cls in chal_report:
                    row = chal_report[cls]
                    class_rows.append({
                        "Class": cls,
                        "Precision": f"{row.get('precision', 0):.3f}",
                        "Recall": f"{row.get('recall', 0):.3f}",
                        "F1-Score": f"{row.get('f1-score', 0):.3f}",
                        "Support": int(row.get('support', 0))
                    })
            st.table(pd.DataFrame(class_rows))

            pr_auc = report.get("challenger_metrics", {}).get("pr_auc", {})
            if pr_auc:
                st.markdown("**One-vs-Rest PR-AUC Scores:**")
                pr_rows = [{"Class": k, "PR-AUC": f"{v:.4f}"} for k, v in pr_auc.items()]
                st.table(pd.DataFrame(pr_rows))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_feat:
            st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
            st.subheader("🏆 Feature Importance (Gini)")
            feat_imp = report.get("feature_importance", {})
            if feat_imp:
                imp_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
                fig_imp.update_layout(get_plotly_layout(height=380))
                st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
        st.subheader("🎯 Confusion Matrix")
        conf_mat = report.get("challenger_metrics", {}).get("confusion_matrix", [])
        if conf_mat:
            labels = ["Critical", "High", "Low", "Medium"]
            fig_cm = px.imshow(
                conf_mat, 
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=labels, 
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(get_plotly_layout(height=340))
            st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEW 6: SETTINGS & DATA IMPORT
# -----------------------------------------------------------------------------
elif st.session_state.page == "Settings & Data Import":
    st.title("⚙️ Settings & Telemetry Data Import")
    st.markdown("Manage telemetry data sources, import custom datasets, and control observability thresholds.")

    st.markdown("<div class='sre-panel'>", unsafe_allow_html=True)
    st.subheader("📂 Custom Telemetry CSV Import")
    st.caption("Upload a custom telemetry CSV matching the microservice schema to evaluate predictions.")

    uploaded_csv = st.file_uploader("Upload Telemetry CSV", type=["csv"], key="telemetry_csv_uploader")
    if uploaded_csv is not None:
        try:
            custom_df = pd.read_csv(uploaded_csv)
            st.success(f"Successfully loaded CSV with {len(custom_df)} rows and {len(custom_df.columns)} columns.")
            st.dataframe(custom_df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

    st.markdown("---")
    st.subheader("🔄 Cache & Pipeline Management")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Clear Streamlit Caches & Reload Data", use_container_width=True):
            clear_all_caches()
            st.success("All Streamlit data caches cleared successfully.")
            st.rerun()
    with col_c2:
        st.write(f"Processed Feature Store: `{len(df_raw)} records loaded`")
        st.write(f"Active Model Features: `{len(model_features)} inputs`")

    st.markdown("</div>", unsafe_allow_html=True)