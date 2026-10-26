# API Failure Cascade Predictor

An ML-powered system designed to predict, simulate, and mitigate failure cascades in a microservice architecture.

## 🚀 Project Purpose
The system identifies microservices at high risk of triggering or suffering from cascading failures. It provides Site Reliability Engineers (SREs) with:
1. **Real-time Risk Overview**: Instant visibility into at-risk services and active hysteresis-based alerts.
2. **Dependency Graph Visualization**: Interactive caller-callee topology with in/out-degree inspection.
3. **What-if Simulation**: Heuristic Monte Carlo propagation engine estimating blast radius and propagation depth.
4. **Service Detail & Mitigation**: Live model inference and rule-based mitigation recommendations (circuit breakers, timeout adjustments, scaling, retry storm prevention).
5. **Model Evaluation & Performance**: Full audit metrics comparing Baseline vs. Challenger models, confusion matrices, PR-AUC, and feature importance.

---

## 🛠️ Architecture & Data Flow
1. **Data Collection**: Microservice metrics (error rate, p95/p99 latency, CPU, memory, timeout rate).
2. **Feature Engineering** (`src/features/`):
   - **Temporal Features**: Leakage-safe rolling means and slopes (15m, 30m, 60m), error acceleration, and lag terms.
   - **Graph Features**: In-degree, out-degree, betweenness centrality, transitive callers, and critical path length.
   - **Context Features**: Peak-hour indicators and weekend context.
3. **Predictive Modeling** (`src/models/`):
   - Strict chronological split (70% Train / 15% Validation / 15% Test) to prevent temporal leakage.
   - Target: `cascade_level` (Categorical: `Critical`, `High`, `Medium`, `Low`).
   - Macro-F1 optimization for severe class imbalance.
4. **Decision Engine** (`src/decision/`):
   - **Simulator (`simulator.py`)**: Heuristic Monte Carlo simulation of upstream blast radius.
   - **Mitigation Recommender (`mitigation.py`)**: Explainable SRE action recommendations.
   - **Alert Engine (`alerts.py`)**: Hysteresis-based multi-interval threshold detection.
5. **Dashboard (`app/dashboard.py`)**: Streamlit web application providing 5 interactive views.

---

## 📂 Folder Structure
```
api_failure_cascade_predictor_starter/
├── app/
│   ├── components/
│   │   └── data_loader.py       # Cached loaders for dataset, models, graph, report
│   └── dashboard.py             # Streamlit 5-page dashboard entrypoint
├── data/
│   ├── raw/
│   │   └── api_failure_cascade_dataset.csv  # Raw metrics dataset
│   └── processed/
│       └── features.parquet     # Processed feature store
├── models/
│   ├── cascade_predictor.joblib # Trained Random Forest model (scikit-learn 1.9.1)
│   └── label_encoder.joblib     # Target label encoder
├── reports/
│   └── evaluation_report.json   # Model evaluation report & metrics
├── src/
│   ├── data/
│   │   └── graph.py             # Documented microservice dependency graph
│   ├── decision/
│   │   ├── alerts.py            # AlertEngine with hysteresis
│   │   ├── demo.py              # Interactive demo engine & synthetic stress tests
│   │   ├── mitigation.py        # MitigationRecommender rules
│   │   └── simulator.py         # CascadeSimulator (Heuristic Monte Carlo)
│   ├── features/
│   │   ├── build_features.py    # Feature engineering orchestration
│   │   ├── context.py           # Temporal/calendar context
│   │   ├── graph_features.py    # Graph topology features
│   │   └── temporal.py          # Rolling time-series features
│   └── models/
│       └── train.py             # Chronological split training & evaluation
├── tests/
│   ├── model_tests.py           # Model and chronological split tests
│   ├── test_decision.py         # Decision engine & simulator tests
│   ├── test_demo.py             # Demo scenario, scale & isolation tests
│   └── test_features.py         # Leakage and feature computation tests
├── pytest.ini                   # Pytest test discovery configuration
├── requirements.txt             # Verified compatible dependencies
└── README.md                    # System documentation
```

---

## 📊 Model Evaluation Summary
- **Chronological Split**: 70% Train / 15% Validation / 15% Test
- **Baseline Model**: Random Forest (100 trees, balanced weights) — Macro-F1: `0.7549`
- **Challenger Model**: Random Forest (500 trees, depth 15, balanced subsample) — Macro-F1: `0.7346`
- **Test Accuracy**: `97.83%`

---

## ⚙️ Windows Installation & Setup Instructions

### Prerequisites
- Python 3.9 - 3.14 (Verified on Python 3.14.2)
- Windows PowerShell or Windows Command Prompt

### 1. Environment Setup (PowerShell)
```powershell
# Navigate to the project root
cd c:\Users\HANSIKA\Downloads\api_failure_cascade_predictor_starter

# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install -r requirements.txt
```

*(If execution of scripts is restricted in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or activate using Command Prompt: `venv\Scripts\activate.bat`)*

### 2. Run Test Suite
```powershell
pytest
```
*Expected output: `19 passed` across `tests/model_tests.py`, `tests/test_audit_fixes.py`, `tests/test_decision.py`, `tests/test_demo.py`, and `tests/test_features.py`.*

### 3. Launch Dashboard
```powershell
streamlit run app/dashboard.py
```
Open your browser to `http://localhost:8501`.

---

## 🎮 Interactive Demo Mode & Scenarios
The dashboard includes an interactive Demo Mode with three operational scenarios:

### Scenario A: Normal Operation (Original Telemetry)
- Evaluates live model inference across all 14 microservices on the latest observed data.
- Accurately demonstrates calm, healthy baseline behavior (`Low` risk across services).
- Includes **Historical Incident Replay** toggle to demonstrate model behavior during an actual historical incident (e.g. `UserService` Critical failure).

### Scenario B: Simulated Incident (What-if Blast Radius)
- Interactive root failure injection using `CascadeSimulator` across service and severity levels.
- Traverses the reverse dependency graph (`Caller <- Callee`) to compute upstream blast radius and propagation depth.
- Clearly separates **heuristic Monte Carlo probabilities** from ML model predictions, with explicit heuristic disclaimers.

### Scenario C: Synthetic Stress Test (Controlled In-Memory Simulation)
- Generates a separate, temporary in-memory synthetic stress dataset with user-selected target services, stress intensity (`Moderate`, `Severe`, `Extreme`), and consecutive intervals.
- Dynamically recomputes temporal, graph, and contextual features through the feature pipeline.
- Evaluates the trained model against ground-truth simulation rules, displaying the Confusion Matrix, per-class Precision/Recall/F1, and Macro-F1.
- Never mutates or overwrites raw CSV or processed feature artifacts.

---

## 🧭 Dashboard Views & Panels
1. **Overview (3 Structured Panels)**:
   - **Panel 1: Telemetry Data**: Raw and aggregated service metrics (Error Rate %, p95 Latency ms, CPU %, Timeout %).
   - **Panel 2: ML Model Predictions**: Live inference, predicted risk badge, risk probability `P(Non-Low)`, and risk distribution chart.
   - **Panel 3: Simulation & Stress Evaluation**: Blast radius breakdown (Scenario B) or Confusion Matrix and per-class metrics report (Scenario C).
   - **Active Sustained Alerts**: Hysteresis-based multi-interval alerts using normalized probability scaling to eliminate false positives.
2. **Dependency Graph**: Interactive network graph showing callers and dependencies, in/out-degree, and service inspector.
3. **What-if Simulator**: Interactive simulation allowing selection of root service and severity to evaluate blast radius and cascade depth.
4. **Service Detail**: Deep dive into individual service performance, recent metric trends, live cascade risk classification, and tailored mitigation rules.
5. **Model Performance**: Comprehensive evaluation report, Challenger classification report, confusion matrix heatmap, one-vs-rest PR-AUC, and Gini feature importances.

---

## ⚠️ Limitations & Notes
- **Documented Dependency Map**: The microservice dependency topology follows the documented architecture (`Caller -> Callee`).
- **Heuristic Propagation**: What-if simulation probabilities are heuristic Monte Carlo estimates based on graph distance decay, not calibrated causal probabilities.
- **Class Imbalance**: High skew towards `Low` risk (93%+ of samples); Macro-F1 is prioritized over raw accuracy.
