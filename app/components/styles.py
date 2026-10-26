"""
Styles and UI components for the SRE Observability Dashboard.
Visual Identity:
- Jet-black background: #05070B
- Panel background: #0B111A
- Cerulean blue: #159FE8
- Bright blue highlight: #38BDF8
- Main text: #F4F8FC
- Muted text: #8FA2B8
"""

import streamlit as st
import plotly.graph_objects as go

THEME = {
    "bg": "#05070B",
    "panel_bg": "#0B111A",
    "panel_border": "#162232",
    "cerulean": "#159FE8",
    "bright_blue": "#38BDF8",
    "text_main": "#F4F8FC",
    "text_muted": "#8FA2B8",
    "critical": "#EF4444",
    "high": "#F97316",
    "medium": "#FBBF24",
    "low": "#10B981"
}

def inject_custom_css():
    """Injects high-end production SRE observability CSS theme."""
    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global resets and typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {THEME["text_main"]};
        background-color: {THEME["bg"]};
    }}

    .stApp {{
        background: {THEME["bg"]};
        color: {THEME["text_main"]};
    }}

    .block-container {{
        padding-top: 1.0rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #070C13 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
    }}
    
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
        gap: 0.6rem;
    }}

    /* Radio navigation buttons styling - hide dummy widget label */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
        display: none !important;
    }}

    section[data-testid="stSidebar"] .stRadio label[data-testid="stWidgetLabel"] {{
        display: none !important;
    }}

    section[data-testid="stSidebar"] .stRadio > div {{
        background-color: transparent;
        gap: 4px;
    }}
    
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
        background-color: #0B121C;
        border: 1px solid #14202F;
        border-radius: 8px;
        padding: 8px 14px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 3px;
        color: {THEME["text_muted"]};
        font-size: 0.86rem;
        font-weight: 500;
        display: flex;
        align-items: center;
    }}
    
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
        border-color: {THEME["bright_blue"]};
        color: {THEME["text_main"]};
        background-color: #0E1825;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.15);
    }}

    /* Card Panels */
    .sre-panel {{
        background: {THEME["panel_bg"]};
        border: 1px solid {THEME["panel_border"]};
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 1px rgba(56, 189, 248, 0.15);
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }}

    .sre-panel:hover {{
        border-color: rgba(56, 189, 248, 0.35);
    }}

    /* KPI Cards */
    .kpi-card {{
        background: linear-gradient(180deg, rgba(14, 24, 37, 0.9) 0%, {THEME["panel_bg"]} 100%);
        border: 1px solid {THEME["panel_border"]};
        border-top: 2px solid {THEME["bright_blue"]};
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 82px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 12px rgba(56, 189, 248, 0.15);
        border-color: {THEME["cerulean"]};
    }}

    .kpi-title {{
        color: {THEME["text_muted"]};
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 3px;
    }}

    .kpi-value {{
        color: {THEME["text_main"]};
        font-size: 1.30rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .kpi-subtext {{
        color: {THEME["text_muted"]};
        font-size: 0.72rem;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Status Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    .badge-critical {{
        background: rgba(239, 68, 68, 0.15);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
    }}

    .badge-high {{
        background: rgba(249, 115, 22, 0.15);
        color: #FDBA74;
        border: 1px solid rgba(249, 115, 22, 0.4);
    }}

    .badge-medium {{
        background: rgba(251, 191, 36, 0.15);
        color: #FDE68A;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }}

    .badge-low {{
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }}

    .badge-info {{
        background: rgba(56, 189, 248, 0.12);
        color: {THEME["bright_blue"]};
        border: 1px solid rgba(56, 189, 248, 0.3);
    }}

    /* Top App Header */
    .app-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 20px;
        background: {THEME["panel_bg"]};
        border: 1px solid {THEME["panel_border"]};
        border-radius: 10px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }}

    .app-header-left {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}

    .app-header-right {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    /* Modern Buttons */
    .stButton > button {{
        background: linear-gradient(180deg, #18283D 0%, #0E1A29 100%) !important;
        color: {THEME["text_main"]} !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 7px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }}

    .stButton > button:hover {{
        border-color: {THEME["bright_blue"]} !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.25) !important;
        transform: translateY(-1px) !important;
    }}

    /* Primary Buttons */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(180deg, {THEME["bright_blue"]} 0%, {THEME["cerulean"]} 100%) !important;
        color: #030712 !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.4) !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.6) !important;
    }}

    /* Dataframe table customization */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {THEME["panel_border"]};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Selectboxes and Inputs */
    div[data-baseweb="select"] > div {{
        background-color: #0A111A !important;
        border-color: {THEME["panel_border"]} !important;
        color: {THEME["text_main"]} !important;
        border-radius: 6px !important;
    }}

    div[data-baseweb="select"]:hover > div {{
        border-color: {THEME["bright_blue"]} !important;
    }}

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid {THEME["panel_border"]};
        padding-bottom: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border: none;
        color: {THEME["text_muted"]};
        font-weight: 500;
        padding: 8px 16px;
        border-radius: 6px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: rgba(56, 189, 248, 0.1) !important;
        color: {THEME["bright_blue"]} !important;
        font-weight: 600;
        border-bottom: 2px solid {THEME["bright_blue"]} !important;
    }}

    /* Alerts and Callouts */
    .stAlert {{
        background-color: #0A121D !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 8px !important;
        color: {THEME["text_main"]} !important;
    }}
    /* SRE Custom HTML Table */
    .sre-table-container {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid #162436;
        border-radius: 8px;
        background: #0B111A;
        max-height: 380px;
        overflow-y: auto;
    }}
    .sre-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        text-align: left;
    }}
    .sre-table th {{
        background: #0E1825;
        color: #8FA2B8;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        padding: 9px 12px;
        border-bottom: 1px solid #1A293D;
        white-space: nowrap;
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    .sre-table td {{
        padding: 9px 12px;
        border-bottom: 1px solid #121E2E;
        color: #F4F8FC;
        white-space: nowrap;
    }}
    .sre-table tr:hover td {{
        background: rgba(56, 189, 248, 0.05);
    }}
    .progress-bar-bg {{
        width: 55px;
        height: 5px;
        background: #14202F;
        border-radius: 3px;
        overflow: hidden;
        display: inline-block;
        vertical-align: middle;
        margin-right: 6px;
    }}
    .progress-bar-fill {{
        height: 100%;
        border-radius: 3px;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def get_plotly_layout(height=350, title=None, margin=None):
    """Returns a unified dark observability layout for Plotly figures."""
    if margin is None:
        margin = dict(l=35, r=20, t=40 if title else 20, b=30)
    
    return go.Layout(
        title=dict(
            text=title or "",
            font=dict(family="Inter, sans-serif", size=14, color=THEME["text_main"]),
            x=0.02,
            y=0.96
        ) if title else None,
        paper_bgcolor=THEME["panel_bg"],
        plot_bgcolor=THEME["panel_bg"],
        font=dict(family="Inter, sans-serif", color=THEME["text_muted"], size=11),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color=THEME["text_muted"], size=10),
            showline=False
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color=THEME["text_muted"], size=10),
            showline=False
        ),
        margin=margin,
        height=height,
        hoverlabel=dict(
            bgcolor="#0E1825",
            bordercolor=THEME["bright_blue"],
            font=dict(color=THEME["text_main"], family="JetBrains Mono, monospace")
        )
    )

def render_kpi_card(title, value, subtitle=None, border_color=None):
    """Renders a styled KPI card with dark theme, responsive font sizing, and glowing accent."""
    top_border = border_color or THEME["bright_blue"]
    subtext_html = f"<div class='kpi-subtext'>{subtitle}</div>" if subtitle else ""
    val_str = str(value)
    
    # Scale font size dynamically for long service names so PaymentService is never truncated
    if len(val_str) > 13:
        val_font_size = "1.10rem"
    elif len(val_str) > 8:
        val_font_size = "1.25rem"
    else:
        val_font_size = "1.55rem"

    card_html = (
        f"<div class='kpi-card' style='border-top: 2px solid {top_border};'>"
        f"<div>"
        f"<div class='kpi-title'>{title}</div>"
        f"<div class='kpi-value' style='font-size: {val_font_size}; white-space: nowrap; overflow: visible;' title='{val_str}'>{val_str}</div>"
        f"</div>"
        f"{subtext_html}"
        f"</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)

def render_badge(text, level="low"):
    """Renders a stylized risk badge pill."""
    lvl = str(level).lower()
    if "crit" in lvl:
        cls_name = "badge-critical"
    elif "high" in lvl:
        cls_name = "badge-high"
    elif "med" in lvl:
        cls_name = "badge-medium"
    elif "low" in lvl:
        cls_name = "badge-low"
    else:
        cls_name = "badge-info"
    return f"<span class='badge {cls_name}'>{text}</span>"

def render_sre_table_html(df):
    """Renders a custom styled high-end SRE table for monitored services without Markdown indentation artifacts."""
    rows_html = []
    for _, row in df.iterrows():
        svc = row["Service"]
        ml_pred = row["ML Pred"]
        badge_html = render_badge(ml_pred, ml_pred)
        p_casc = f"{float(row['P(Cascade)']):.3f}"
        
        risk_val = float(row["Risk Index"])
        fill_color = THEME["critical"] if risk_val >= 60 else (THEME["high"] if risk_val >= 35 else (THEME["medium"] if risk_val >= 15 else THEME["low"]))
        progress_html = (
            f"<div style='display: flex; align-items: center;'>"
            f"<div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {min(100, max(4, risk_val))}%; background: {fill_color};'></div></div>"
            f"<span style='font-family: monospace; font-size: 0.80rem;'>{risk_val:.1f}</span>"
            f"</div>"
        )
        err_val = row["Error %"]
        p95_val = row["p95 ms"]
        flags_val = row.get("SRE Flags", "Nominal")
        flag_badge = f"<span style='color: {THEME['text_muted']}; font-size: 0.75rem;'>{flags_val}</span>" if flags_val == "Nominal" else f"<span style='color: {THEME['critical']}; font-size: 0.75rem; font-weight: 600;'>{flags_val}</span>"

        row_html = (
            f"<tr>"
            f"<td style='font-weight: 600; color: #F4F8FC;'><span style='color: {THEME['bright_blue']}; font-size: 0.8rem;'>●</span> {svc}</td>"
            f"<td>{badge_html}</td>"
            f"<td style='font-family: monospace; font-size: 0.80rem; color: #8FA2B8;'>{p_casc}</td>"
            f"<td>{progress_html}</td>"
            f"<td style='font-family: monospace;'>{err_val}</td>"
            f"<td style='font-family: monospace;'>{p95_val}</td>"
            f"<td>{flag_badge}</td>"
            f"</tr>"
        )
        rows_html.append(row_html)
        
    full_table = (
        f"<div class='sre-table-container'>"
        f"<table class='sre-table'>"
        f"<thead><tr>"
        f"<th>Service Name</th>"
        f"<th>ML Prediction</th>"
        f"<th>P(Cascade)</th>"
        f"<th>Risk Index (0-100)</th>"
        f"<th>Error Rate</th>"
        f"<th>p95 Latency</th>"
        f"<th>Risk Factors / SRE Flags</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )
    return full_table
