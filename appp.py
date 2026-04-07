"""
Gas Station Safety Intelligence Dashboard
Dueling Double DQN Autonomous Agent — Production Monitor
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import streamlit as st

# ── Page config must be first ──────────────────────────────────────
st.set_page_config(
    page_title="GasSafe AI — Control Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ──────────────────────────────────────────────────
try:
    from src.tools.decision_tool import DecisionTool
    from src.agent.reward_system import compute_reward, is_correct_action, get_expected_action
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# =========================================================
# CONSTANTS
# =========================================================
ACTIONS = {0:"Monitor", 1:"Increase Sampling", 2:"Request Verification",
           3:"Raise Alarm", 4:"Emergency Shutdown"}
GAS_MAP   = {"NoGas":0, "Smoke":1, "Mixture":2, "Perfume":3}
GAS_NAMES = {0:"NoGas", 1:"Smoke", 2:"Mixture", 3:"Perfume"}
CORRECT_ACTIONS = {0:[0], 1:[3], 2:[4], 3:[1,2]}
DANGER_GAS_IDS  = {1, 2}

RAW_SENSOR_COLS = ["MQ2","MQ3","MQ5","MQ6","MQ7","MQ8","MQ135"]
DELTA_COLS      = ["dMQ2","dMQ3","dMQ5","dMQ6","dMQ7","dMQ8","dMQ135"]
STD_COLS        = ["sMQ2","sMQ3","sMQ5","sMQ6","sMQ7","sMQ8","sMQ135"]

ACTION_COLORS = {
    0: "#22c55e",   # green — Monitor
    1: "#3b82f6",   # blue  — Increase Sampling
    2: "#f59e0b",   # amber — Request Verification
    3: "#f97316",   # orange — Raise Alarm
    4: "#ef4444",   # red   — Emergency Shutdown
}
GAS_COLORS = {
    "NoGas":   "#22c55e",
    "Smoke":   "#f97316",
    "Mixture": "#ef4444",
    "Perfume": "#3b82f6",
}


# =========================================================
# CSS — INDUSTRIAL SCADA HUD AESTHETIC
# =========================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@200;300;400;600;700&display=swap');

    :root {
        --bg0:    #060a12;
        --bg1:    #0d1420;
        --bg2:    #111c2e;
        --bg3:    #162035;
        --border: #1e3a5f;
        --border2:#264d7a;
        --amber:  #f59e0b;
        --amber2: #fbbf24;
        --green:  #22c55e;
        --red:    #ef4444;
        --orange: #f97316;
        --blue:   #3b82f6;
        --cyan:   #06b6d4;
        --text:   #e2e8f0;
        --muted:  #64748b;
        --mono:   'Share Tech Mono', monospace;
        --display:'Rajdhani', sans-serif;
        --body:   'Exo 2', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: var(--body);
        background-color: var(--bg0) !important;
        color: var(--text) !important;
    }

    .stApp { background: var(--bg0); }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg1) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

    /* Headers */
    h1,h2,h3,h4 { font-family: var(--display) !important; letter-spacing: 0.05em; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    [data-testid="stMetricLabel"] { font-family: var(--mono) !important; font-size: 11px !important; color: var(--muted) !important; letter-spacing: 0.1em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { font-family: var(--display) !important; font-size: 2rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 12px !important; }

    /* Buttons */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--amber) !important;
        color: var(--amber) !important;
        font-family: var(--display) !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        border-radius: 4px !important;
    }
    .stButton > button:hover {
        background: var(--amber) !important;
        color: var(--bg0) !important;
    }
    .run-btn > button {
        background: var(--amber) !important;
        color: var(--bg0) !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
    }

    /* Selectbox, slider */
    .stSelectbox div, .stSlider { color: var(--text) !important; }
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--bg2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: var(--bg1); border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: var(--display); letter-spacing: 0.05em; }
    .stTabs [aria-selected="true"] { color: var(--amber) !important; border-bottom: 2px solid var(--amber) !important; }

    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-family: var(--mono) !important;
        color: var(--text) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg1) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }

    /* Progress bar */
    .stProgress > div > div { background: var(--amber) !important; }

    /* Divider */
    hr { border-color: var(--border) !important; }

    /* Custom classes */
    .hud-header {
        background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
        border: 1px solid var(--border);
        border-top: 3px solid var(--amber);
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .hud-title {
        font-family: var(--display);
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--amber);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0;
    }
    .hud-subtitle {
        font-family: var(--mono);
        font-size: 0.75rem;
        color: var(--muted);
        letter-spacing: 0.15em;
        margin-top: 4px;
    }
    .status-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .status-online { background: var(--green); }
    .status-offline { background: var(--red); }
    @keyframes pulse {
        0%,100% { opacity:1; }
        50% { opacity:0.4; }
    }
    .section-header {
        font-family: var(--mono);
        font-size: 0.7rem;
        color: var(--muted);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }
    .gas-badge {
        display: inline-block;
        font-family: var(--mono);
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 3px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .action-badge {
        display: inline-block;
        font-family: var(--mono);
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 3px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card-label {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--muted);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-card-value {
        font-family: var(--display);
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-card-sub {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        margin-top: 4px;
    }
    .label-card {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
    }
    .label-card-title {
        font-family: var(--display);
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.08em;
    }
    .expl-box {
        background: var(--bg1);
        border-left: 3px solid var(--cyan);
        border-radius: 0 6px 6px 0;
        padding: 0.8rem 1rem;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.6;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    .crit-box {
        background: var(--bg1);
        border-left: 3px solid var(--amber);
        border-radius: 0 6px 6px 0;
        padding: 0.8rem 1rem;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.6;
        color: #94a3b8;
        margin-top: 0.4rem;
    }
    .correct-row { border-left: 3px solid var(--green); }
    .wrong-row   { border-left: 3px solid var(--red); }
    .tick-correct { color: var(--green); font-size: 1.1rem; }
    .tick-wrong   { color: var(--red); font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# EXPLANATION ENGINE
# =========================================================
def generate_explanation(gas_id, action, anomaly, q_values, is_correct, policy_conf):
    gas = GAS_NAMES.get(gas_id, "Unknown")
    act = ACTIONS.get(action, "Unknown")
    q   = np.array(q_values)
    q_gap = float(q.max() - np.sort(q)[-2]) if len(q) > 1 else 0.0

    if not is_correct:
        correct = CORRECT_ACTIONS.get(gas_id, [])
        exp_str = " or ".join(ACTIONS[a] for a in correct)
        sev = "SAFETY CRITICAL" if gas_id in DANGER_GAS_IDS else "suboptimal"
        return (
            f"[{sev}] DQN chose {act} for {gas} (gas_id={gas_id}), correct is {exp_str}. "
            f"Q-gap={q_gap:.4f} {'— very low, model uncertain.' if q_gap < 0.5 else '.'} "
            f"Anomaly={anomaly:.4f}. Review sensor boundary conditions."
        )

    desc = {
        2: (f"Emergency Shutdown confirmed for Mixture gas (gas_id=2) — highest hazard class. "
            f"Anomaly={anomaly:.3f} {'(elevated)' if anomaly>0.4 else '(low — model uses sensor features not anomaly)'}. "
            f"Q(Emergency)={q[4]:.3f}, gap={q_gap:.3f}. Conf={policy_conf*100:.1f}%. "
            f"Protocol: evacuate, isolate fuel lines, emergency services."),
        1: (f"Raise Alarm confirmed for Smoke (gas_id=1). Combustion byproducts present. "
            f"Anomaly={anomaly:.3f}. Q(Alarm)={q[3]:.3f}, gap={q_gap:.3f}. Conf={policy_conf*100:.1f}%. "
            f"Protocol: audible alarm, evacuation, check ignition sources."),
        0: (f"Monitor confirmed (NoGas, gas_id=0). Anomaly={anomaly:.3f} within baseline. "
            f"Q(Monitor)={q[0]:.3f}. No hazardous signature. System continues normal surveillance."),
        3: (f"VOC response (Perfume, gas_id=3). Anomaly={anomaly:.3f} — "
            f"{'ventilation advised (high anomaly)' if anomaly>0.5 else 'log-only response (low anomaly)'}. "
            f"Agent chose {act} (action {action}). Conf={policy_conf*100:.1f}%. Non-hazardous."),
    }
    return desc.get(gas_id, f"Action {action} ({act}). Anomaly={anomaly:.3f}.")


def generate_critique(gas_id, action, anomaly, q_values, is_correct, policy_conf):
    q     = np.array(q_values)
    q_gap = float(q.max() - np.sort(q)[-2]) if len(q) > 1 else 0.0
    gas   = GAS_NAMES.get(gas_id, "Unknown")
    parts = []

    if policy_conf < 0.15:
        parts.append(f"CONFIDENCE WARNING: {policy_conf:.4f} < 0.15. Q-gap={q_gap:.4f}. Escalate or flag for human review.")
    elif policy_conf < 0.30:
        parts.append(f"Moderate confidence ({policy_conf:.4f}). Monitor this class boundary.")
    else:
        parts.append(f"Confidence {policy_conf:.4f} — acceptable for autonomous operation.")

    if is_correct:
        if gas_id == 2 and action == 4:
            parts.append("Maximum-severity response for maximum-hazard gas. Correct severity discrimination.")
        elif gas_id == 0 and action == 0:
            parts.append("Conservative no-false-alarm response. Sensor calibration check recommended periodically.")
        else:
            parts.append("Policy-correct decision. No safety concerns.")
    else:
        if gas_id in DANGER_GAS_IDS:
            parts.append(f"CRITICAL: {gas} misclassified. Wrong action {action} risks delayed emergency response. Retrain with higher danger-miss penalty.")
        else:
            parts.append(f"Non-critical error for {gas}. Affects operational efficiency, not safety.")

    if gas_id == 2 and anomaly < 0.20:
        parts.append(f"Note: Low anomaly {anomaly:.3f} for Mixture is expected — LSTM reconstructs it well. Model correctly uses sensor features.")

    return " | ".join(parts)


# =========================================================
# DATA HELPERS
# =========================================================
def infer_label_col(df):
    for c in ["label","Label","class","Gas"]:
        if c in df.columns: return c
    return None

def infer_anomaly_col(df):
    for c in ["anomaly","anomaly_norm","anomaly_normalized","anom"]:
        if c in df.columns: return c
    return None

def build_state_cols(df):
    acol = infer_anomaly_col(df)
    if not acol: raise ValueError("No anomaly column.")
    miss = [c for c in RAW_SENSOR_COLS+DELTA_COLS+STD_COLS if c not in df.columns]
    if miss: raise ValueError(f"Missing: {miss}")
    return acol, [acol]+RAW_SENSOR_COLS+DELTA_COLS+STD_COLS


# =========================================================
# PLOTLY THEME
# =========================================================
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,20,32,0.8)",
    font=dict(family="Share Tech Mono, monospace", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zeroline=False),
    yaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f", borderwidth=1),
)


def q_bar_chart(q_values, action, label):
    colors = [ACTION_COLORS.get(i, "#64748b") for i in range(5)]
    opacities = [1.0 if i == action else 0.35 for i in range(5)]
    fig = go.Figure(go.Bar(
        x=[f"A{i}\n{ACTIONS[i][:8]}" for i in range(5)],
        y=q_values,
        marker_color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},{op})"
                      for c, op in zip(colors, opacities)],
        marker_line_color=colors,
        marker_line_width=1.5,
        text=[f"{v:.2f}" for v in q_values],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=f"Q-Values — {label}", font=dict(size=13, color="#f59e0b")), height=260)
    fig.update_yaxis(title="Q-Value")
    return fig


def action_distribution_chart(logs):
    counts = pd.Series([r["action"] for r in logs]).value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[f"{i}: {ACTIONS[i]}" for i in counts.index],
        y=counts.values,
        marker_color=[ACTION_COLORS.get(i, "#64748b") for i in counts.index],
        marker_line_color="#1e3a5f", marker_line_width=1,
        text=counts.values, textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="Action Distribution", font=dict(size=13,color="#f59e0b")), height=280)
    return fig


def anomaly_scatter(logs):
    df_plot = pd.DataFrame(logs)
    fig = go.Figure()
    for lbl, color in GAS_COLORS.items():
        sub = df_plot[df_plot["true_label"] == lbl]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["anomaly_used"], y=sub["action"],
            mode="markers",
            name=lbl,
            marker=dict(color=color, size=6, opacity=0.7, line=dict(color=color, width=0.5)),
        ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Anomaly Score vs Action", font=dict(size=13, color="#f59e0b")),
        height=300,
        yaxis=dict(tickmode="array", tickvals=list(range(5)),
                   ticktext=[f"{i}:{ACTIONS[i][:10]}" for i in range(5)],
                   gridcolor="#1e3a5f"),
        xaxis=dict(title="Anomaly Score", gridcolor="#1e3a5f"),
    )
    return fig


def confidence_histogram(logs):
    df_plot = pd.DataFrame(logs)
    fig = go.Figure()
    for is_corr, color, name in [(True,"#22c55e","Correct"), (False,"#ef4444","Wrong")]:
        sub = df_plot[df_plot["is_correct"] == is_corr]["policy_confidence"]
        if sub.empty: continue
        fig.add_trace(go.Histogram(x=sub, name=name, marker_color=color, opacity=0.7,
                                   nbinsx=20, marker_line_color="#1e3a5f", marker_line_width=0.5))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="Confidence Distribution", font=dict(size=13,color="#f59e0b")), height=260, barmode="overlay")
    return fig


def accuracy_gauge(accuracy, label):
    color = "#22c55e" if accuracy >= 0.95 else ("#f59e0b" if accuracy >= 0.80 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy * 100,
        number=dict(suffix="%", font=dict(size=28, color=color, family="Rajdhani")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#64748b", tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.3),
            bgcolor="#0d1420",
            bordercolor="#1e3a5f",
            borderwidth=1,
            steps=[
                dict(range=[0,80], color="#1a2744"),
                dict(range=[80,95], color="#1e3a1f"),
                dict(range=[95,100], color="#1a3a20"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=0.8, value=accuracy*100),
        ),
        title=dict(text=label, font=dict(size=12, color="#64748b", family="Share Tech Mono")),
        domain=dict(x=[0,1], y=[0,1]),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(l=20,r=20,t=30,b=10))
    return fig


# =========================================================
# MAIN APP
# =========================================================
inject_css()

# ── Header ───────────────────────────────────────────────
status_dot = '<span class="status-dot status-online"></span>' if IMPORTS_OK else '<span class="status-dot status-offline"></span>'
status_text = "SYSTEM ONLINE" if IMPORTS_OK else "IMPORT ERROR"

st.markdown(f"""
<div class="hud-header">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <p class="hud-title">⚡ GasSafe AI — Control Dashboard</p>
      <p class="hud-subtitle">DUELING DOUBLE DQN · AUTONOMOUS SAFETY AGENT · REAL-TIME MONITORING</p>
    </div>
    <div style="text-align:right;">
      <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#64748b; letter-spacing:0.15em;">
        {status_dot}<span style="color:#{'22c55e' if IMPORTS_OK else 'ef4444'};">{status_text}</span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#475569; margin-top:4px;" id="clock">
        GAS STATION SAFETY INTELLIGENCE v2.0
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not IMPORTS_OK:
    st.error(f"Failed to import model tools: {IMPORT_ERROR}")
    st.info("Make sure you are running from the project root directory with the virtual environment activated.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙ Configuration</p>', unsafe_allow_html=True)

    model_path = st.text_input("DQN Model Path", value="models/DeepQmodel.pth")
    csv_path   = st.text_input("Test CSV Path", value="test_df_processed.csv")

    st.markdown('<p class="section-header">▶ Run Settings</p>', unsafe_allow_html=True)
    max_steps    = st.slider("Max steps per class", 10, 400, 400, 10)
    use_mc       = st.checkbox("MC Dropout (slower)", value=False)
    show_expl    = st.checkbox("Show Explanations", value=True)
    show_crit    = st.checkbox("Show Critique", value=True)

    st.markdown('<p class="section-header">◈ Filter</p>', unsafe_allow_html=True)
    filter_label  = st.selectbox("Filter by label", ["All", "NoGas", "Smoke", "Mixture", "Perfume"])
    show_only_wrong = st.checkbox("Show only wrong predictions", value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#475569; line-height:1.8;">
    GAS_MAP<br>
    ├ NoGas   → 0 → Monitor<br>
    ├ Smoke   → 1 → Raise Alarm<br>
    ├ Mixture → 2 → Emergency<br>
    └ Perfume → 3 → Inc. Sampling
    </div>
    """, unsafe_allow_html=True)

# ── Load model & CSV ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    return DecisionTool(path, device="cpu", mc_dropout_samples=5, window_size=20)

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

col_load1, col_load2 = st.columns([3, 1])
with col_load1:
    run_btn_placeholder = st.empty()
with col_load2:
    model_status = st.empty()

model_ok = Path(model_path).exists()
csv_ok   = Path(csv_path).exists()

if not model_ok:
    st.error(f"Model not found: {model_path}")
if not csv_ok:
    st.error(f"CSV not found: {csv_path}")

run_clicked = run_btn_placeholder.button(
    "▶  RUN VERIFICATION" if (model_ok and csv_ok) else "⚠ FILES NOT FOUND",
    disabled=not (model_ok and csv_ok),
    use_container_width=True,
)

# ── If CSV exists, show static overview immediately ───────
if csv_ok:
    try:
        df_raw = load_csv(csv_path)
        label_col = infer_label_col(df_raw)
        if label_col:
            counts = df_raw[label_col].value_counts()
            c1,c2,c3,c4 = st.columns(4)
            for col_, (lbl, cnt) in zip([c1,c2,c3,c4], counts.items()):
                color = GAS_COLORS.get(lbl, "#64748b")
                with col_:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-card-label">{lbl}</div>
                      <div class="metric-card-value" style="color:{color};">{cnt}</div>
                      <div class="metric-card-sub">test samples</div>
                    </div>""", unsafe_allow_html=True)
    except Exception:
        pass

st.markdown("---")

# ── Run inference ─────────────────────────────────────────
if run_clicked or ("results" in st.session_state):

    if run_clicked or "results" not in st.session_state:
        # Load
        with st.spinner("Loading model weights..."):
            model = load_model(model_path)
        with st.spinner("Loading test data..."):
            df = load_csv(csv_path)

        label_col = infer_label_col(df)
        if not label_col:
            st.error("No label column found in CSV.")
            st.stop()

        try:
            anomaly_col, state_cols = build_state_cols(df)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        available_labels = sorted(df[label_col].dropna().unique().tolist())
        unknown = [l for l in available_labels if l not in GAS_MAP]
        if unknown:
            st.error(f"Labels not in GAS_MAP: {unknown}")
            st.stop()

        # Run inference
        all_logs = []
        progress_bar = st.progress(0, text="Running inference...")
        total_steps  = 0

        for li, label in enumerate(available_labels):
            df_lbl      = df[df[label_col] == label].reset_index(drop=True)
            steps       = min(len(df_lbl), max_steps)
            true_gas_id = GAS_MAP[label]
            expected    = CORRECT_ACTIONS[true_gas_id]
            exp_str     = " or ".join(f"{a}={ACTIONS[a]}" for a in expected)

            for step in range(steps):
                row    = df_lbl.iloc[step]
                state  = row[state_cols].values.astype(np.float32)
                t0     = time.time()
                action, q_values, q_std, policy_conf = model.decide(state=state, use_mc_dropout=use_mc)
                latency_ms = (time.time() - t0) * 1000

                correct = is_correct_action(true_gas_id, action)
                reward  = compute_reward(state=state, action=int(action), gas_id=true_gas_id, anomaly=float(state[0]))
                expl = generate_explanation(true_gas_id, int(action), float(state[0]), q_values.tolist(), correct, float(policy_conf)) if show_expl else ""
                crit = generate_critique(true_gas_id, int(action), float(state[0]), q_values.tolist(), correct, float(policy_conf)) if show_crit else ""

                all_logs.append({
                    "true_label": label, "true_gas_id": true_gas_id, "step": step,
                    "action": int(action), "action_name": ACTIONS[int(action)],
                    "is_correct": correct, "expected_actions": exp_str, "reward": reward,
                    "policy_confidence": float(policy_conf), "latency_ms": float(latency_ms),
                    "q0":float(q_values[0]),"q1":float(q_values[1]),"q2":float(q_values[2]),
                    "q3":float(q_values[3]),"q4":float(q_values[4]),
                    "anomaly_used": float(state[0]),
                    "explanation": expl, "critique": crit,
                })
                total_steps += 1
                progress_bar.progress(
                    (li * max_steps + step + 1) / (len(available_labels) * max_steps),
                    text=f"Processing {label} — step {step+1}/{steps}"
                )

        progress_bar.empty()
        st.session_state["results"] = all_logs
        st.success(f"✓ Inference complete — {total_steps} samples evaluated")

    # ── Use cached results ────────────────────────────────
    all_logs = st.session_state["results"]

    # Apply filter
    filtered = all_logs
    if filter_label != "All":
        filtered = [r for r in filtered if r["true_label"] == filter_label]
    if show_only_wrong:
        filtered = [r for r in filtered if not r["is_correct"]]

    # ── TABS ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["◈ OVERVIEW", "◈ PER-LABEL ANALYSIS", "◈ STEP DETAIL", "◈ CHARTS"])

    # ============================================================
    # TAB 1: OVERVIEW
    # ============================================================
    with tab1:
        st.markdown('<p class="section-header">◈ System Performance Metrics</p>', unsafe_allow_html=True)

        total_n   = len(all_logs)
        correct_n = sum(r["is_correct"] for r in all_logs)
        danger_r  = [r for r in all_logs if r["true_gas_id"] in DANGER_GAS_IDS]
        danger_m  = sum(1 for r in danger_r if r["action"] == 0)
        nogas_r   = [r for r in all_logs if r["true_gas_id"] == 0]
        false_a   = sum(1 for r in nogas_r if r["action"] >= 3)
        avg_lat   = np.mean([r["latency_ms"] for r in all_logs])
        avg_conf  = np.mean([r["policy_confidence"] for r in all_logs])
        overall_acc = correct_n / total_n if total_n else 0

        # Big metrics row
        m1,m2,m3,m4,m5 = st.columns(5)
        metrics = [
            ("OVERALL ACCURACY", f"{overall_acc*100:.2f}%", f"{correct_n}/{total_n}", "#22c55e" if overall_acc>=0.95 else "#f59e0b"),
            ("DANGER MISS RATE", f"{danger_m/max(len(danger_r),1)*100:.2f}%", f"{danger_m}/{len(danger_r)} hazardous", "#22c55e" if danger_m==0 else "#ef4444"),
            ("FALSE ALARM RATE", f"{false_a/max(len(nogas_r),1)*100:.2f}%", f"{false_a}/{len(nogas_r)} NoGas", "#22c55e" if false_a==0 else "#f59e0b"),
            ("AVG LATENCY", f"{avg_lat:.2f}ms", "per inference", "#3b82f6"),
            ("AVG CONFIDENCE", f"{avg_conf:.4f}", "policy confidence", "#a78bfa"),
        ]
        for col_, (label_, val_, sub_, color_) in zip([m1,m2,m3,m4,m5], metrics):
            with col_:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-card-label">{label_}</div>
                  <div class="metric-card-value" style="color:{color_};">{val_}</div>
                  <div class="metric-card-sub">{sub_}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Accuracy gauges
        st.markdown('<p class="section-header">◈ Per-Label Accuracy</p>', unsafe_allow_html=True)
        g_cols = st.columns(4)
        for i, lbl in enumerate(["NoGas","Smoke","Mixture","Perfume"]):
            lbl_logs = [r for r in all_logs if r["true_label"] == lbl]
            if not lbl_logs: continue
            acc = sum(r["is_correct"] for r in lbl_logs) / len(lbl_logs)
            with g_cols[i]:
                st.plotly_chart(accuracy_gauge(acc, lbl), use_container_width=True)

        # Gas-id mapping table
        st.markdown('<p class="section-header">◈ Policy Mapping</p>', unsafe_allow_html=True)
        map_cols = st.columns(4)
        mapping = [
            ("NoGas", 0, "Monitor", "#22c55e"),
            ("Smoke", 1, "Raise Alarm", "#f97316"),
            ("Mixture", 2, "Emergency Shutdown", "#ef4444"),
            ("Perfume", 3, "Inc. Sampling / Req. Verification", "#3b82f6"),
        ]
        for col_, (lbl_, gid_, act_, clr_) in zip(map_cols, mapping):
            lbl_logs = [r for r in all_logs if r["true_label"] == lbl_]
            acc = sum(r["is_correct"] for r in lbl_logs)/len(lbl_logs) if lbl_logs else 0
            with col_:
                st.markdown(f"""
                <div class="label-card">
                  <div class="label-card-title" style="color:{clr_};">{lbl_}</div>
                  <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#475569; margin:4px 0;">
                    gas_id = {gid_}
                  </div>
                  <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#94a3b8;">
                    → {act_}
                  </div>
                  <div style="margin-top:10px;">
                    <span style="font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:700; color:{'#22c55e' if acc>=0.95 else '#f59e0b'};">
                      {acc*100:.1f}%
                    </span>
                    <span style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#475569; margin-left:6px;">
                      ({sum(r['is_correct'] for r in lbl_logs)}/{len(lbl_logs)})
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ============================================================
    # TAB 2: PER-LABEL ANALYSIS
    # ============================================================
    with tab2:
        for lbl in ["Mixture","NoGas","Perfume","Smoke"]:
            lbl_logs = [r for r in all_logs if r["true_label"] == lbl]
            if not lbl_logs: continue
            acc     = sum(r["is_correct"] for r in lbl_logs)/len(lbl_logs)
            gid     = GAS_MAP[lbl]
            clr     = GAS_COLORS[lbl]
            avg_anom= np.mean([r["anomaly_used"] for r in lbl_logs])
            avg_c   = np.mean([r["policy_confidence"] for r in lbl_logs])
            avg_r   = np.mean([r["reward"] for r in lbl_logs])
            dom_act = int(pd.Series([r["action"] for r in lbl_logs]).value_counts().index[0])

            with st.expander(f"◈ {lbl.upper()}  —  gas_id={gid}  —  Accuracy: {acc*100:.1f}%  ({sum(r['is_correct'] for r in lbl_logs)}/{len(lbl_logs)})", expanded=(lbl=="Mixture")):
                lc1, lc2, lc3, lc4 = st.columns(4)
                for col_, (lab_, val_) in zip([lc1,lc2,lc3,lc4], [
                    ("Accuracy", f"{acc*100:.2f}%"),
                    ("Avg Anomaly", f"{avg_anom:.4f}"),
                    ("Avg Confidence", f"{avg_c:.4f}"),
                    ("Mean Reward", f"{avg_r:.4f}"),
                ]):
                    with col_:
                        st.markdown(f"""<div class="metric-card">
                          <div class="metric-card-label">{lab_}</div>
                          <div class="metric-card-value" style="font-size:1.4rem; color:{clr};">{val_}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Q-value chart from first correct sample
                correct_samples = [r for r in lbl_logs if r["is_correct"]]
                if correct_samples:
                    r0 = correct_samples[0]
                    st.plotly_chart(q_bar_chart([r0["q0"],r0["q1"],r0["q2"],r0["q3"],r0["q4"]], r0["action"], f"{lbl} — Sample Q-Values"), use_container_width=True)

                # Anomaly distribution for this class
                anoms = [r["anomaly_used"] for r in lbl_logs]
                fig_hist = go.Figure(go.Histogram(
                    x=anoms, nbinsx=30,
                    marker_color=clr, opacity=0.8,
                    marker_line_color="#1e3a5f", marker_line_width=0.5,
                ))
                fig_hist.update_layout(**PLOT_LAYOUT, title=dict(text=f"{lbl} — Anomaly Distribution", font=dict(size=12,color="#f59e0b")), height=220)
                st.plotly_chart(fig_hist, use_container_width=True)

                # Wrong predictions
                wrong = [r for r in lbl_logs if not r["is_correct"]]
                if wrong:
                    st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace; font-size:11px; color:#ef4444; margin:8px 0;">⚠ {len(wrong)} wrong predictions</div>', unsafe_allow_html=True)
                    wrong_df = pd.DataFrame([{
                        "Step": r["step"], "Action": r["action_name"],
                        "Expected": r["expected_actions"], "Anomaly": round(r["anomaly_used"],4),
                        "Confidence": round(r["policy_confidence"],4),
                    } for r in wrong[:10]])
                    st.dataframe(wrong_df, use_container_width=True, hide_index=True)

    # ============================================================
    # TAB 3: STEP DETAIL + EXPLANATIONS
    # ============================================================
    with tab3:
        st.markdown(f'<p class="section-header">◈ Step-Level Detail — {len(filtered)} records</p>', unsafe_allow_html=True)

        if not filtered:
            st.info("No records match the current filter.")
        else:
            # Summary table
            table_data = []
            for r in filtered[:200]:
                table_data.append({
                    "Label": r["true_label"],
                    "Step": r["step"],
                    "Action": r["action_name"],
                    "✓": "✓" if r["is_correct"] else "✗",
                    "Expected": r["expected_actions"],
                    "Anomaly": round(r["anomaly_used"], 4),
                    "Confidence": round(r["policy_confidence"], 4),
                    "Reward": round(r["reward"], 4),
                    "Q0":round(r["q0"],2),"Q1":round(r["q1"],2),"Q2":round(r["q2"],2),
                    "Q3":round(r["q3"],2),"Q4":round(r["q4"],2),
                    "Latency(ms)": round(r["latency_ms"],3),
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True, height=300)

            st.markdown("<br>", unsafe_allow_html=True)

            # Explanation cards
            if show_expl or show_crit:
                st.markdown('<p class="section-header">◈ Decision Explanations & Critique</p>', unsafe_allow_html=True)
                display_n = st.slider("Show explanations for N steps", 1, min(50, len(filtered)), min(20, len(filtered)))

                for r in filtered[:display_n]:
                    tick_html = '<span class="tick-correct">✓</span>' if r["is_correct"] else '<span class="tick-wrong">✗</span>'
                    border_class = "correct-row" if r["is_correct"] else "wrong-row"
                    act_color = ACTION_COLORS.get(r["action"], "#64748b")
                    gas_color = GAS_COLORS.get(r["true_label"], "#64748b")

                    st.markdown(f"""
                    <div class="label-card {border_class}" style="margin-bottom:0.5rem;">
                      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div style="display:flex; align-items:center; gap:10px;">
                          {tick_html}
                          <span style="font-family:'Rajdhani',sans-serif; font-weight:600; font-size:0.95rem;">
                            Step {r['step']} — {r['true_label']}
                          </span>
                          <span class="gas-badge" style="background:rgba({'34,197,94' if r['true_label']=='NoGas' else '239,68,68' if r['true_label']=='Mixture' else '249,115,22' if r['true_label']=='Smoke' else '59,130,246'},0.15); color:{gas_color}; border:1px solid {gas_color}40;">
                            gas_id={r['true_gas_id']}
                          </span>
                          <span class="action-badge" style="background:rgba({'34,197,94' if r['is_correct'] else '239,68,68'},0.15); color:{'#22c55e' if r['is_correct'] else '#ef4444'}; border:1px solid {'#22c55e' if r['is_correct'] else '#ef4444'}40;">
                            {r['action_name']}
                          </span>
                        </div>
                        <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#475569;">
                          anomaly={r['anomaly_used']:.4f} · conf={r['policy_confidence']:.4f} · reward={r['reward']:.3f} · {r['latency_ms']:.2f}ms
                        </div>
                      </div>
                    """, unsafe_allow_html=True)

                    if show_expl and r.get("explanation"):
                        st.markdown(f'<div class="expl-box">▸ EXPLANATION: {r["explanation"]}</div>', unsafe_allow_html=True)
                    if show_crit and r.get("critique"):
                        st.markdown(f'<div class="crit-box">▸ CRITIQUE: {r["critique"]}</div>', unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # TAB 4: CHARTS
    # ============================================================
    with tab4:
        st.markdown('<p class="section-header">◈ Policy Analytics</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(action_distribution_chart(filtered if filtered else all_logs), use_container_width=True)
        with c2:
            st.plotly_chart(confidence_histogram(filtered if filtered else all_logs), use_container_width=True)

        st.plotly_chart(anomaly_scatter(filtered if filtered else all_logs), use_container_width=True)

        # Q-value comparison across all labels
        st.markdown('<p class="section-header">◈ Mean Q-Values by Gas Class</p>', unsafe_allow_html=True)
        q_fig = go.Figure()
        for lbl in ["Mixture","NoGas","Perfume","Smoke"]:
            lbl_logs = [r for r in all_logs if r["true_label"] == lbl]
            if not lbl_logs: continue
            mean_q = [np.mean([r[f"q{i}"] for r in lbl_logs]) for i in range(5)]
            q_fig.add_trace(go.Bar(
                name=lbl,
                x=[f"A{i}:{ACTIONS[i][:8]}" for i in range(5)],
                y=mean_q,
                marker_color=GAS_COLORS[lbl], opacity=0.85,
                marker_line_color="#1e3a5f", marker_line_width=0.5,
            ))
        q_fig.update_layout(**PLOT_LAYOUT, title=dict(text="Mean Q-Values per Class", font=dict(size=13,color="#f59e0b")), height=320, barmode="group")
        st.plotly_chart(q_fig, use_container_width=True)

        # Latency distribution
        lat_fig = go.Figure(go.Histogram(
            x=[r["latency_ms"] for r in all_logs],
            nbinsx=40,
            marker_color="#3b82f6", opacity=0.8,
            marker_line_color="#1e3a5f", marker_line_width=0.5,
        ))
        lat_fig.update_layout(**PLOT_LAYOUT, title=dict(text="Inference Latency Distribution (ms)", font=dict(size=13,color="#f59e0b")), height=260)
        st.plotly_chart(lat_fig, use_container_width=True)

    # ── Download buttons ──────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">◈ Export Results</p>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        df_log = pd.DataFrame(all_logs)
        st.download_button("⬇ Download Detailed Log (CSV)", df_log.to_csv(index=False).encode(),
                           "evaluation_log.csv", "text/csv", use_container_width=True)
    with d2:
        summary_rows = []
        for lbl in GAS_MAP:
            lbl_logs = [r for r in all_logs if r["true_label"] == lbl]
            if not lbl_logs: continue
            acc = sum(r["is_correct"] for r in lbl_logs)/len(lbl_logs)
            summary_rows.append({"label":lbl, "gas_id":GAS_MAP[lbl], "accuracy":round(acc,4),
                                  "correct":sum(r["is_correct"] for r in lbl_logs), "total":len(lbl_logs),
                                  "mean_reward":round(np.mean([r["reward"] for r in lbl_logs]),4),
                                  "mean_conf":round(np.mean([r["policy_confidence"] for r in lbl_logs]),4),
                                  "avg_latency_ms":round(np.mean([r["latency_ms"] for r in lbl_logs]),3)})
        df_sum = pd.DataFrame(summary_rows)
        st.download_button("⬇ Download Summary (CSV)", df_sum.to_csv(index=False).encode(),
                           "episode_summary.csv", "text/csv", use_container_width=True)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
      <div style="font-family:'Share Tech Mono',monospace; font-size:13px; color:#475569; letter-spacing:0.2em; margin-bottom:2rem;">
        SYSTEM READY — AWAITING VERIFICATION RUN
      </div>
      <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; color:#334155; line-height:2;">
        Configure paths in the sidebar<br>
        Set step count and options<br>
        Click RUN VERIFICATION to start
      </div>
    </div>
    """, unsafe_allow_html=True)