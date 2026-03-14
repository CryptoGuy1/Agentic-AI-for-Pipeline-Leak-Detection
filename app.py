"""
Methane AI Control Room  v6.0
==============================
  • Tab bar rendered FIRST — top of every page
  • All cards have rounded corners (12 px)
  • Light / Dark with full colour swap for visibility
  • Classical AI Analyst page (chat + side panel)
  • Auto Gemma3:1b explanation on Overview after each cycle
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from PIL import Image

try:
    import requests as _req
except ImportError:
    _req = None

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Methane AI Control Room",
    page_icon  = "⚠️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
ACTIONS      = ["Monitor","Increase Sampling","Verify","Raise Alarm","Shutdown"]
RISK_LEVELS  = ["Low","Moderate","High","Critical"]
N_SENSORS    = 7
SENSOR_NAMES = [f"S{i}" for i in range(1, N_SENSORS + 1)]
SENSOR_ZONES = ["Zone A","Zone A","Zone B","Zone B","Zone C","Zone C","Zone D"]
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:1b"

TABS = [
    ("🏠","Overview"),
    ("📈","Telemetry"),
    ("🔬","Sensors"),
    ("🤖","RL Decisions"),
    ("📐","RL Metrics"),
    ("📊","Analytics"),
    ("🚨","Incidents"),
    ("🔭","Correlation"),
    ("🧠","AI Analyst"),
]

# ──────────────────────────────────────────────────────────────────────────────
# THEMES  — full colour set for dark + light
# ──────────────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": dict(
        bg          = "#0d1117",
        surface     = "#161b22",
        surface2    = "#21262d",
        border      = "#30363d",
        text        = "#e6edf3",
        text_sub    = "#8b949e",
        accent      = "#3fb950",   # green
        accent_dim  = "#1a7f37",
        blue        = "#58a6ff",
        orange      = "#f0883e",
        red         = "#f85149",
        yellow      = "#e3b341",
        risk_low    = "#3fb950",
        risk_med    = "#58a6ff",
        risk_high   = "#e3b341",
        risk_crit   = "#f85149",
        bubble_user = "#0d2818",
        bubble_ai   = "#0d1e35",
        shadow      = "0 3px 12px rgba(0,0,0,.45)",
        nav_active_bg = "#1f2d1f",
    ),
    "light": dict(
        bg          = "#f6f8fa",
        surface     = "#ffffff",
        surface2    = "#eaeef2",
        border      = "#d0d7de",
        text        = "#1f2328",
        text_sub    = "#57606a",
        accent      = "#1a7f37",   # darker green for contrast on white
        accent_dim  = "#d1f0d9",
        blue        = "#0969da",
        orange      = "#bc4c00",
        red         = "#cf222e",
        yellow      = "#9a6700",
        risk_low    = "#1a7f37",
        risk_med    = "#0969da",
        risk_high   = "#9a6700",
        risk_crit   = "#cf222e",
        bubble_user = "#dafbe1",
        bubble_ai   = "#ddf4ff",
        shadow      = "0 2px 8px rgba(140,149,159,.2)",
        nav_active_bg = "#dafbe1",
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    history=[], incidents=[], cycle=0, last_result=None,
    history_loaded_file=None, chat_messages=[], active_tab=0,
    theme="dark", auto_explanation="",
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

T = THEMES[st.session_state.theme]
IS_DARK = st.session_state.theme == "dark"

def rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-char hex color + alpha float to rgba() string for Plotly."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC CSS  — rebuilt on every theme flip
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ═══════════════  BASE  ═══════════════════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background: {T['bg']} !important;
    color: {T['text']} !important;
}}
[data-testid="block-container"] {{
    padding: 0 clamp(8px, 2vw, 24px) !important;
    max-width: 100% !important;
    background: {T['bg']} !important;
}}
/* Hide Streamlit header so nav bar is truly at the top */
.stAppHeader, header[data-testid="stHeader"] {{ display: none !important; }}

/* ═══════════════  GLOBAL OVERFLOW GUARD  ═════════════════════════════════ */
/* Every column and flex child must never overflow its container */
[data-testid="stColumns"],
[data-testid="column"] {{
    min-width: 0 !important;
    overflow: hidden !important;
    gap: clamp(4px, 1vw, 14px) !important;
}}
[data-testid="stColumns"] > div,
[data-testid="column"] > div {{
    min-width: 0 !important;
    overflow: hidden !important;
}}
/* All plain text inside the app */
p, span, div, h1, h2, h3, h4, h5, h6, li {{
    min-width: 0;
    max-width: 100%;
}}

/* ═══════════════  SIDEBAR  ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {{
    background: {T['surface']} !important;
    border-right: 1px solid {T['border']} !important;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{ color: {T['text']} !important; }}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] input {{
    background: {T['surface2']} !important;
    border-color: {T['border']} !important;
    color: {T['text']} !important;
    border-radius: 8px !important;
}}

/* ═══════════════  METRIC CARDS  ══════════════════════════════════════════ */
div[data-testid="metric-container"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 12px !important;
    padding: clamp(8px, 1.2vw, 16px) clamp(8px, 1.2vw, 18px) !important;
    box-shadow: {T['shadow']} !important;
    min-width: 0 !important;
    overflow: hidden !important;
}}
div[data-testid="metric-container"] label {{
    color: {T['text_sub']} !important;
    font-size: clamp(0.55rem, 0.9vw, 0.7rem) !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    display: block !important;
    max-width: 100% !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.95rem, 1.8vw, 1.55rem) !important;
    color: {T['text']} !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    display: block !important;
    max-width: 100% !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] span {{
    font-size: clamp(0.6rem, 0.9vw, 0.75rem) !important;
    white-space: nowrap !important;
}}

/* ═══════════════  CHARTS & DATA  ════════════════════════════════════════ */
[data-testid="stPlotlyChart"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: {T['shadow']} !important;
    padding: 4px !important;
    min-width: 0 !important;
    width: 100% !important;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {T['border']} !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    min-width: 0 !important;
    width: 100% !important;
}}

/* ═══════════════  TEXT INPUT  ═══════════════════════════════════════════ */
.stTextInput > div > div > input {{
    background: {T['surface']} !important;
    border: 1.5px solid {T['border']} !important;
    border-radius: 10px !important;
    color: {T['text']} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: clamp(0.78rem, 1.2vw, 0.9rem) !important;
    padding: clamp(7px, 1vw, 10px) clamp(8px, 1.2vw, 14px) !important;
    width: 100% !important;
    transition: border-color .15s;
}}
.stTextInput > div > div > input:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 3px {T['accent']}33 !important;
    outline: none !important;
}}
.stTextInput > div > div > input::placeholder {{
    color: {T['text_sub']} !important;
    opacity: 0.7;
}}

/* ═══════════════  FORM LABELS  ══════════════════════════════════════════ */
.stSelectbox label, .stSlider label,
.stCheckbox label, .stFileUploader label {{
    color: {T['text_sub']} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}

/* ═══════════════  SCROLLBAR  ════════════════════════════════════════════ */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {T['surface2']}; }}
::-webkit-scrollbar-thumb {{ background: {T['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {T['accent']}; }}

/* ═══════════════  BUTTON PALETTE  ══════════════════════════════════════ */
/* Default */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-size: clamp(0.7rem, 1vw, 0.82rem) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: clamp(6px, 0.8vw, 9px) clamp(8px, 1.2vw, 16px) !important;
    transition: all .18s ease !important;
    line-height: 1.4 !important;
    background: transparent !important;
    border: 1.5px solid {T['border']} !important;
    color: {T['text_sub']} !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    border-color: {T['accent']} !important;
    color: {T['accent']} !important;
    background: {T['accent']}18 !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}}
/* NAV inactive */
.nav-tab .stButton > button {{
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: {T['text_sub']} !important;
    font-size: clamp(0.62rem, 0.9vw, 0.78rem) !important;
    font-weight: 500 !important;
    padding: 10px clamp(2px, 0.5vw, 6px) !important;
    transform: none !important;
    box-shadow: none !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}}
.nav-tab .stButton > button:hover {{
    background: {T['surface2']} !important;
    border-bottom: 2px solid {T['accent']}88 !important;
    color: {T['accent']} !important;
    transform: none !important;
    box-shadow: none !important;
}}
/* NAV active */
.nav-tab-active .stButton > button {{
    background: {T['nav_active_bg']} !important;
    border: none !important;
    border-bottom: 2px solid {T['accent']} !important;
    border-radius: 0 !important;
    color: {T['accent']} !important;
    font-weight: 700 !important;
    font-size: clamp(0.62rem, 0.9vw, 0.78rem) !important;
    padding: 10px clamp(2px, 0.5vw, 6px) !important;
    transform: none !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}
/* RUN */
.btn-run .stButton > button {{
    background: {T['accent']} !important;
    border-color: {T['accent']} !important;
    color: #fff !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px {T['accent']}44 !important;
}}
.btn-run .stButton > button:hover {{
    opacity: .92 !important;
    transform: translateY(-1px) !important;
    color: #fff !important;
}}
/* RESET */
.btn-reset .stButton > button {{
    border-color: {T['border']} !important;
    color: {T['text_sub']} !important;
    border-radius: 9px !important;
}}
.btn-reset .stButton > button:hover {{
    border-color: {T['red']} !important;
    color: {T['red']} !important;
    background: {T['red']}18 !important;
    transform: none !important;
}}
/* THEME */
.btn-theme .stButton > button {{
    background: {T['surface2']} !important;
    border: 1.5px solid {T['border']} !important;
    color: {T['text']} !important;
    border-radius: 20px !important;
}}
.btn-theme .stButton > button:hover {{
    background: {T['accent']} !important;
    border-color: {T['accent']} !important;
    color: #fff !important;
    transform: none !important;
}}
/* SEND */
.btn-send .stButton > button {{
    background: linear-gradient(135deg, {T['accent']}, {T['blue']}) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px {T['accent']}44 !important;
}}
.btn-send .stButton > button:hover {{
    opacity: .92 !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
}}
/* CLEAR */
.btn-clear .stButton > button {{
    border: 1.5px solid {T['border']} !important;
    color: {T['text_sub']} !important;
    border-radius: 9px !important;
}}
.btn-clear .stButton > button:hover {{
    border-color: {T['red']} !important;
    color: {T['red']} !important;
    background: {T['red']}18 !important;
    transform: none !important;
}}
/* EXPLAIN */
.btn-explain .stButton > button {{
    background: {T['blue']}18 !important;
    border: 1.5px solid {T['blue']} !important;
    color: {T['blue']} !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}}
.btn-explain .stButton > button:hover {{
    background: {T['blue']} !important;
    color: #fff !important;
    transform: none !important;
}}
/* NEW CHAT */
.btn-newchat .stButton > button {{
    border: 1.5px solid {T['border']} !important;
    color: {T['text_sub']} !important;
    border-radius: 9px !important;
}}
.btn-newchat .stButton > button:hover {{
    border-color: {T['accent']} !important;
    color: {T['accent']} !important;
    background: transparent !important;
    transform: none !important;
}}
/* DOWNLOAD */
.stDownloadButton > button {{
    background: transparent !important;
    border: 1.5px solid {T['blue']} !important;
    color: {T['blue']} !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}}
.stDownloadButton > button:hover {{
    background: {T['blue']} !important;
    color: #fff !important;
}}

/* ═══════════════  CUSTOM HTML CARDS (global rules)  ═════════════════════ */
/* All hand-written divs in the dashboard */
.dash-card {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: clamp(10px, 1.5vw, 18px) clamp(10px, 1.8vw, 20px);
    box-shadow: {T['shadow']};
    overflow: hidden;
    min-width: 0;
    width: 100%;
}}
/* All text inside any div that is a direct child of a column */
[data-testid="column"] div,
[data-testid="column"] span,
[data-testid="column"] p {{
    max-width: 100%;
    min-width: 0;
    word-break: break-word;
    overflow-wrap: break-word;
}}
/* Row items in detail cards (key:value rows) */
.detail-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: clamp(5px, 0.6vw, 8px) 0;
    border-bottom: 1px solid {T['border']};
    gap: 8px;
    min-width: 0;
    overflow: hidden;
}}
.detail-key {{
    font-size: clamp(0.62rem, 0.9vw, 0.77rem);
    font-weight: 600;
    color: {T['text_sub']};
    white-space: nowrap;
    flex-shrink: 0;
}}
.detail-val {{
    font-size: clamp(0.7rem, 1vw, 0.85rem);
    color: {T['text']};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
    text-align: right;
}}
/* Status strip */
.status-strip {{
    display: flex;
    align-items: center;
    flex-wrap: nowrap;
    gap: clamp(6px, 1.2vw, 18px);
    overflow: hidden;
    padding: clamp(7px, 1vw, 10px) clamp(10px, 1.8vw, 20px);
}}
.status-strip > * {{
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
    flex-shrink: 1;
}}
.status-strip > .fixed {{
    flex-shrink: 0;
}}

/* Misc */
.stAlert {{ border-radius: 10px !important; }}
hr {{ border-color: {T['border']} !important; margin: 12px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def card_open(extra_style=""):
    st.markdown(
        f"<div style='background:{T['surface']};border:1px solid {T['border']};"
        f"border-radius:12px;padding:clamp(6px,1.12vw,18px) clamp(6px,1.25vw,20px);box-shadow:{T['shadow']};{extra_style}'>",
        unsafe_allow_html=True,
    )

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def section_hdr(icon, title, sub=""):
    s = f"<span style='font-size:.78rem;color:{T['text_sub']};margin-left:4px'>{sub}</span>" if sub else ""
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:8px;margin:22px 0 10px'>"
        f"<span style='font-size:1rem'>{icon}</span>"
        f"<span style='font-size:.95rem;font-weight:700;color:{T['text']}'>{title}</span>"
        f"{s}</div>",
        unsafe_allow_html=True,
    )

def empty_box(msg="Run a cycle to populate this chart"):
    st.markdown(
        f"<div style='background:{T['surface']};border:1.5px dashed {T['border']};"
        f"border-radius:12px;padding:clamp(17px,3.25vw,52px) clamp(6px,1.25vw,20px);text-align:center;"
        f"color:{T['text_sub']};font-size:.84rem;box-shadow:{T['shadow']}'>"
        f"<div style='font-size:1.8rem;margin-bottom:10px'>📭</div>{msg}</div>",
        unsafe_allow_html=True,
    )

def risk_pill(risk):
    c = T.get(f"risk_{risk.lower()[:4]}", T["accent"])
    if risk == "Moderate": c = T["risk_med"]
    return (f"<span style='background:{c}22;color:{c};"
            f"border:1px solid {c}55;border-radius:20px;"
            f"padding:clamp(3px,0.12vw,2px) clamp(4px,0.69vw,11px);font-size:.72rem;font-weight:700;"
            f"letter-spacing:.06em'>{risk.upper()}</span>")

def plo(height=280, title=""):
    d = dict(
        paper_bgcolor=T["surface"], plot_bgcolor=T["surface"],
        font=dict(family="Inter", color=T["text_sub"], size=11),
        legend=dict(bgcolor=T["surface"], bordercolor=T["border"],
                    borderwidth=1, font=dict(color=T["text"], size=11)),
        margin=dict(l=10,r=10,t=42 if title else 18,b=10),
        height=height,
    )
    if title:
        d["title"] = dict(text=title,
                          font=dict(color=T["text"], size=13, family="Inter"),
                          x=0.01, xanchor="left")
    return d

def ax(title=""):
    return dict(
        gridcolor=T["border"], zeroline=False,
        title=dict(text=title, font=dict(color=T["text_sub"], size=11)),
        tickfont=dict(color=T["text_sub"], size=10),
        linecolor=T["border"], showline=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    from src.agent.agent_core       import MultimodalAgent
    from src.tools.vision_tool      import VisionTool
    from src.tools.anomaly_tool     import AnomalyTool
    from src.tools.decision_tool    import DecisionTool
    from src.tools.explanation_tool import ExplanationTool
    from src.agent.memory           import ShortTermMemory
    from src.agent.goal_manager     import GoalManager
    return MultimodalAgent(
        VisionTool("models/yolov8_gas_classifier.pt"),
        AnomalyTool("models/lstm_autoencoder_weights.pth"),
        DecisionTool("models/drf_gas_model.pth"),
        ExplanationTool(OLLAMA_MODEL),
        ShortTermMemory(), GoalManager(),
    )

_agent_err = None
try:    agent = load_agent();  AGENT_LIVE = True
except Exception as _e: agent = None; AGENT_LIVE = False; _agent_err = str(_e)

def _mock(arr, temp, hum, step):
    sm  = float(arr.mean())
    spk = 0.28 if step % 7 == 0 else 0.0
    ano = float(np.clip(np.random.beta(2,5) + sm*0.4 + spk, 0, 1))
    con = float(np.clip(0.65 + np.random.randn()*.08, .4, .99))
    gas = float(np.clip(ano*130 + np.random.randn()*4, 0, 150))
    idx = min(int(ano*5), 4)
    q   = np.random.dirichlet(np.ones(5)*2)
    q[idx] += .35; q /= q.sum()
    return dict(state=[ano,gas,con,temp,hum], action=ACTIONS[idx], action_idx=idx,
                q_values=q.tolist(), reward=float(1-ano*2+np.random.randn()*.04),
                anomaly=ano, gas_conc=gas, confidence=con)

# ──────────────────────────────────────────────────────────────────────────────
# SENSORS
# ──────────────────────────────────────────────────────────────────────────────
def build_sensors(csv_file, window=50):
    if csv_file is not None:
        try:
            dfc = pd.read_csv(csv_file).select_dtypes(include=[np.number])
            for i in range(max(0, N_SENSORS-dfc.shape[1])): dfc[f"_p{i}"] = 0.0
            arr = dfc.iloc[-window:, :N_SENSORS].values.astype(np.float32)
            if arr.shape[0] < window:
                arr = np.vstack([np.zeros((window-arr.shape[0],N_SENSORS), np.float32), arr])
            return arr
        except Exception as ex: st.sidebar.error(f"CSV error: {ex}")
    rng  = np.random.default_rng()
    base = np.array([.20,.35,.15,.45,.30,.25,.40], np.float32)
    arr  = np.zeros((window, N_SENSORS), np.float32)
    for s in range(N_SENSORS):
        d = np.cumsum(rng.normal(0,.005,window)).astype(np.float32)
        n = rng.normal(0,.03,window).astype(np.float32)
        sp = np.zeros(window, np.float32)
        if rng.random() < .3:
            i = rng.integers(5, window-5); sp[i:i+3] = rng.uniform(.2,.5)
        arr[:,s] = np.clip(base[s]+d+n+sp, 0, 1)
    return arr

# ──────────────────────────────────────────────────────────────────────────────
# GEMMA / OLLAMA
# ──────────────────────────────────────────────────────────────────────────────
def ollama_online():
    if _req is None: return False
    try: return _req.get("http://localhost:11434", timeout=2).status_code == 200
    except: return False

def ask_gemma(user_msg, history, system_prompt):
    if _req is None: return "⚠️ Run: `pip install requests`"
    msgs = [{"role":"system","content":system_prompt}]
    for m in history[-12:]: msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":user_msg})
    try:
        r = _req.post(OLLAMA_URL,
                      json={"model":OLLAMA_MODEL,"messages":msgs,"stream":False},
                      timeout=90)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except _req.exceptions.ConnectionError:
        return "⚠️ Ollama offline. Run `ollama serve` then `ollama pull gemma3:1b`"
    except Exception as ex:
        return f"⚠️ Error: {ex}"

def build_ctx(df, incidents):
    if df.empty: block = "No monitoring data yet."
    else:
        stats  = df[["anomaly","gas_conc","confidence","reward"]].describe().round(3).to_string()
        recent = df.tail(5)[["cycle","anomaly","gas_conc","risk","action"]].to_string(index=False)
        block  = f"STATISTICS\n{stats}\n\nRECENT 5 CYCLES\n{recent}"
    inc = ("\nINCIDENT LOG\n" +
           pd.DataFrame(incidents)[["cycle","timestamp","risk","gas_conc","event"]].to_string(index=False)
           ) if incidents else ""
    return (f"You are an expert AI analyst in a Methane Pipeline Monitoring Control Room.\n"
            f"System: YOLOv8 (vision) · LSTM autoencoder ({N_SENSORS} sensors) · "
            f"RL agent (actions: {', '.join(ACTIONS)}) · Gemma3:1b (you).\n"
            f"Risk scale: Low → Moderate → High → Critical.\n"
            f"Gas: Safe <20 ppm · Warning 20–80 ppm · Danger >80 ppm.\n\n"
            f"{block}{inc}\n\n"
            f"Be concise, accurate, and safety-focused. State uncertainty where it exists.")

def auto_explain(result, cycle):
    ano = result.get("anomaly", 0)
    gas = result.get("gas_conc", 0)
    act = result.get("action", "—")
    rsk = result.get("risk", "Low")
    con = result.get("confidence", 0)
    prompt = (f"Write 2–3 plain-English sentences explaining cycle {cycle}: "
              f"anomaly={ano:.3f}, gas={gas:.1f} ppm, risk={rsk}, "
              f"action={act}, confidence={con*100:.0f}%. "
              f"Focus on what the pipeline operator should understand.")
    return ask_gemma(prompt, [], build_ctx(pd.DataFrame(), []))

def rl_metrics(df, thr):
    if df.empty:
        return dict(episode_reward=0.0, policy_entropy=0.0,
                    false_alarm_rate=0.0, avg_latency_ms=0.0, total_episodes=0)
    probs = df["action"].value_counts(normalize=True).values
    alarm = df["action"].isin(["Raise Alarm","Shutdown"])
    return dict(
        episode_reward   = round(float(df["reward"].sum()),3),
        policy_entropy   = round(float(-np.sum(probs*np.log(probs+1e-9))),4),
        false_alarm_rate = round(float((alarm&(df["anomaly"]<thr)).sum()/max(len(df),1))*100,2),
        avg_latency_ms   = round(float(df["latency_s"].mean()*1000),1) if "latency_s" in df.columns else 0.0,
        total_episodes   = len(df),
    )

# ──────────────────────────────────────────────────────────────────────────────
# ███  TOP NAVIGATION BAR  ████████████████████████████████████████████████████
# Rendered FIRST so it appears at the very top of the page content area
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div id='main-nav' style='position:sticky;top:0;z-index:999;background:{T['surface']};border-bottom:2px solid {T['border']};padding:0 8px;margin:0 -1.5rem 0 -1.5rem;box-shadow:0 2px 8px rgba(0,0,0,.15)'>",
    unsafe_allow_html=True,
)

nav_cols = st.columns(len(TABS))
for i, (icon, label) in enumerate(TABS):
    with nav_cols[i]:
        active = st.session_state.active_tab == i
        st.markdown(f'<div class="{"nav-tab-active" if active else "nav-tab"}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{i}", use_container_width=True):
            st.session_state.active_tab = i
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

tab = st.session_state.active_tab

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Theme toggle
    tlabel = "☀  Light Mode" if IS_DARK else "☾  Dark Mode"
    st.markdown('<div class="btn-theme">', unsafe_allow_html=True)
    if st.button(tlabel, key="theme_btn", use_container_width=True):
        st.session_state.theme = "light" if IS_DARK else "dark"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<hr>", unsafe_allow_html=True)

    def sb_label(txt):
        st.markdown(f"<p style='font-size:.68rem;font-weight:700;letter-spacing:.1em;"
                    f"text-transform:uppercase;color:{T['text_sub']};margin:0 0 6px'>{txt}</p>",
                    unsafe_allow_html=True)

    sb_label("Controls")
    agent_mode   = st.selectbox("Agent Mode", ["Autonomous","Semi-Auto","Manual"])
    alert_thresh = st.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05)
    show_raw     = st.checkbox("Show Raw Log")

    st.markdown("<hr>", unsafe_allow_html=True)
    sb_label("Inputs")
    cam_img     = st.file_uploader("Camera Frame", type=["png","jpg","jpeg"])
    sensor_csv  = st.file_uploader(f"Sensor CSV ({N_SENSORS} cols)", type=["csv"], key="s_up")
    history_csv = st.file_uploader("Load History CSV", type=["csv"], key="h_up")

    st.markdown("<hr>", unsafe_allow_html=True)
    sb_label("Environment")
    manual_temp = st.slider("Temperature (°C)", 15, 55, 32)
    manual_hum  = st.slider("Humidity (%)", 20, 95, 55)

    st.markdown("<hr>", unsafe_allow_html=True)
    ra, rb = st.columns(2)
    with ra:
        st.markdown('<div class="btn-run">', unsafe_allow_html=True)
        start_btn = st.button("▶  Run Cycle", key="run_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with rb:
        st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
        reset_btn = st.button("↺  Reset", key="reset_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if reset_btn:
        for k in ["history","incidents","chat_messages"]:
            st.session_state[k] = []
        st.session_state.update(dict(cycle=0, last_result=None,
            history_loaded_file=None, auto_explanation=""))
        st.rerun()

    # Load history CSV
    if history_csv is not None:
        fid = history_csv.name + str(history_csv.size)
        if st.session_state.history_loaded_file != fid:
            try:
                h = pd.read_csv(history_csv)
                lh, li, mc = [], [], 0
                for _, row in h.iterrows():
                    cy  = int(row.get("cycle",0)); gas = float(row.get("gas_conc",0))
                    rsk = str(row.get("risk","Low")); ts  = str(row.get("timestamp","—"))
                    ev  = str(row.get("event",""));   ano = min(gas/130, 1.0)
                    lh.append(dict(cycle=cy, timestamp=ts, anomaly=ano, gas_conc=gas,
                                   confidence=.85,
                                   action="Raise Alarm" if rsk in ("High","Critical") else "Verify",
                                   reward=1-ano*2, risk=rsk, temp=32, hum=55, latency_s=.12))
                    li.append(dict(cycle=cy, timestamp=ts, event=ev, risk=rsk, gas_conc=gas))
                    mc = max(mc, cy)
                st.session_state.history            = lh
                st.session_state.incidents          = li
                st.session_state.cycle              = mc + 1
                st.session_state.history_loaded_file= fid
                st.sidebar.success(f"✅ Loaded {len(lh)} records")
                st.rerun()
            except Exception as ex: st.sidebar.error(f"Load error: {ex}")

    st.markdown("<hr>", unsafe_allow_html=True)
    _on = ollama_online()
    st.markdown(f"""
    <div style='background:{T['surface2']};border:1px solid {T['border']};
         border-radius:10px;padding:clamp(4px,0.75vw,12px) clamp(4px,0.88vw,14px);font-size:.75rem;line-height:2.2;color:{T['text_sub']}'>
      <div style='color:{"#3fb950" if AGENT_LIVE else "#f0883e"}'>
        {"✅" if AGENT_LIVE else "⚠️"}  Agent: {"Live" if AGENT_LIVE else "Simulation"}
      </div>
      <div style='color:{"#3fb950" if _on else "#f0883e"}'>
        {"✅" if _on else "⚠️"}  Gemma3:1b: {"Online" if _on else "Offline"}
      </div>
      <div>Mode: {agent_mode}</div>
      <div>Cycles: {st.session_state.cycle} &nbsp;|&nbsp; Incidents: {len(st.session_state.incidents)}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# RUN CYCLE
# ──────────────────────────────────────────────────────────────────────────────
if start_btn:
    arr = build_sensors(sensor_csv)
    img = np.array(Image.open(cam_img).convert("RGB")) if cam_img else None
    t0  = time.perf_counter()
    res = (agent.run_once(image=img, sensor_array=arr,
                          temp=manual_temp, hum=manual_hum, step=st.session_state.cycle)
           if AGENT_LIVE else _mock(arr, manual_temp, manual_hum, st.session_state.cycle))
    lat  = time.perf_counter() - t0
    s    = res.get("state",[0,0,0,manual_temp,manual_hum])
    ano  = float(res.get("anomaly",  s[0] if len(s)>0 else 0))
    gas  = float(res.get("gas_conc", s[1] if len(s)>1 else 0))
    con  = float(res.get("confidence",s[2] if len(s)>2 else 0))
    act  = str(res.get("action", ACTIONS[0]))
    rew  = float(res.get("reward", 0))
    qv   = list(res.get("q_values",[.2]*5))
    rsk  = RISK_LEVELS[min(int(ano*4),3)]
    ts   = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append(dict(
        cycle=st.session_state.cycle, timestamp=ts, anomaly=ano, gas_conc=gas,
        confidence=con, action=act, reward=rew, risk=rsk,
        temp=manual_temp, hum=manual_hum, latency_s=lat,
    ))
    st.session_state.last_result = {**res, "anomaly":ano, "gas_conc":gas,
                                     "confidence":con, "q_values":qv, "risk":rsk, "action":act}
    if ano > alert_thresh:
        st.session_state.incidents.append(dict(
            cycle=st.session_state.cycle, timestamp=ts,
            event="Methane anomaly detected", risk=rsk, gas_conc=round(gas,2),
        ))
    if ollama_online():
        st.session_state.auto_explanation = auto_explain(st.session_state.last_result,
                                                          st.session_state.cycle)
    else:
        st.session_state.auto_explanation = ""
    st.session_state.cycle += 1
    st.rerun()

# Derived state
df   = pd.DataFrame(st.session_state.history)
_has = not df.empty
last     = df.iloc[-1] if _has else None
threat   = min(100, float(last["anomaly"])*100) if _has else 0.0
gas_d    = float(last["gas_conc"])              if _has else 0.0
conf_d   = float(last["confidence"])            if _has else 0.0
cur_risk = str(last["risk"])                    if _has else "Low"
cur_act  = str(last["action"])                  if _has else "—"
last_ts  = str(last["timestamp"])               if _has else "—"
rc       = T.get(f"risk_{cur_risk.lower()[:4]}", T["accent"])
if cur_risk == "Moderate": rc = T["risk_med"]
_sarr    = build_sensors(sensor_csv)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE HEADER  (below nav bar)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:{T['surface']};border-bottom:2px solid {T['accent']};
     padding:clamp(4px,0.88vw,14px) clamp(8px,1.50vw,24px);margin-top:0;display:flex;align-items:center;
     justify-content:space-between;flex-wrap:wrap;gap:10px'>
  <div>
    <div style='font-family:"JetBrains Mono",monospace;font-size:1.2rem;
         font-weight:700;color:{T['accent']};letter-spacing:.05em'>
      ⚠  Methane AI Control Room
    </div>
    <div style='font-size:.73rem;color:{T['text_sub']};margin-top:3px'>
      Autonomous Multimodal Pipeline Surveillance · v6.0 ·&nbsp;
      <span style='color:{"#3fb950" if AGENT_LIVE else "#f0883e"}'>
        {"Agent Live" if AGENT_LIVE else "Simulation Mode"}
      </span>
    </div>
  </div>
  <div style='font-family:"JetBrains Mono",monospace;font-size:.75rem;
       color:{T['text_sub']};background:{T['surface2']};border:1px solid {T['border']};
       border-radius:8px;padding:clamp(3px,0.31vw,5px) clamp(4px,0.88vw,14px)'>
    {datetime.now().strftime("%d %b %Y  ·  %H:%M")}
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row (always visible) ─────────────────────────────────────────────────
st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1: st.metric("Agent",         "LIVE" if AGENT_LIVE else "SIM")
with k2:
    prev = float(df.iloc[-2]["anomaly"])*100 if len(df)>1 else threat
    st.metric("Threat Score", f"{threat:.0f} / 100", delta=f"{threat-prev:+.1f}")
with k3: st.metric("Gas (ppm)",     f"{gas_d:.1f}",           delta="HIGH" if gas_d>80 else "Safe")
with k4: st.metric("Confidence",    f"{conf_d*100:.0f}%")
with k5: st.metric("Incidents",     len(st.session_state.incidents))
with k6: st.metric("Cycles",        st.session_state.cycle)
st.markdown("</div>", unsafe_allow_html=True)

# ── Status strip ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:{rc}14;border:1px solid {rc}44;border-left:5px solid {rc};
     border-radius:10px;padding:clamp(3px,0.56vw,9px) clamp(6px,1.25vw,20px);margin:12px 0 4px;
     display:flex;align-items:center;flex-wrap:nowrap;gap:clamp(8px,1.5vw,20px);overflow:hidden'>
  <div style='display:flex;align-items:center;gap:6px;flex-shrink:0'>
    <span style='font-size:.68rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:{T['text_sub']}'>Risk</span>
    {risk_pill(cur_risk)}
  </div>
  <span style='color:{T['text']};font-size:.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:1'><b>Action:</b> {cur_act}</span>
  <span style='color:{T['text_sub']};font-size:.78rem;white-space:nowrap;flex-shrink:0'>Threshold: {alert_thresh}</span>
  <span style='color:{T['text_sub']};font-size:.78rem;white-space:nowrap;flex-shrink:0'>Updated: {last_ts}</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# ░░░░░  TAB CONTENT  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ──────────────────────────────────────────────────────────────────────────────

# ═══ TAB 0 — OVERVIEW ════════════════════════════════════════════════════════
if tab == 0:
    left, right = st.columns([3, 2])

    with left:
        section_hdr("📈","Anomaly & Gas Timeline")
        if _has:
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["anomaly"], name="Anomaly",
                line=dict(color=T["accent"],width=2.5),
                fill="tozeroy", fillcolor=rgba(T['accent'], 0.09), mode="lines"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df["cycle"], y=df["gas_conc"], name="Gas ppm",
                line=dict(color=T["orange"],width=1.5,dash="dot"), mode="lines"), secondary_y=True)
            fig.add_hline(y=alert_thresh, line_dash="dash", line_color=T["yellow"],
                          annotation_text=f"Threshold {alert_thresh}",
                          annotation_font_color=T["yellow"], secondary_y=False)
            fig.update_layout(**plo(300,"Anomaly Score & Gas Concentration"),
                xaxis=ax("Cycle"),
                yaxis=dict(**ax("Anomaly Score"),range=[0,1.15]),
                yaxis2=ax("Gas ppm"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_box("Timeline appears after the first cycle runs")

        section_hdr("📉","Agent Reward Trend")
        if _has:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=df["cycle"],
                y=df["reward"].rolling(5,min_periods=1).mean(), name="MA-5",
                line=dict(color=T["blue"],width=2.5),
                fill="tozeroy", fillcolor=rgba(T['blue'], 0.08)))
            fig_r.add_trace(go.Scatter(x=df["cycle"], y=df["reward"], name="Raw",
                line=dict(color=T["border"],width=1), opacity=.7))
            fig_r.add_hline(y=0, line_dash="dash", line_color=T["red"], line_width=1)
            fig_r.update_layout(**plo(220,"Agent Reward Over Time"),
                xaxis=ax("Cycle"), yaxis=ax("Reward"))
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            empty_box("Reward trend appears after cycles run")

    with right:
        section_hdr("🎯","Threat Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=threat,
            delta={"reference":50,
                   "increasing":{"color":T["red"]},
                   "decreasing":{"color":T["accent"]}},
            gauge={
                "axis":{"range":[0,100],
                        "tickcolor":T["text_sub"],
                        "tickfont":{"family":"JetBrains Mono","color":T["text_sub"],"size":9}},
                "bar":{"color":rc,"thickness":.28},
                "bgcolor":T["surface"],"borderwidth":1,"bordercolor":T["border"],
                "steps":[
                    {"range":[0,25],  "color":T["surface2"]},
                    {"range":[25,50], "color":T["surface"]},
                    {"range":[50,75], "color":"#3b2200" if IS_DARK else "#fff7e0"},
                    {"range":[75,100],"color":"#3b0000" if IS_DARK else "#ffe8e8"},
                ],
                "threshold":{"line":{"color":T["red"],"width":2.5},
                             "thickness":.8,"value":75},
            },
            title={"text":"THREAT SCORE",
                   "font":{"family":"JetBrains Mono","color":T["accent"],"size":12}},
            number={"font":{"family":"JetBrains Mono","color":T["text"],"size":44}},
        ))
        fig_g.update_layout(paper_bgcolor=T["surface"],
                            margin=dict(l=20,r=20,t=14,b=10), height=270)
        st.plotly_chart(fig_g, use_container_width=True)

        # ── GEMMA AUTO-EXPLANATION ──────────────────────────────────────────
        section_hdr("🧠","Gemma AI — Cycle Analysis",
                    "Auto-generated after every cycle")

        exp_text = st.session_state.auto_explanation
        if exp_text:
            st.markdown(f"""
            <div style='background:{T['surface']};border:1px solid {T['border']};
                 border-left:4px solid {rc};border-radius:12px;
                 padding:clamp(6px,1.12vw,18px) clamp(6px,1.25vw,20px);box-shadow:{T['shadow']}'>
              <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>
                <div style='width:36px;height:36px;background:{T['surface2']};
                     border:1.5px solid {T['accent']}55;border-radius:10px;
                     display:flex;align-items:center;justify-content:center;font-size:1.1rem'>🤖</div>
                <div>
                  <div style='font-size:.7rem;font-weight:700;letter-spacing:.08em;
                       text-transform:uppercase;color:{T['text_sub']}'>
                    GEMMA3:1B · CYCLE {st.session_state.cycle - 1}
                  </div>
                  <div style='font-size:.7rem;color:{T['text_sub']}'>{last_ts}</div>
                </div>
                <div style='margin-left:auto'>{risk_pill(cur_risk)}</div>
              </div>
              <div style='font-size:.88rem;color:{T['text']};line-height:1.75'>
                {exp_text.replace(chr(10),"<br>")}
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif not ollama_online():
            st.markdown(f"""
            <div style='background:{T['surface']};border:1.5px dashed {T['border']};
                 border-radius:12px;padding:clamp(9px,1.75vw,28px);text-align:center'>
              <div style='font-size:1.6rem;margin-bottom:8px'>⚠️</div>
              <div style='font-size:.85rem;font-weight:600;color:{T['text']}'>Gemma3:1b offline</div>
              <div style='font-size:.78rem;color:{T['text_sub']};margin-top:6px'>
                Run <code style='background:{T['surface2']};padding:clamp(3px,0.12vw,2px) clamp(4px,0.44vw,7px);
                border-radius:5px;color:{T['text']}'>ollama serve</code>
                to enable auto-explanations
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background:{T['surface']};border:1.5px dashed {T['border']};
                 border-radius:12px;padding:clamp(12px,2.25vw,36px) clamp(6px,1.25vw,20px);text-align:center'>
              <div style='font-size:1.6rem;margin-bottom:8px'>🤖</div>
              <div style='font-size:.85rem;color:{T['text_sub']}'>
                Gemma will explain the agent's reasoning here after each cycle.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Last decision summary ────────────────────────────────────────────
        section_hdr("📋","Last Decision")
        lr = st.session_state.last_result
        if lr:
            rows = [("Action",     f"<b>{lr.get('action','—')}</b>"),
                    ("Risk",       risk_pill(lr.get("risk","Low"))),
                    ("Anomaly",    f"{lr.get('anomaly',0):.3f}"),
                    ("Gas",        f"{lr.get('gas_conc',0):.1f} ppm"),
                    ("Confidence", f"{lr.get('confidence',0)*100:.0f}%"),
                    ("Reward",     f"{lr.get('reward',0):.3f}")]
            html = "".join(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:clamp(4px,0.50vw,8px) 0;border-bottom:1px solid {T['border']}'>"
                f"<span style='font-size:.77rem;font-weight:600;color:{T['text_sub']}'>{k}</span>"
                f"<span style='font-size:.85rem;color:{T['text']}'>{v}</span></div>"
                for k, v in rows)
            st.markdown(
                f"<div style='background:{T['surface']};border:1px solid {T['border']};"
                f"border-radius:12px;padding:clamp(4px,0.88vw,14px) clamp(6px,1.12vw,18px);box-shadow:{T['shadow']}'>{html}</div>",
                unsafe_allow_html=True)
        else:
            empty_box("Decision summary appears after the first cycle")

# ═══ TAB 1 — TELEMETRY ═══════════════════════════════════════════════════════
elif tab == 1:
    section_hdr("📈","Telemetry","Full cycle history")
    if _has:
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=df["cycle"],y=df["anomaly"],name="Anomaly",
            line=dict(color=T["accent"],width=2),fill="tozeroy",
            fillcolor=rgba(T['accent'], 0.08)),secondary_y=False)
        fig.add_trace(go.Scatter(x=df["cycle"],y=df["gas_conc"],name="Gas ppm",
            line=dict(color=T["orange"],width=1.5,dash="dot")),secondary_y=True)
        fig.add_hline(y=alert_thresh,line_dash="dash",line_color=T["yellow"],
                      annotation_text=f"Threshold {alert_thresh}",
                      annotation_font_color=T["yellow"],secondary_y=False)
        fig.update_layout(**plo(340,"Anomaly Score & Gas Concentration"),
            xaxis=ax("Cycle"),
            yaxis=dict(**ax("Anomaly Score"),range=[0,1.15]),
            yaxis2=ax("Gas ppm"))
        st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=df["cycle"],
                y=df["reward"].rolling(5,min_periods=1).mean(),name="MA-5",
                line=dict(color=T["blue"],width=2),fill="tozeroy",
                fillcolor=rgba(T['blue'], 0.08)))
            fig_r.add_trace(go.Scatter(x=df["cycle"],y=df["reward"],name="Raw",
                line=dict(color=T["border"],width=1),opacity=.7))
            fig_r.add_hline(y=0,line_dash="dash",line_color=T["red"],line_width=1)
            fig_r.update_layout(**plo(260,"Agent Reward"),xaxis=ax("Cycle"),yaxis=ax("Reward"))
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            fig_cf = px.histogram(df,x="confidence",nbins=20,
                                  color_discrete_sequence=[T["blue"]])
            fig_cf.update_traces(marker_line_color=T["bg"],marker_line_width=1,opacity=.85)
            fig_cf.update_layout(**plo(260,"Confidence Distribution"),
                xaxis=ax("Confidence"),yaxis=ax("Count"),bargap=.06)
            st.plotly_chart(fig_cf, use_container_width=True)
    else:
        empty_box("Telemetry charts appear after cycles run")

# ═══ TAB 2 — SENSORS ═════════════════════════════════════════════════════════
elif tab == 2:
    section_hdr("🔬","Sensor Array","7-channel LSTM input · 50-row rolling window")
    c1,c2 = st.columns([2,3])
    with c1:
        sm   = _sarr.mean(axis=0)
        sdf  = pd.DataFrame({"Sensor":SENSOR_NAMES,"Zone":SENSOR_ZONES,"Value":sm})
        fig_sb = px.bar(sdf, x="Sensor", y="Value", color="Zone",
            color_discrete_sequence=[T["accent"],T["blue"],T["yellow"],T["orange"]],
            text=sdf["Value"].apply(lambda x: f"{x:.3f}"))
        fig_sb.add_hline(y=.7,line_dash="dash",line_color=T["red"],
            annotation_text="Alert",annotation_font_color=T["red"])
        fig_sb.update_traces(textposition="outside",
                             textfont=dict(size=10,color=T["text_sub"]))
        fig_sb.update_layout(**plo(320,"Mean Reading per Sensor"),
            xaxis=ax("Sensor"),yaxis=dict(**ax("Mean Value"),range=[0,1.25]))
        st.plotly_chart(fig_sb, use_container_width=True)
    with c2:
        fig_h = go.Figure(data=go.Heatmap(
            z=_sarr, x=SENSOR_NAMES,
            colorscale=[[0,T["surface"]],[.3,T["accent"]],
                        [.6,T["yellow"]],[.85,T["orange"]],[1,T["red"]]],
            colorbar=dict(tickfont=dict(family="Inter",color=T["text_sub"],size=10),
                         bgcolor=T["surface"],bordercolor=T["border"],
                         title=dict(text="Level",font=dict(color=T["text_sub"],size=11))),
            hovertemplate="Sensor: %{x}<br>Row: %{y}<br>Value: %{z:.4f}<extra></extra>"))
        fig_h.update_layout(**plo(320,"Live Sensor Heatmap — 50-Row Window"),
            xaxis=ax("Sensor"),yaxis=ax("Time Step"))
        st.plotly_chart(fig_h, use_container_width=True)

    section_hdr("⚡","Per-Sensor Sparklines")
    sp_cols = st.columns(7)
    for i,(col,name) in enumerate(zip(sp_cols,SENSOR_NAMES)):
        with col:
            v     = _sarr[:,i]
            color = T["red"] if v.max()>.7 else T["accent"]
            fsp   = go.Figure(go.Scatter(y=v,mode="lines",
                line=dict(color=color,width=1.5),fill="tozeroy",
                fillcolor=rgba(color, 0.12)))
            fsp.update_layout(paper_bgcolor=T["surface"],plot_bgcolor=T["surface"],
                margin=dict(l=4,r=4,t=22,b=4),height=100,
                xaxis=dict(visible=False),yaxis=dict(visible=False,range=[0,1]),
                title=dict(text=name,font=dict(family="JetBrains Mono",
                           color=color,size=11),x=.5))
            st.plotly_chart(fsp, use_container_width=True)

# ═══ TAB 3 — RL DECISIONS ════════════════════════════════════════════════════
elif tab == 3:
    section_hdr("🤖","RL Agent Decisions","Q-values · distributions · action history")
    c1,c2,c3 = st.columns(3)
    with c1:
        lr  = st.session_state.last_result
        qv  = lr["q_values"] if lr else [.2]*5
        ca  = lr["action"]   if lr else "—"
        clr = [T["red"] if a==ca else T["accent"] for a in ACTIONS]
        fq  = go.Figure(go.Bar(x=ACTIONS,y=qv,marker_color=clr,
            text=[f"{v:.3f}" for v in qv],textposition="outside",
            textfont=dict(family="JetBrains Mono",size=10,color=T["text_sub"])))
        fq.update_layout(**plo(300,"RL Q-Values (red = chosen)"),
            xaxis=dict(**ax(),tickfont=dict(size=10)),
            yaxis=dict(**ax("Q-Value"),range=[0,1.3]))
        st.plotly_chart(fq, use_container_width=True)
    with c2:
        if _has:
            rc2 = df["risk"].value_counts().reindex(RISK_LEVELS,fill_value=0)
            fpie = go.Figure(go.Pie(
                labels=rc2.index,values=rc2.values,hole=.6,
                marker=dict(colors=[T["risk_low"],T["risk_med"],T["risk_high"],T["risk_crit"]],
                            line=dict(color=T["bg"],width=2)),
                textfont=dict(family="Inter",size=12)))
            fpie.add_annotation(text=f"<b>{len(df)}</b><br>cycles",
                font=dict(family="JetBrains Mono",color=T["text"],size=13),showarrow=False)
            fpie.update_layout(**plo(300,"Threat Distribution"))
            st.plotly_chart(fpie, use_container_width=True)
        else: empty_box()
    with c3:
        if _has:
            ac = df["action"].value_counts().reset_index()
            ac.columns = ["Action","Count"]
            faf = px.bar(ac,x="Count",y="Action",orientation="h",color="Count",
                color_continuous_scale=[[0,T["border"]],[.5,T["blue"]],[1,T["accent"]]],
                text="Count")
            faf.update_traces(textposition="outside",
                              textfont=dict(family="JetBrains Mono",size=10))
            faf.update_layout(**plo(300,"Action Frequency"),
                xaxis=ax(),yaxis=ax(),coloraxis_showscale=False)
            st.plotly_chart(faf, use_container_width=True)
        else: empty_box()

# ═══ TAB 4 — RL METRICS ══════════════════════════════════════════════════════
elif tab == 4:
    section_hdr("📐","RL Performance Metrics")
    rl = rl_metrics(df, alert_thresh)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Episode Reward",   f"{rl['episode_reward']:.2f}")
    m2.metric("Policy Entropy",   f"{rl['policy_entropy']:.4f}")
    m3.metric("False Alarm Rate", f"{rl['false_alarm_rate']:.1f}%")
    m4.metric("Avg Latency",      f"{rl['avg_latency_ms']:.0f} ms")
    m5.metric("Episodes",         rl["total_episodes"])
    if _has:
        r1,r2,r3 = st.columns(3)
        with r1:
            drl = df.copy(); drl["cum"] = drl["reward"].cumsum()
            fcr = go.Figure(go.Scatter(x=drl["cycle"],y=drl["cum"],
                line=dict(color=T["blue"],width=2),fill="tozeroy",fillcolor=rgba(T['blue'], 0.08)))
            fcr.update_layout(**plo(260,"Cumulative Reward"),xaxis=ax("Cycle"),yaxis=ax("Cumulative"))
            st.plotly_chart(fcr, use_container_width=True)
        with r2:
            if "latency_s" in df.columns:
                lc = [T["red"] if v>1 else T["accent"] for v in df["latency_s"]]
                fl = go.Figure(go.Bar(x=df["cycle"],y=df["latency_s"]*1000,marker_color=lc))
                fl.add_hline(y=1000,line_dash="dash",line_color=T["yellow"])
                fl.update_layout(**plo(260,"Inference Latency (ms)"),xaxis=ax("Cycle"),yaxis=ax("ms"))
                st.plotly_chart(fl, use_container_width=True)
        with r3:
            drl["fa"]  = (drl["action"].isin(["Raise Alarm","Shutdown"])&
                          (drl["anomaly"]<alert_thresh)).astype(int)
            drl["far"] = drl["fa"].rolling(10,min_periods=1).mean()*100
            ff = go.Figure(go.Scatter(x=drl["cycle"],y=drl["far"],
                line=dict(color=T["orange"],width=2),fill="tozeroy",fillcolor=rgba(T['orange'], 0.08)))
            ff.add_hline(y=10,line_dash="dash",line_color=T["yellow"])
            ff.update_layout(**plo(260,"Rolling False Alarm Rate (%)"),
                xaxis=ax("Cycle"),yaxis=dict(**ax("FAR %"),range=[0,105]))
            st.plotly_chart(ff, use_container_width=True)
    else: empty_box("Run cycles to see RL metrics")

# ═══ TAB 5 — ANALYTICS ═══════════════════════════════════════════════════════
elif tab == 5:
    section_hdr("📊","Historical Analytics")
    if _has:
        c1,c2 = st.columns(2)
        with c1:
            if "temp" in df.columns:
                fe = make_subplots(specs=[[{"secondary_y":True}]])
                fe.add_trace(go.Scatter(x=df["cycle"],y=df["temp"],name="Temp °C",
                    line=dict(color=T["orange"],width=2)),secondary_y=False)
                fe.add_trace(go.Scatter(x=df["cycle"],y=df["hum"],name="Humidity %",
                    line=dict(color=T["blue"],width=2,dash="dot")),secondary_y=True)
                fe.update_layout(**plo(280,"Environmental Conditions"),
                    xaxis=ax("Cycle"),yaxis=ax("Temp °C"),yaxis2=ax("Humidity %"))
                st.plotly_chart(fe, use_container_width=True)
        with c2:
            fig_cf = px.histogram(df,x="confidence",nbins=20,
                                  color_discrete_sequence=[T["accent"]])
            fig_cf.update_traces(marker_line_color=T["bg"],marker_line_width=1,opacity=.85)
            fig_cf.update_layout(**plo(280,"Confidence Distribution"),
                xaxis=ax("Confidence"),yaxis=ax("Count"),bargap=.06)
            st.plotly_chart(fig_cf, use_container_width=True)
        if show_raw:
            section_hdr("🗃","Raw Log")
            st.dataframe(df.sort_values("cycle",ascending=False).head(100),
                         use_container_width=True, height=300)
        section_hdr("💾","Export")
        st.download_button("⬇  Download Monitoring Log (CSV)",
            data=df.to_csv(index=False).encode(),
            file_name=f"methane_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv")
    else: empty_box("Analytics appear after cycles run")

# ═══ TAB 6 — INCIDENTS ═══════════════════════════════════════════════════════
elif tab == 6:
    section_hdr("🚨","Incident Log","Anomaly events above threshold")
    c1,c2 = st.columns([3,2])
    with c1:
        if st.session_state.incidents:
            inc = pd.DataFrame(st.session_state.incidents)
            st.dataframe(
                inc[["cycle","timestamp","risk","gas_conc","event"]].sort_values("cycle",ascending=False),
                use_container_width=True, height=300,
                column_config={
                    "cycle":    st.column_config.NumberColumn("Cycle",format="%d"),
                    "timestamp":st.column_config.TextColumn("Time"),
                    "risk":     st.column_config.TextColumn("Risk"),
                    "gas_conc": st.column_config.NumberColumn("Gas ppm",format="%.1f"),
                    "event":    st.column_config.TextColumn("Event"),
                })
            ipc = inc.groupby("cycle").size().reset_index(name="count")
            fi  = px.bar(ipc,x="cycle",y="count",color="count",
                color_continuous_scale=[[0,T["border"]],[.5,T["yellow"]],[1,T["red"]]])
            fi.update_layout(**plo(200,"Incidents per Cycle"),
                xaxis=ax("Cycle"),yaxis=ax("Count"),coloraxis_showscale=False)
            st.plotly_chart(fi, use_container_width=True)
        else: empty_box("No incidents yet — run cycles")
    with c2:
        section_hdr("📷","Camera Feed")
        if cam_img:
            st.image(cam_img, caption="Last frame", use_container_width=True)
        else:
            st.markdown(f"""
            <div style='background:{T['surface']};border:1.5px dashed {T['border']};
                 border-radius:12px;height:160px;display:flex;align-items:center;
                 justify-content:center;flex-direction:column;gap:8px'>
              <span style='font-size:1.8rem'>📷</span>
              <span style='font-size:.78rem;color:{T['text_sub']}'>Upload image in sidebar</span>
            </div>""", unsafe_allow_html=True)
        lr = st.session_state.last_result
        if lr:
            section_hdr("⚙","Last Decision")
            dc = T.get(f"risk_{lr.get('risk','Low').lower()[:4]}", T["accent"])
            if lr.get("risk") == "Moderate": dc = T["risk_med"]
            rows = [("Action",f"<b>{lr.get('action','—')}</b>"),
                    ("Risk",risk_pill(lr.get("risk","Low"))),
                    ("Anomaly",f"{lr.get('anomaly',0):.3f}"),
                    ("Gas",f"{lr.get('gas_conc',0):.1f} ppm"),
                    ("Confidence",f"{lr.get('confidence',0)*100:.0f}%")]
            html = "".join(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:clamp(4px,0.44vw,7px) 0;border-bottom:1px solid {T['border']}'>"
                f"<span style='font-size:.77rem;font-weight:600;color:{T['text_sub']}'>{k}</span>"
                f"<span style='font-size:.84rem;color:{T['text']}'>{v}</span></div>"
                for k,v in rows)
            st.markdown(
                f"<div style='background:{T['surface']};border:1px solid {T['border']};"
                f"border-left:4px solid {dc};border-radius:12px;"
                f"padding:clamp(4px,0.88vw,14px) clamp(5px,1.00vw,16px);box-shadow:{T['shadow']}'>{html}</div>",
                unsafe_allow_html=True)

# ═══ TAB 7 — CORRELATION ═════════════════════════════════════════════════════
elif tab == 7:
    section_hdr("🔭","Correlation & Trend Analysis")
    c1,c2 = st.columns(2)
    with c1:
        if _has:
            fsc = px.scatter(df,x="anomaly",y="gas_conc",color="risk",
                color_discrete_map={"Low":T["risk_low"],"Moderate":T["risk_med"],
                                    "High":T["risk_high"],"Critical":T["risk_crit"]},
                size="confidence",hover_data=["cycle","action","reward"])
            fsc.update_layout(**plo(340,"Anomaly vs Gas Concentration"),
                xaxis=ax("Anomaly Score"),yaxis=ax("Gas Concentration (ppm)"))
            st.plotly_chart(fsc, use_container_width=True)
        else: empty_box()
    with c2:
        if _has and len(df)>=5:
            dr = df.copy()
            dr["ma"]  = dr["anomaly"].rolling(5).mean()
            dr["std"] = dr["anomaly"].rolling(5).std()
            fb = go.Figure()
            fb.add_trace(go.Scatter(
                x=pd.concat([dr["cycle"],dr["cycle"].iloc[::-1]]),
                y=pd.concat([dr["ma"]+dr["std"],(dr["ma"]-dr["std"]).iloc[::-1]]),
                fill="toself",fillcolor=rgba(T["accent"], 0.08),
                line=dict(color="rgba(0,0,0,0)"),name="±1 Std"))
            fb.add_trace(go.Scatter(x=dr["cycle"],y=dr["ma"],
                line=dict(color=T["accent"],width=2.5),name="MA-5"))
            fb.add_trace(go.Scatter(x=dr["cycle"],y=dr["anomaly"],
                line=dict(color=T["border"],width=1),name="Raw",opacity=.7))
            fb.update_layout(**plo(340,"Anomaly Rolling Mean ± Std Dev"),
                xaxis=ax("Cycle"),
                yaxis=dict(**ax("Anomaly Score"),range=[0,1.15]))
            st.plotly_chart(fb, use_container_width=True)
        elif _has: st.info("Need ≥ 5 cycles for rolling analysis")
        else: empty_box()

# ═══ TAB 8 — AI ANALYST ══════════════════════════════════════════════════════
elif tab == 8:
    _on = ollama_online()

    # ── Page header card ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{T['surface']};border:1px solid {T['border']};border-radius:12px;
         padding:clamp(6px,1.25vw,20px) clamp(8px,1.50vw,24px);margin-bottom:20px;box-shadow:{T['shadow']};
         display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px'>
      <div style='display:flex;align-items:center;gap:14px'>
        <div style='width:48px;height:48px;
             background:{"#0d2818" if IS_DARK else "#d1fae5"};
             border:1.5px solid {T['accent']};border-radius:12px;
             display:flex;align-items:center;justify-content:center;font-size:1.5rem'>🤖</div>
        <div>
          <div style='font-size:1rem;font-weight:700;color:{T['text']}'>
            Gemma AI Analyst
          </div>
          <div style='font-size:.77rem;color:{T['text_sub']};margin-top:2px'>
            Powered by Gemma3:1b via Ollama · Context-aware · Full session data
          </div>
        </div>
      </div>
      <div style='display:flex;align-items:center;gap:10px'>
        <div style='padding:clamp(3px,0.25vw,4px) clamp(4px,0.88vw,14px);border-radius:20px;font-size:.72rem;
             font-weight:700;font-family:"JetBrains Mono",monospace;
             background:{"#0d2818" if _on else "#3b1000"};
             border:1px solid {"#3fb950" if _on else "#f0883e"};
             color:{"#3fb950" if _on else "#f0883e"}'>
          {"● ONLINE" if _on else "● OFFLINE"}
        </div>
        <div style='font-size:.75rem;color:{T['text_sub']};
             font-family:"JetBrains Mono",monospace'>{OLLAMA_MODEL}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not _on:
        st.warning("**Gemma3:1b is offline.** "
                   "Run `ollama serve` then `ollama pull gemma3:1b` to enable the AI Analyst.")

    # ── Two-column layout: chat left, tools right ────────────────────────────
    chat_col, side_col = st.columns([3, 1])

    with side_col:
        # Tools card
        st.markdown(f"""
        <div style='background:{T['surface']};border:1px solid {T['border']};
             border-radius:12px;padding:clamp(6px,1.12vw,18px);box-shadow:{T['shadow']};margin-bottom:12px'>
          <div style='font-size:.7rem;font-weight:700;letter-spacing:.1em;
               text-transform:uppercase;color:{T['text_sub']};margin-bottom:14px'>
            Quick Actions
          </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="btn-explain">', unsafe_allow_html=True)
        if st.button("🔍  Explain Last Decision", key="explain_btn", use_container_width=True):
            if _has:
                lr2 = df.iloc[-1]
                prompt = (f"Explain clearly for a pipeline operator:\n"
                          f"Cycle {int(lr2['cycle'])} — Anomaly: {lr2['anomaly']:.3f}, "
                          f"Gas: {lr2['gas_conc']:.1f} ppm, Risk: {lr2['risk']}, "
                          f"Action: {lr2['action']}, Confidence: {lr2['confidence']*100:.0f}%")
                with st.spinner("Gemma3:1b reasoning..."):
                    try:
                        exp = (agent.explainer.explain(prompt) if AGENT_LIVE
                               else ask_gemma(prompt, [],
                                             build_ctx(df, st.session_state.incidents)))
                        st.session_state.chat_messages.append(
                            {"role":"user","content":f"[Quick] {prompt}"})
                        st.session_state.chat_messages.append(
                            {"role":"assistant","content":exp})
                        st.rerun()
                    except Exception as ex: st.error(str(ex))
            else:
                st.warning("Run at least one cycle first.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="btn-newchat">', unsafe_allow_html=True)
        if st.button("↺  New Conversation", key="new_chat_btn", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)   # close tools card

        # Session context card
        st.markdown(f"""
        <div style='background:{T['surface']};border:1px solid {T['border']};
             border-radius:12px;padding:clamp(5px,1.00vw,16px) clamp(6px,1.12vw,18px);box-shadow:{T['shadow']}'>
          <div style='font-size:.7rem;font-weight:700;letter-spacing:.1em;
               text-transform:uppercase;color:{T['text_sub']};margin-bottom:12px'>
            Session Context
          </div>
        """, unsafe_allow_html=True)

        ctx_items = [
            ("Cycles run",   st.session_state.cycle),
            ("Incidents",    len(st.session_state.incidents)),
            ("Messages",     len(st.session_state.chat_messages)),
            ("Peak gas",     f"{df['gas_conc'].max():.1f} ppm" if _has else "—"),
            ("Current risk", risk_pill(cur_risk)),
        ]
        rows_html = "".join(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:clamp(4px,0.44vw,7px) 0;border-bottom:1px solid {T['border']}'>"
            f"<span style='font-size:.77rem;color:{T['text_sub']}'>{k}</span>"
            f"<span style='font-size:.82rem;color:{T['text']};font-weight:600'>{v}</span></div>"
            for k, v in ctx_items)
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

    with chat_col:
        # Chat history
        if not st.session_state.chat_messages:
            st.markdown(f"""
            <div style='background:{T['surface']};border:1.5px dashed {T['border']};
                 border-radius:12px;padding:clamp(21px,4.00vw,64px) clamp(10px,1.88vw,30px);text-align:center;
                 box-shadow:{T['shadow']};margin-bottom:16px'>
              <div style='font-size:2.8rem;margin-bottom:14px'>💬</div>
              <div style='font-size:1rem;font-weight:700;color:{T['text']};margin-bottom:8px'>
                Start a conversation
              </div>
              <div style='font-size:.85rem;color:{T['text_sub']};max-width:460px;
                   margin:0 auto;line-height:1.75'>
                Ask about anomaly spikes, gas trends, sensor behaviour, RL decisions,
                incident causes, or pipeline safety recommendations.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            bubbles = "<div style='display:flex;flex-direction:column;gap:14px;padding:clamp(4px,0.50vw,8px) 0'>"
            for msg in st.session_state.chat_messages:
                is_q   = msg["content"].startswith("[Quick]") or msg["content"].startswith("[Auto]")
                body   = msg["content"].replace("[Quick] ","").replace("[Auto] ","").replace("\n","<br>")
                if msg["role"] == "user":
                    bubbles += f"""
                    <div style='display:flex;justify-content:flex-end'>
                      <div style='background:{T['bubble_user']};
                           border:1px solid {T['accent']}33;
                           border-radius:16px 16px 4px 16px;
                           padding:clamp(4px,0.75vw,12px) clamp(5px,1.00vw,16px);max-width:72%'>
                        <div style='font-size:.65rem;font-weight:700;letter-spacing:.08em;
                             text-transform:uppercase;color:{T['accent']};margin-bottom:6px'>
                          {"Auto-Query" if is_q else "You"}
                        </div>
                        <div style='font-size:.88rem;color:{T['text']};line-height:1.65'>{body}</div>
                      </div>
                    </div>"""
                else:
                    bubbles += f"""
                    <div style='display:flex;gap:10px;align-items:flex-start'>
                      <div style='width:34px;height:34px;flex-shrink:0;
                           background:{T['surface2']};
                           border:1.5px solid {T['blue']}44;
                           border-radius:10px;display:flex;align-items:center;
                           justify-content:center;font-size:1rem;margin-top:2px'>🤖</div>
                      <div style='background:{T['bubble_ai']};
                           border:1px solid {T['blue']}33;
                           border-radius:4px 16px 16px 16px;
                           padding:clamp(4px,0.75vw,12px) clamp(5px,1.00vw,16px);max-width:80%'>
                        <div style='font-size:.65rem;font-weight:700;letter-spacing:.08em;
                             text-transform:uppercase;color:{T['blue']};margin-bottom:6px'>
                          Gemma3:1b
                        </div>
                        <div style='font-size:.88rem;color:{T['text']};line-height:1.75'>{body}</div>
                      </div>
                    </div>"""
            bubbles += "</div>"
            st.markdown(
                f"<div style='background:{T['bg']};border:1px solid {T['border']};"
                f"border-radius:12px;padding:clamp(4px,0.88vw,14px) clamp(5px,1.00vw,16px);"
                f"max-height:480px;overflow-y:auto;margin-bottom:14px;"
                f"box-shadow:{T['shadow']}'>{bubbles}</div>",
                unsafe_allow_html=True)

        # Input box + buttons
        st.markdown(f"""
        <div style='background:{T['surface']};border:1px solid {T['border']};
             border-radius:12px;padding:clamp(5px,1.00vw,16px) clamp(6px,1.12vw,18px);box-shadow:{T['shadow']}'>
          <div style='font-size:.7rem;font-weight:700;letter-spacing:.08em;
               text-transform:uppercase;color:{T['text_sub']};margin-bottom:10px'>
            Ask Gemma AI Analyst
          </div>
        """, unsafe_allow_html=True)

        ic, sc, cc = st.columns([7,1,1])
        with ic:
            user_input = st.text_input(
                label="msg", label_visibility="collapsed",
                placeholder="e.g.  What caused the spike?  Is S4 normal?  Should I shut down?",
                key="chat_input",
            )
        with sc:
            st.markdown('<div class="btn-send">', unsafe_allow_html=True)
            send_btn = st.button("⚡ Send", key="send_btn", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cc:
            st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
            if st.button("✕ Clear", key="clear_btn", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)   # close input card

        if send_btn and user_input.strip():
            st.session_state.chat_messages.append({"role":"user","content":user_input.strip()})
            with st.spinner("Gemma3:1b is analysing your pipeline data..."):
                reply = ask_gemma(user_input.strip(),
                                  st.session_state.chat_messages[:-1],
                                  build_ctx(df, st.session_state.incidents))
            st.session_state.chat_messages.append({"role":"assistant","content":reply})
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:40px;background:{T['surface']};border-top:1px solid {T['border']};
     border-radius:12px 12px 0 0;padding:clamp(4px,0.88vw,14px) clamp(8px,1.50vw,24px);
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
  <span style='font-size:.73rem;color:{T['text_sub']};font-weight:600'>
    Methane AI Control Room  v6.0  ·  Multimodal Pipeline Surveillance
  </span>
  <span style='font-size:.7rem;color:{T['border']};font-family:"JetBrains Mono",monospace'>
    YOLO · LSTM · RL · GEMMA3:1B · {st.session_state.theme.upper()} MODE
  </span>
</div>
""", unsafe_allow_html=True)
