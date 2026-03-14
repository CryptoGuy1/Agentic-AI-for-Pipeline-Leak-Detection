"""Methane AI Control Room — v8.0 (clean rewrite)"""

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

st.set_page_config(
    page_title="Methane AI Control Room",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACTIONS      = ["Monitor","Increase Sampling","Verify","Raise Alarm","Shutdown"]
RISK_LEVELS  = ["Low","Moderate","High","Critical"]
N_SENSORS    = 7
SENSOR_NAMES = [f"S{i}" for i in range(1, N_SENSORS+1)]
SENSOR_ZONES = ["Zone A","Zone A","Zone B","Zone B","Zone C","Zone C","Zone D"]
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:1b"

for k, v in [
    ("history",[]),("incidents",[]),("cycle",0),
    ("last_result",None),("chat_messages",[]),
    ("theme","dark"),("auto_explanation",""),
    ("sidebar_open", True),
]:
    if k not in st.session_state:
        st.session_state[k] = v

DARK = st.session_state.theme == "dark"

if DARK:
    BG,SURF,SURF2,BORDER = "#0d1117","#161b22","#21262d","#30363d"
    TEXT,TSUB            = "#e6edf3","#8b949e"
    ACCENT,BLUE,ORANGE,RED,YELLOW = "#3fb950","#58a6ff","#f0883e","#f85149","#e3b341"
else:
    BG,SURF,SURF2,BORDER = "#f6f8fa","#ffffff","#eaeef2","#d0d7de"
    TEXT,TSUB            = "#1f2328","#57606a"
    ACCENT,BLUE,ORANGE,RED,YELLOW = "#1a7f37","#0969da","#bc4c00","#cf222e","#9a6700"

RISK_COL = {"Low":ACCENT,"Moderate":BLUE,"High":YELLOW,"Critical":RED}

def rgba(h, a):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp {{
    font-family: 'Inter', sans-serif !important;
    background: {BG} !important;
    color: {TEXT} !important;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
}}
[data-testid="block-container"] {{ padding-top:0 !important; background:{BG} !important; }}

[data-testid="stSidebar"] {{
    background:{SURF} !important;
    border-right:1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
[data-testid="stSidebar"] .stSelectbox > div > div {{
    background:{SURF2} !important; border-color:{BORDER} !important;
    border-radius:8px !important;
}}

div[data-testid="metric-container"] {{
    background:{SURF} !important;
    border:1px solid {BORDER} !important;
    border-radius:12px !important;
    padding:14px 16px !important;
    overflow:hidden !important;
    transition:transform .2s,box-shadow .2s !important;
}}
div[data-testid="metric-container"]:hover {{
    transform:translateY(-2px) !important;
    box-shadow:0 4px 16px {rgba(ACCENT,0.2)} !important;
    border-color:{rgba(ACCENT,0.5)} !important;
}}
div[data-testid="metric-container"] label {{
    color:{TSUB} !important; font-size:0.66rem !important;
    font-weight:600 !important; letter-spacing:.09em !important;
    text-transform:uppercase !important; white-space:nowrap !important;
    overflow:hidden !important; text-overflow:ellipsis !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family:'JetBrains Mono',monospace !important;
    font-size:1.4rem !important; font-weight:600 !important;
    color:{TEXT} !important; white-space:nowrap !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] span {{
    font-size:0.72rem !important; white-space:nowrap !important;
}}

[data-testid="stPlotlyChart"] {{
    border:1px solid {BORDER} !important;
    border-radius:12px !important;
    overflow:hidden !important;
    background:{SURF} !important;
    transition:border-color .2s !important;
}}
[data-testid="stPlotlyChart"]:hover {{
    border-color:{rgba(ACCENT,0.4)} !important;
}}
[data-testid="stDataFrame"] {{
    border:1px solid {BORDER} !important;
    border-radius:12px !important;
    overflow:hidden !important;
}}

.stTextInput > div > div > input {{
    background:{SURF} !important; border:1.5px solid {BORDER} !important;
    border-radius:10px !important; color:{TEXT} !important;
    font-size:0.9rem !important; padding:10px 14px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color:{ACCENT} !important;
    box-shadow:0 0 0 3px {rgba(ACCENT,0.15)} !important;
}}
.stTextInput > div > div > input::placeholder {{ color:{TSUB} !important; }}

.stTabs [data-baseweb="tab-list"] {{
    background:{SURF} !important;
    border-bottom:2px solid {BORDER} !important;
    padding:0 6px !important;
    gap:2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; border:none !important;
    border-bottom:3px solid transparent !important;
    border-radius:0 !important; color:{TSUB} !important;
    font-family:'Inter',sans-serif !important;
    font-size:0.78rem !important; font-weight:500 !important;
    padding:11px 14px !important; transition:all .18s !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color:{ACCENT} !important; background:{rgba(ACCENT,0.06)} !important;
}}
.stTabs [aria-selected="true"] {{
    color:{ACCENT} !important; font-weight:700 !important;
    border-bottom:3px solid {ACCENT} !important;
    background:{rgba(ACCENT,0.08)} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background:{ACCENT} !important; height:3px !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:16px !important; }}

[data-testid="stChatMessage"] {{
    background:{SURF} !important;
    border:1px solid {BORDER} !important;
    border-radius:12px !important;
    margin-bottom:8px !important;
}}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div {{
    color:{TEXT} !important;
    background:transparent !important;
}}

[data-testid="stColumns"] {{ gap:10px !important; }}
[data-testid="stColumns"] > div {{ min-width:0 !important; overflow:hidden !important; }}

.stSelectbox label, .stSlider label, .stCheckbox label, .stFileUploader label {{
    color:{TSUB} !important; font-size:0.7rem !important;
    font-weight:600 !important; text-transform:uppercase !important;
    letter-spacing:.07em !important;
}}

::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:{ACCENT}; }}

.stAlert {{ border-radius:10px !important; }}
hr {{ border-color:{BORDER} !important; margin:12px 0 !important; }}

.stButton > button {{
    font-family:'Inter',sans-serif !important;
    font-weight:600 !important; font-size:0.82rem !important;
    border-radius:9px !important; padding:9px 14px !important;
    transition:all .18s !important; white-space:nowrap !important;
    background:transparent !important;
    border:1.5px solid {BORDER} !important;
    color:{TSUB} !important; width:100% !important;
}}
.stButton > button:hover {{
    border-color:{ACCENT} !important; color:{ACCENT} !important;
    background:{rgba(ACCENT,0.08)} !important;
    transform:translateY(-1px) !important;
}}
.btn-run .stButton > button {{
    background:linear-gradient(135deg,{ACCENT},{BLUE}) !important;
    border:none !important; color:#fff !important;
    font-weight:700 !important;
    box-shadow:0 2px 12px {rgba(ACCENT,0.4)} !important;
}}
.btn-run .stButton > button:hover {{
    opacity:.9 !important; color:#fff !important;
    box-shadow:0 4px 20px {rgba(ACCENT,0.6)} !important;
    transform:translateY(-2px) !important;
}}
.btn-reset .stButton > button {{
    border-color:{BORDER} !important; color:{TSUB} !important;
}}
.btn-reset .stButton > button:hover {{
    border-color:{RED} !important; color:{RED} !important;
    background:{rgba(RED,0.08)} !important; transform:none !important;
}}
.btn-theme .stButton > button {{
    background:{SURF2} !important; border:1.5px solid {BORDER} !important;
    color:{TEXT} !important; border-radius:20px !important;
}}
.btn-theme .stButton > button:hover {{
    background:{ACCENT} !important; border-color:{ACCENT} !important;
    color:#fff !important; transform:none !important;
}}
.btn-send .stButton > button {{
    background:linear-gradient(135deg,{ACCENT},{BLUE}) !important;
    border:none !important; color:#fff !important; font-weight:700 !important;
    box-shadow:0 2px 12px {rgba(ACCENT,0.35)} !important;
}}
.btn-send .stButton > button:hover {{
    opacity:.9 !important; color:#fff !important; transform:translateY(-1px) !important;
}}
.btn-clear .stButton > button {{
    border:1.5px solid {BORDER} !important; color:{TSUB} !important;
}}
.btn-clear .stButton > button:hover {{
    border-color:{RED} !important; color:{RED} !important;
    background:{rgba(RED,0.08)} !important; transform:none !important;
}}
.btn-explain .stButton > button {{
    background:{rgba(BLUE,0.1)} !important;
    border:1.5px solid {BLUE} !important; color:{BLUE} !important;
    font-weight:600 !important;
}}
.btn-explain .stButton > button:hover {{
    background:{BLUE} !important; color:#fff !important; transform:none !important;
}}
.btn-chip .stButton > button {{
    background:{SURF2} !important; border:1px solid {BORDER} !important;
    color:{TSUB} !important; font-size:0.76rem !important;
    border-radius:20px !important; padding:6px 12px !important;
    font-weight:500 !important;
}}
.btn-chip .stButton > button:hover {{
    border-color:{ACCENT} !important; color:{ACCENT} !important;
    background:{rgba(ACCENT,0.08)} !important; transform:none !important;
}}
.stDownloadButton > button {{
    background:transparent !important; border:1.5px solid {BLUE} !important;
    color:{BLUE} !important; border-radius:9px !important; font-weight:600 !important;
}}
.stDownloadButton > button:hover {{
    background:{BLUE} !important; color:#fff !important;
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50% {{ opacity:.4; transform:scale(1.5); }}
}}
.dot-live {{
    display:inline-block; width:8px; height:8px;
    background:{ACCENT}; border-radius:50%;
    animation:pulse 1.8s infinite;
    box-shadow:0 0 6px {ACCENT};
    vertical-align:middle; margin-right:5px;
}}
.dot-off {{
    display:inline-block; width:8px; height:8px;
    background:{RED}; border-radius:50%;
    vertical-align:middle; margin-right:5px;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def plo(h=280, title=""):
    d = dict(
        paper_bgcolor=SURF, plot_bgcolor=SURF,
        font=dict(family="Inter", color=TSUB, size=11),
        legend=dict(bgcolor=SURF, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT, size=11)),
        margin=dict(l=10,r=10,t=40 if title else 16,b=10), height=h,
    )
    if title:
        d["title"] = dict(text=title, font=dict(color=TEXT,size=13,family="Inter"),
                          x=0.01, xanchor="left")
    return d

def ax(title=""):
    return dict(
        gridcolor=BORDER, zeroline=False,
        title=dict(text=title, font=dict(color=TSUB,size=11)),
        tickfont=dict(color=TSUB,size=10),
        linecolor=BORDER, showline=True,
    )

def section(icon, title, sub=""):
    s = f" <span style='font-size:.76rem;color:{TSUB};font-weight:400'>{sub}</span>" if sub else ""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin:18px 0 10px'>"
        f"<span style='font-size:1rem'>{icon}</span>"
        f"<span style='font-size:.8rem;font-weight:700;letter-spacing:.08em;"
        f"text-transform:uppercase;color:{TEXT}'>{title}</span>{s}"
        f"<div style='flex:1;height:1px;background:linear-gradient(90deg,{BORDER},transparent);"
        f"margin-left:6px'></div></div>",
        unsafe_allow_html=True,
    )

def empty(msg="Run a cycle to see data"):
    st.markdown(
        f"<div style='background:{SURF};border:1.5px dashed {BORDER};border-radius:12px;"
        f"padding:40px 20px;text-align:center;color:{TSUB};font-size:.84rem'>"
        f"<div style='font-size:1.6rem;margin-bottom:8px'>📭</div>{msg}</div>",
        unsafe_allow_html=True,
    )

def sep():
    st.markdown(f"<hr style='border-color:{BORDER};margin:14px 0'>", unsafe_allow_html=True)


# ── Agent ─────────────────────────────────────────────────────────────────────
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

try:    agent = load_agent();  AGENT_LIVE = True
except: agent = None;          AGENT_LIVE = False

def mock(arr, temp, hum, step):
    sm  = float(arr.mean())
    spk = 0.28 if step % 7 == 0 else 0.0
    ano = float(np.clip(np.random.beta(2,5)+sm*0.4+spk, 0, 1))
    con = float(np.clip(0.65+np.random.randn()*.08, .4, .99))
    gas = float(np.clip(ano*130+np.random.randn()*4, 0, 150))
    idx = min(int(ano*5), 4)
    q   = np.random.dirichlet(np.ones(5)*2)
    q[idx] += .35; q /= q.sum()
    return dict(anomaly=ano, gas_conc=gas, confidence=con,
                action=ACTIONS[idx], action_idx=idx,
                q_values=q.tolist(),
                reward=float(1-ano*2+np.random.randn()*.04))


# ── Sensors ───────────────────────────────────────────────────────────────────
def build_sensors(window=50):
    rng  = np.random.default_rng()
    base = np.array([.20,.35,.15,.45,.30,.25,.40], np.float32)
    arr  = np.zeros((window, N_SENSORS), np.float32)
    for s in range(N_SENSORS):
        d  = np.cumsum(rng.normal(0,.005,window)).astype(np.float32)
        n  = rng.normal(0,.03,window).astype(np.float32)
        sp = np.zeros(window, np.float32)
        if rng.random() < .3:
            i = rng.integers(5,window-5); sp[i:i+3] = rng.uniform(.2,.5)
        arr[:,s] = np.clip(base[s]+d+n+sp, 0, 1)
    return arr


# ── Gemma ─────────────────────────────────────────────────────────────────────
def ollama_ok():
    if _req is None: return False
    try: return _req.get("http://localhost:11434", timeout=2).status_code == 200
    except: return False

def ask_gemma(msg, history, ctx):
    if _req is None: return "⚠️ `pip install requests` to enable Gemma."
    msgs = [{"role":"system","content":ctx}]
    for m in history[-12:]: msgs.append({"role":m["role"],"content":m["content"]})
    msgs.append({"role":"user","content":msg})
    try:
        r = _req.post(OLLAMA_URL, json={"model":OLLAMA_MODEL,"messages":msgs,"stream":False}, timeout=90)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except _req.exceptions.ConnectionError:
        return "⚠️ Ollama offline. Run: `ollama serve` then `ollama pull gemma3:1b`"
    except Exception as ex:
        return f"⚠️ Error: {ex}"

def build_ctx(df, incidents):
    if df.empty: data = "No monitoring data yet."
    else:
        stats  = df[["anomaly","gas_conc","confidence","reward"]].describe().round(3).to_string()
        recent = df.tail(5)[["cycle","anomaly","gas_conc","risk","action"]].to_string(index=False)
        data   = f"STATISTICS\n{stats}\n\nRECENT 5 CYCLES\n{recent}"
    inc = ""
    if incidents:
        inc = "\nINCIDENT LOG\n" + pd.DataFrame(incidents)[
            ["cycle","timestamp","risk","gas_conc","event"]].to_string(index=False)
    return (f"You are an expert AI analyst in a Methane Pipeline Monitoring Control Room.\n"
            f"System: YOLOv8 (vision) · LSTM autoencoder ({N_SENSORS} sensors) · "
            f"RL agent (actions: {', '.join(ACTIONS)}) · Gemma3:1b (you).\n"
            f"Risk: Low→Moderate→High→Critical. Gas: Safe<20ppm, Warning 20-80, Danger>80.\n\n"
            f"{data}{inc}\n\nBe concise, clear, and safety-focused.")

def rl_metrics(df, thr):
    if df.empty:
        return dict(reward=0.0, entropy=0.0, far=0.0, lat=0.0, n=0)
    p = df["action"].value_counts(normalize=True).values
    alarm = df["action"].isin(["Raise Alarm","Shutdown"])
    return dict(
        reward = round(float(df["reward"].sum()),3),
        entropy= round(float(-np.sum(p*np.log(p+1e-9))),4),
        far    = round(float((alarm&(df["anomaly"]<thr)).sum()/max(len(df),1))*100,2),
        lat    = round(float(df["latency_s"].mean()*1000),1) if "latency_s" in df.columns else 0.0,
        n      = len(df),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # App branding
    st.markdown(
        f"<div style='text-align:center;padding:8px 0 10px'>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:.9rem;"
        f"font-weight:700;color:{ACCENT}'>⚠ Methane AI</div>"
        f"<div style='font-size:.68rem;color:{TSUB};letter-spacing:.1em'>CONTROL ROOM v8.0</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    sep()

    # ── RUN / RESET ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;"
        f"text-transform:uppercase;color:{TSUB};margin-bottom:6px'>Cycle Control</div>",
        unsafe_allow_html=True,
    )
    _c1, _c2 = st.columns(2)
    with _c1:
        st.markdown('<div class="btn-run">', unsafe_allow_html=True)
        start_btn = st.button("▶  Run", key="run_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with _c2:
        st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
        reset_btn = st.button("↺  Reset", key="reset_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if reset_btn:
        for k in ["history","incidents","chat_messages"]:
            st.session_state[k] = []
        st.session_state.update(dict(cycle=0, last_result=None, auto_explanation="",
                                     sidebar_open=True))
        st.rerun()

    sep()

    # ── CAMERA ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;"
        f"text-transform:uppercase;color:{TSUB};margin-bottom:6px'>📷 Camera Frame</div>",
        unsafe_allow_html=True,
    )
    cam_img = st.file_uploader(
        "Camera", type=["png","jpg","jpeg"],
        key="cam_up", label_visibility="collapsed",
    )
    if cam_img:
        st.image(cam_img, use_container_width=True)
        st.caption(f"✅ {cam_img.name}  ·  {cam_img.size//1024} KB")
    else:
        st.markdown(
            f"<div style='background:{SURF2};border:1px dashed {BORDER};border-radius:8px;"
            f"padding:12px;text-align:center;color:{TSUB};font-size:.78rem'>"
            f"📷 No image — upload PNG/JPG</div>",
            unsafe_allow_html=True,
        )

    sep()

    # ── CONTROLS ──────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;"
        f"text-transform:uppercase;color:{TSUB};margin-bottom:6px'>⚙ Settings</div>",
        unsafe_allow_html=True,
    )
    agent_mode   = st.selectbox("Agent Mode", ["Autonomous","Semi-Auto","Manual"])
    alert_thresh = st.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05)
    show_raw     = st.checkbox("Show Raw Log")

    sep()

    # ── ENVIRONMENT ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;"
        f"text-transform:uppercase;color:{TSUB};margin-bottom:6px'>🌡 Environment</div>",
        unsafe_allow_html=True,
    )
    manual_temp = st.slider("Temperature (°C)", 15, 55, 32)
    manual_hum  = st.slider("Humidity (%)",     20, 95, 55)

    sep()

    # ── THEME ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="btn-theme">', unsafe_allow_html=True)
    if st.button("☀ Light Mode" if DARK else "☾ Dark Mode",
                 key="theme_btn", use_container_width=True):
        st.session_state.theme = "light" if DARK else "dark"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    sep()

    # ── STATUS ────────────────────────────────────────────────────────────────
    _on = ollama_ok()
    st.markdown(
        f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;"
        f"text-transform:uppercase;color:{TSUB};margin-bottom:8px'>📡 Status</div>",
        unsafe_allow_html=True,
    )
    for _lbl, _val, _col in [
        ("Agent",    "Live" if AGENT_LIVE else "Simulation", ACCENT if AGENT_LIVE else ORANGE),
        ("Gemma",    "Online" if _on else "Offline",         ACCENT if _on else RED),
        ("Mode",     agent_mode,                             BLUE),
        ("Cycles",   str(st.session_state.cycle),            TEXT),
        ("Incidents",str(len(st.session_state.incidents)),   RED if st.session_state.incidents else TEXT),
        ("Camera",   cam_img.name[:16] if cam_img else "None", ACCENT if cam_img else TSUB),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:5px 0;border-bottom:1px solid {BORDER};font-size:.8rem'>"
            f"<span style='color:{TSUB}'>{_lbl}</span>"
            f"<span style='color:{_col};font-weight:600'>{_val}</span></div>",
            unsafe_allow_html=True,
        )

    sep()

    # ── SIDEBAR TOGGLE (show/hide button inside sidebar) ──────────────────────
    st.markdown(
        f"<div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;"
        f"text-transform:uppercase;color:{TSUB};margin-bottom:6px'>🔲 Sidebar</div>",
        unsafe_allow_html=True,
    )
    if st.button("← Hide Sidebar", key="hide_sidebar_btn", use_container_width=True):
        st.session_state.sidebar_open = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR TOGGLE BUTTON — shown in main area when sidebar is hidden
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.sidebar_open:
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{ display: none !important; }}
    [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN CYCLE
# ══════════════════════════════════════════════════════════════════════════════
if start_btn:
    arr = build_sensors()
    img = np.array(Image.open(cam_img).convert("RGB")) if cam_img else None
    t0  = time.perf_counter()
    if AGENT_LIVE:
        res = agent.run_once(image=img, sensor_array=arr,
                             temp=manual_temp, hum=manual_hum,
                             step=st.session_state.cycle)
    else:
        res = mock(arr, manual_temp, manual_hum, st.session_state.cycle)
    lat = time.perf_counter() - t0
    ano = float(res.get("anomaly", 0))
    gas = float(res.get("gas_conc", 0))
    con = float(res.get("confidence", 0))
    act = str(res.get("action", ACTIONS[0]))
    rew = float(res.get("reward", 0))
    qv  = list(res.get("q_values", [.2]*5))
    rsk = RISK_LEVELS[min(int(ano*4), 3)]
    ts  = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append(dict(
        cycle=st.session_state.cycle, timestamp=ts,
        anomaly=ano, gas_conc=gas, confidence=con,
        action=act, reward=rew, risk=rsk,
        temp=manual_temp, hum=manual_hum, latency_s=lat,
    ))
    st.session_state.last_result = dict(
        anomaly=ano, gas_conc=gas, confidence=con,
        action=act, reward=rew, risk=rsk,
        q_values=qv, cycle=st.session_state.cycle, timestamp=ts,
    )
    if ano > alert_thresh:
        st.session_state.incidents.append(dict(
            cycle=st.session_state.cycle, timestamp=ts,
            event="Methane anomaly detected", risk=rsk, gas_conc=round(gas,2),
        ))
    if ollama_ok():
        p = (f"In 2-3 plain sentences for an operator: cycle {st.session_state.cycle}, "
             f"anomaly={ano:.3f}, gas={gas:.1f}ppm, risk={rsk}, "
             f"action={act}, confidence={con*100:.0f}%. What happened and what's next?")
        st.session_state.auto_explanation = ask_gemma(p, [], build_ctx(pd.DataFrame(), []))
    else:
        st.session_state.auto_explanation = ""
    st.session_state.cycle += 1
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED STATE
# ══════════════════════════════════════════════════════════════════════════════
df   = pd.DataFrame(st.session_state.history)
_has = not df.empty
lr   = st.session_state.last_result
last = df.iloc[-1] if _has else None

threat   = min(100, float(last["anomaly"])*100) if _has else 0.0
gas_d    = float(last["gas_conc"])   if _has else 0.0
conf_d   = float(last["confidence"]) if _has else 0.0
cur_risk = str(last["risk"])         if _has else "Low"
cur_act  = str(last["action"])       if _has else "—"
last_ts  = str(last["timestamp"])    if _has else "—"
rc       = RISK_COL.get(cur_risk, ACCENT)
sarr     = build_sensors()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
_live_dot = f'<span class="dot-live"></span>' if AGENT_LIVE else f'<span class="dot-off"></span>'

_hcol, _title_col = st.columns([0.05, 0.95])
with _hcol:
    # Show sidebar button only when sidebar is hidden
    if not st.session_state.sidebar_open:
        st.markdown(f"""
        <style>
        div[data-testid="stButton"][id="show_sb_wrap"] button {{
            background: linear-gradient(135deg,{ACCENT},{BLUE}) !important;
            border: none !important; color: #fff !important;
            border-radius: 9px !important; font-size: 1.1rem !important;
            font-weight: 700 !important; padding: 8px !important;
            box-shadow: 0 2px 10px {rgba(ACCENT,0.45)} !important;
            height: 42px !important; width: 100% !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        if st.button("☰", key="show_sb_btn", help="Show sidebar"):
            st.session_state.sidebar_open = True
            st.rerun()

with _title_col:
    st.markdown(
        f"<div style='background:{SURF};border-bottom:2px solid {BORDER};"
        f"padding:12px 22px;display:flex;align-items:center;"
        f"justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:0'>"
        f"<div style='display:flex;align-items:center;gap:14px'>"
        f"<div style='width:40px;height:40px;"
        f"background:linear-gradient(135deg,{ACCENT},{BLUE});"
        f"border-radius:10px;display:flex;align-items:center;"
        f"justify-content:center;font-size:1.2rem;box-shadow:0 0 14px {rgba(ACCENT,0.4)}'>⚠</div>"
        f"<div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:1rem;font-weight:700;"
        f"background:linear-gradient(90deg,{ACCENT},{BLUE});"
        f"-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
        f"Methane AI Control Room</div>"
        f"<div style='font-size:.72rem;color:{TSUB};margin-top:2px'>"
        f"Autonomous Pipeline Surveillance · v8.0</div>"
        f"</div></div>"
        f"<div style='display:flex;align-items:center;gap:10px'>"
        f"<div style='background:{SURF2};border:1px solid {rgba(rc,0.4)};"
        f"border-radius:20px;padding:5px 14px;font-size:.72rem;font-weight:700;"
        f"font-family:JetBrains Mono,monospace;color:{rc}'>"
        f"{_live_dot}{'AGENT LIVE' if AGENT_LIVE else 'SIMULATION'}</div>"
        f"<div style='background:{SURF2};border:1px solid {BORDER};"
        f"border-radius:8px;padding:5px 12px;font-family:JetBrains Mono,monospace;"
        f"font-size:.72rem;color:{TSUB}'>"
        f"{datetime.now().strftime('%d %b %Y · %H:%M')}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
_k = st.columns(3, gap="small")
with _k[0]:
    prev = float(df.iloc[-2]["anomaly"])*100 if len(df)>1 else threat
    st.metric("Threat Score", f"{threat:.0f}/100", delta=f"{threat-prev:+.1f}")
with _k[1]:
    st.metric("Gas Concentration", f"{gas_d:.1f} ppm", delta="HIGH ↑" if gas_d>80 else "Safe ↓")
with _k[2]:
    st.metric("AI Confidence", f"{conf_d*100:.0f}%")

_k2 = st.columns(3, gap="small")
with _k2[0]: st.metric("Current Risk",    cur_risk)
with _k2[1]: st.metric("Total Incidents", len(st.session_state.incidents))
with _k2[2]: st.metric("Cycles Run",      st.session_state.cycle)

# Status strip
st.markdown(
    f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px'>"
    f"<span style='background:{rgba(rc,0.12)};border:1px solid {rgba(rc,0.4)};"
    f"border-radius:20px;padding:5px 14px;font-size:.72rem;font-weight:700;"
    f"font-family:JetBrains Mono,monospace;color:{rc}'>● {cur_risk.upper()} RISK</span>"
    f"<span style='background:{SURF2};border:1px solid {BORDER};"
    f"border-radius:20px;padding:5px 14px;font-size:.8rem;color:{TEXT}'>"
    f"Action: <b>{cur_act}</b></span>"
    f"<span style='background:{SURF2};border:1px solid {BORDER};"
    f"border-radius:20px;padding:5px 14px;font-size:.78rem;color:{TSUB}'>"
    f"Threshold: {alert_thresh}</span>"
    f"<span style='background:{SURF2};border:1px solid {BORDER};"
    f"border-radius:20px;padding:5px 14px;font-size:.78rem;color:{TSUB}'>"
    f"Updated: {last_ts}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
_tab_labels = [
    "🏠  Overview","📈  Telemetry","🔬  Sensors","🤖  RL Decisions",
    "📐  RL Metrics","📊  Analytics","🚨  Incidents","🔭  Correlation","🧠  AI Analyst",
]
tabs = st.tabs(_tab_labels)


# ── TAB 0: OVERVIEW ──────────────────────────────────────────────────────────
with tabs[0]:
    _l, _r = st.columns([3,2], gap="medium")
    with _l:
        section("📈","Anomaly & Gas Timeline")
        if _has:
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=df["cycle"],y=df["anomaly"],name="Anomaly",
                line=dict(color=ACCENT,width=2.5),
                fill="tozeroy",fillcolor=rgba(ACCENT,.1),mode="lines"),secondary_y=False)
            fig.add_trace(go.Scatter(x=df["cycle"],y=df["gas_conc"],name="Gas ppm",
                line=dict(color=ORANGE,width=1.5,dash="dot"),mode="lines"),secondary_y=True)
            fig.add_hline(y=alert_thresh,line_dash="dash",line_color=YELLOW,
                          annotation_text=f"Threshold {alert_thresh}",
                          annotation_font_color=YELLOW,secondary_y=False)
            fig.update_layout(**plo(290,"Anomaly Score & Gas Concentration"),
                xaxis=ax("Cycle"),
                yaxis=dict(**ax("Anomaly"),range=[0,1.15]),
                yaxis2=ax("Gas ppm"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty("Timeline appears after the first cycle")

        section("📉","Agent Reward Trend")
        if _has:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=df["cycle"],
                y=df["reward"].rolling(5,min_periods=1).mean(),name="MA-5",
                line=dict(color=BLUE,width=2.5),fill="tozeroy",fillcolor=rgba(BLUE,.1)))
            fig_r.add_trace(go.Scatter(x=df["cycle"],y=df["reward"],name="Raw",
                line=dict(color=BORDER,width=1),opacity=.7))
            fig_r.add_hline(y=0,line_dash="dash",line_color=RED,line_width=1)
            fig_r.update_layout(**plo(200,"Reward Over Time"),
                xaxis=ax("Cycle"),yaxis=ax("Reward"))
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            empty("Reward chart appears after cycles run")

    with _r:
        section("🎯","Threat Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=threat,
            delta={"reference":50,"increasing":{"color":RED},"decreasing":{"color":ACCENT}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":TSUB,
                        "tickfont":{"family":"JetBrains Mono","color":TSUB,"size":9}},
                "bar":{"color":rc,"thickness":.28},
                "bgcolor":SURF,"borderwidth":1,"bordercolor":BORDER,
                "steps":[
                    {"range":[0,25],"color":SURF2},
                    {"range":[25,50],"color":SURF},
                    {"range":[50,75],"color":"#3b2200" if DARK else "#fff7e0"},
                    {"range":[75,100],"color":"#3b0000" if DARK else "#ffe8e8"},
                ],
                "threshold":{"line":{"color":RED,"width":2.5},"thickness":.8,"value":75},
            },
            title={"text":"THREAT SCORE","font":{"family":"JetBrains Mono","color":ACCENT,"size":12}},
            number={"font":{"family":"JetBrains Mono","color":TEXT,"size":44}},
        ))
        fig_g.update_layout(paper_bgcolor=SURF,margin=dict(l=20,r=20,t=14,b=10),height=260)
        st.plotly_chart(fig_g, use_container_width=True)

        section("🧠","Gemma AI Analysis","Auto-generated after every cycle")
        exp = st.session_state.auto_explanation
        if exp:
            st.markdown(
                f"<div style='background:{SURF};border:1px solid {BORDER};"
                f"border-left:4px solid {ACCENT};border-radius:12px;padding:16px 18px'>"
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
                f"<div style='width:32px;height:32px;"
                f"background:linear-gradient(135deg,{ACCENT},{BLUE});"
                f"border-radius:8px;display:flex;align-items:center;"
                f"justify-content:center;font-size:1rem;flex-shrink:0'>🤖</div>"
                f"<div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:.68rem;"
                f"font-weight:700;color:{ACCENT};letter-spacing:.08em'>"
                f"GEMMA3:1B · CYCLE {st.session_state.cycle-1}</div>"
                f"<div style='font-size:.7rem;color:{TSUB}'>{last_ts} · "
                f"<span style='color:{rc};font-weight:600'>{cur_risk} Risk</span></div>"
                f"</div></div>"
                f"<div style='font-size:.88rem;color:{TEXT};line-height:1.75;"
                f"border-top:1px solid {BORDER};padding-top:10px'>"
                f"{exp.replace(chr(10),'<br>')}</div></div>",
                unsafe_allow_html=True,
            )
        elif not ollama_ok():
            st.warning("**Gemma3:1b offline** — run `ollama serve` then `ollama pull gemma3:1b`")
        else:
            empty("Gemma explains each cycle here after it runs")

        if lr:
            section("📋","Last Decision")
            _da, _db = st.columns(2)
            with _da:
                st.metric("Action",     lr.get("action","—"))
                st.metric("Anomaly",    f"{lr.get('anomaly',0):.3f}")
                st.metric("Gas",        f"{lr.get('gas_conc',0):.1f} ppm")
            with _db:
                st.metric("Risk",       lr.get("risk","—"))
                st.metric("Confidence", f"{lr.get('confidence',0)*100:.0f}%")
                st.metric("Reward",     f"{lr.get('reward',0):.3f}")


# ── TAB 1: TELEMETRY ─────────────────────────────────────────────────────────
with tabs[1]:
    section("📈","Real-Time Telemetry")
    if _has:
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=df["cycle"],y=df["anomaly"],name="Anomaly",
            line=dict(color=ACCENT,width=2),fill="tozeroy",fillcolor=rgba(ACCENT,.1)),secondary_y=False)
        fig.add_trace(go.Scatter(x=df["cycle"],y=df["gas_conc"],name="Gas ppm",
            line=dict(color=ORANGE,width=1.5,dash="dot")),secondary_y=True)
        fig.add_hline(y=alert_thresh,line_dash="dash",line_color=YELLOW,secondary_y=False)
        fig.update_layout(**plo(320,"Anomaly & Gas"),
            xaxis=ax("Cycle"),
            yaxis=dict(**ax("Anomaly"),range=[0,1.15]),
            yaxis2=ax("Gas ppm"))
        st.plotly_chart(fig, use_container_width=True)
        _tc1, _tc2 = st.columns(2,gap="medium")
        with _tc1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=df["cycle"],
                y=df["reward"].rolling(5,min_periods=1).mean(),name="MA-5",
                line=dict(color=BLUE,width=2),fill="tozeroy",fillcolor=rgba(BLUE,.1)))
            fig_r.add_trace(go.Scatter(x=df["cycle"],y=df["reward"],name="Raw",
                line=dict(color=BORDER,width=1),opacity=.7))
            fig_r.add_hline(y=0,line_dash="dash",line_color=RED,line_width=1)
            fig_r.update_layout(**plo(260,"Reward"),xaxis=ax("Cycle"),yaxis=ax("Reward"))
            st.plotly_chart(fig_r, use_container_width=True)
        with _tc2:
            fig_cf = px.histogram(df,x="confidence",nbins=20,color_discrete_sequence=[BLUE])
            fig_cf.update_traces(marker_line_color=BG,marker_line_width=1,opacity=.85)
            fig_cf.update_layout(**plo(260,"Confidence Distribution"),
                xaxis=ax("Confidence"),yaxis=ax("Count"),bargap=.06)
            st.plotly_chart(fig_cf, use_container_width=True)
    else:
        empty()


# ── TAB 2: SENSORS ───────────────────────────────────────────────────────────
with tabs[2]:
    section("🔬","Sensor Array","7-channel · 50-row window · simulated")
    _sc1, _sc2 = st.columns([2,3],gap="medium")
    with _sc1:
        sm  = sarr.mean(axis=0)
        sdf = pd.DataFrame({"Sensor":SENSOR_NAMES,"Zone":SENSOR_ZONES,"Value":sm})
        fig_sb = px.bar(sdf,x="Sensor",y="Value",color="Zone",
            color_discrete_sequence=[ACCENT,BLUE,YELLOW,ORANGE],
            text=sdf["Value"].apply(lambda x:f"{x:.3f}"))
        fig_sb.add_hline(y=.7,line_dash="dash",line_color=RED,
            annotation_text="Alert",annotation_font_color=RED)
        fig_sb.update_traces(textposition="outside",textfont=dict(size=10,color=TSUB))
        fig_sb.update_layout(**plo(320,"Mean Sensor Reading"),
            xaxis=ax("Sensor"),yaxis=dict(**ax("Value"),range=[0,1.3]))
        st.plotly_chart(fig_sb, use_container_width=True)
    with _sc2:
        fig_h = go.Figure(data=go.Heatmap(
            z=sarr,x=SENSOR_NAMES,
            colorscale=[[0,SURF],[.3,ACCENT],[.6,YELLOW],[.85,ORANGE],[1,RED]],
            colorbar=dict(tickfont=dict(family="Inter",color=TSUB,size=10),
                         bgcolor=SURF,bordercolor=BORDER,
                         title=dict(text="Level",font=dict(color=TSUB,size=11))),
            hovertemplate="Sensor: %{x}<br>Row: %{y}<br>Value: %{z:.4f}<extra></extra>"))
        fig_h.update_layout(**plo(320,"Live Sensor Heatmap"),
            xaxis=ax("Sensor"),yaxis=ax("Time Step"))
        st.plotly_chart(fig_h, use_container_width=True)
    section("⚡","Per-Sensor Sparklines")
    _sp = st.columns(7,gap="small")
    for i,(col,name) in enumerate(zip(_sp,SENSOR_NAMES)):
        with col:
            v  = sarr[:,i]
            c  = RED if v.max()>.7 else ACCENT
            fs = go.Figure(go.Scatter(y=v,mode="lines",
                line=dict(color=c,width=1.5),fill="tozeroy",fillcolor=rgba(c,.15)))
            fs.update_layout(paper_bgcolor=SURF,plot_bgcolor=SURF,
                margin=dict(l=4,r=4,t=22,b=4),height=100,
                xaxis=dict(visible=False),yaxis=dict(visible=False,range=[0,1]),
                title=dict(text=name,font=dict(family="JetBrains Mono",color=c,size=11),x=.5))
            st.plotly_chart(fs, use_container_width=True)


# ── TAB 3: RL DECISIONS ──────────────────────────────────────────────────────
with tabs[3]:
    section("🤖","RL Agent Decisions")
    _rc1,_rc2,_rc3 = st.columns(3,gap="medium")
    with _rc1:
        qv  = lr["q_values"] if lr else [.2]*5
        ca  = lr["action"]   if lr else "—"
        clr = [RED if a==ca else ACCENT for a in ACTIONS]
        fq  = go.Figure(go.Bar(x=ACTIONS,y=qv,marker_color=clr,
            text=[f"{v:.3f}" for v in qv],textposition="outside",
            textfont=dict(family="JetBrains Mono",size=10,color=TSUB)))
        fq.update_layout(**plo(300,"Q-Values (red=chosen)"),
            xaxis=ax(),yaxis=dict(**ax("Q-Value"),range=[0,1.3]))
        st.plotly_chart(fq, use_container_width=True)
    with _rc2:
        if _has:
            rv = df["risk"].value_counts().reindex(RISK_LEVELS,fill_value=0)
            fp = go.Figure(go.Pie(labels=rv.index,values=rv.values,hole=.6,
                marker=dict(colors=[ACCENT,BLUE,YELLOW,RED],line=dict(color=BG,width=2)),
                textfont=dict(family="Inter",size=12)))
            fp.add_annotation(text=f"<b>{len(df)}</b><br>cycles",
                font=dict(family="JetBrains Mono",color=TEXT,size=13),showarrow=False)
            fp.update_layout(**plo(300,"Threat Distribution"))
            st.plotly_chart(fp, use_container_width=True)
        else: empty()
    with _rc3:
        if _has:
            ac = df["action"].value_counts().reset_index()
            ac.columns=["Action","Count"]
            fa = px.bar(ac,x="Count",y="Action",orientation="h",color="Count",
                color_continuous_scale=[[0,BORDER],[.5,BLUE],[1,ACCENT]],text="Count")
            fa.update_traces(textposition="outside",
                             textfont=dict(family="JetBrains Mono",size=10))
            fa.update_layout(**plo(300,"Action Frequency"),
                xaxis=ax(),yaxis=ax(),coloraxis_showscale=False)
            st.plotly_chart(fa, use_container_width=True)
        else: empty()


# ── TAB 4: RL METRICS ────────────────────────────────────────────────────────
with tabs[4]:
    section("📐","RL Performance Metrics")
    rl = rl_metrics(df, alert_thresh)
    _m1,_m2,_m3,_m4,_m5 = st.columns(5)
    _m1.metric("Episode Reward",  f"{rl['reward']:.2f}")
    _m2.metric("Policy Entropy",  f"{rl['entropy']:.4f}")
    _m3.metric("False Alarm Rate",f"{rl['far']:.1f}%")
    _m4.metric("Avg Latency",     f"{rl['lat']:.0f} ms")
    _m5.metric("Episodes",        rl["n"])
    if _has:
        _r1,_r2,_r3 = st.columns(3,gap="medium")
        with _r1:
            drl=df.copy(); drl["cum"]=drl["reward"].cumsum()
            fc=go.Figure(go.Scatter(x=drl["cycle"],y=drl["cum"],
                line=dict(color=BLUE,width=2),fill="tozeroy",fillcolor=rgba(BLUE,.1)))
            fc.update_layout(**plo(260,"Cumulative Reward"),xaxis=ax("Cycle"),yaxis=ax("Cumulative"))
            st.plotly_chart(fc, use_container_width=True)
        with _r2:
            if "latency_s" in df.columns:
                lc=[RED if v>1 else ACCENT for v in df["latency_s"]]
                fl=go.Figure(go.Bar(x=df["cycle"],y=df["latency_s"]*1000,marker_color=lc))
                fl.add_hline(y=1000,line_dash="dash",line_color=YELLOW)
                fl.update_layout(**plo(260,"Inference Latency (ms)"),xaxis=ax("Cycle"),yaxis=ax("ms"))
                st.plotly_chart(fl, use_container_width=True)
        with _r3:
            drl["fa"]=(drl["action"].isin(["Raise Alarm","Shutdown"])&
                       (drl["anomaly"]<alert_thresh)).astype(int)
            drl["far"]=drl["fa"].rolling(10,min_periods=1).mean()*100
            ff=go.Figure(go.Scatter(x=drl["cycle"],y=drl["far"],
                line=dict(color=ORANGE,width=2),fill="tozeroy",fillcolor=rgba(ORANGE,.1)))
            ff.add_hline(y=10,line_dash="dash",line_color=YELLOW)
            ff.update_layout(**plo(260,"Rolling False Alarm Rate (%)"),
                xaxis=ax("Cycle"),yaxis=dict(**ax("FAR %"),range=[0,105]))
            st.plotly_chart(ff, use_container_width=True)
    else: empty()


# ── TAB 5: ANALYTICS ─────────────────────────────────────────────────────────
with tabs[5]:
    section("📊","Historical Analytics")
    if _has:
        _ac1,_ac2 = st.columns(2,gap="medium")
        with _ac1:
            if "temp" in df.columns:
                fe=make_subplots(specs=[[{"secondary_y":True}]])
                fe.add_trace(go.Scatter(x=df["cycle"],y=df["temp"],name="Temp °C",
                    line=dict(color=ORANGE,width=2)),secondary_y=False)
                fe.add_trace(go.Scatter(x=df["cycle"],y=df["hum"],name="Humidity %",
                    line=dict(color=BLUE,width=2,dash="dot")),secondary_y=True)
                fe.update_layout(**plo(280,"Environmental Conditions"),
                    xaxis=ax("Cycle"),yaxis=ax("Temp °C"),yaxis2=ax("Humidity %"))
                st.plotly_chart(fe, use_container_width=True)
        with _ac2:
            fig_cf=px.histogram(df,x="confidence",nbins=20,color_discrete_sequence=[ACCENT])
            fig_cf.update_traces(marker_line_color=BG,marker_line_width=1,opacity=.85)
            fig_cf.update_layout(**plo(280,"Confidence Distribution"),
                xaxis=ax("Confidence"),yaxis=ax("Count"),bargap=.06)
            st.plotly_chart(fig_cf, use_container_width=True)
        if show_raw:
            section("🗃","Raw Log")
            st.dataframe(df.sort_values("cycle",ascending=False).head(100),
                         use_container_width=True,height=300)
        sep()
        st.download_button("⬇ Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"methane_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv")
    else: empty()


# ── TAB 6: INCIDENTS ─────────────────────────────────────────────────────────
with tabs[6]:
    section("🚨","Incident Log")
    _ic1,_ic2 = st.columns([3,2],gap="medium")
    with _ic1:
        if st.session_state.incidents:
            inc=pd.DataFrame(st.session_state.incidents)
            st.dataframe(
                inc[["cycle","timestamp","risk","gas_conc","event"]].sort_values("cycle",ascending=False),
                use_container_width=True,height=300,
                column_config={
                    "cycle":    st.column_config.NumberColumn("Cycle",format="%d"),
                    "timestamp":st.column_config.TextColumn("Time"),
                    "risk":     st.column_config.TextColumn("Risk"),
                    "gas_conc": st.column_config.NumberColumn("Gas ppm",format="%.1f"),
                    "event":    st.column_config.TextColumn("Event"),
                },
            )
            ipc=inc.groupby("cycle").size().reset_index(name="count")
            fi=px.bar(ipc,x="cycle",y="count",color="count",
                color_continuous_scale=[[0,BORDER],[.5,YELLOW],[1,RED]])
            fi.update_layout(**plo(180,"Incidents per Cycle"),
                xaxis=ax("Cycle"),yaxis=ax("Count"),coloraxis_showscale=False)
            st.plotly_chart(fi, use_container_width=True)
        else:
            st.info("No incidents yet — run cycles to detect anomalies.")
    with _ic2:
        section("📷","Camera Feed")
        if cam_img:
            st.image(cam_img, caption="Last uploaded frame", use_container_width=True)
        else:
            st.caption("Upload a camera image in the sidebar.")
        if lr:
            section("⚙","Last Decision")
            st.metric("Action",     lr.get("action","—"))
            _ia,_ib=st.columns(2)
            with _ia:
                st.metric("Risk",    lr.get("risk","—"))
                st.metric("Anomaly", f"{lr.get('anomaly',0):.3f}")
            with _ib:
                st.metric("Gas",     f"{lr.get('gas_conc',0):.1f} ppm")
                st.metric("Conf",    f"{lr.get('confidence',0)*100:.0f}%")


# ── TAB 7: CORRELATION ───────────────────────────────────────────────────────
with tabs[7]:
    section("🔭","Correlation & Trend Analysis")
    _cc1,_cc2=st.columns(2,gap="medium")
    with _cc1:
        if _has:
            fsc=px.scatter(df,x="anomaly",y="gas_conc",color="risk",
                color_discrete_map={"Low":ACCENT,"Moderate":BLUE,"High":YELLOW,"Critical":RED},
                size="confidence",hover_data=["cycle","action","reward"])
            fsc.update_layout(**plo(320,"Anomaly vs Gas Concentration"),
                xaxis=ax("Anomaly Score"),yaxis=ax("Gas (ppm)"))
            st.plotly_chart(fsc, use_container_width=True)
        else: empty()
    with _cc2:
        if _has and len(df)>=5:
            dr=df.copy(); dr["ma"]=dr["anomaly"].rolling(5).mean()
            dr["std"]=dr["anomaly"].rolling(5).std()
            fb=go.Figure()
            fb.add_trace(go.Scatter(
                x=pd.concat([dr["cycle"],dr["cycle"].iloc[::-1]]),
                y=pd.concat([dr["ma"]+dr["std"],(dr["ma"]-dr["std"]).iloc[::-1]]),
                fill="toself",fillcolor=rgba(ACCENT,.1),
                line=dict(color="rgba(0,0,0,0)"),name="±1 Std"))
            fb.add_trace(go.Scatter(x=dr["cycle"],y=dr["ma"],
                line=dict(color=ACCENT,width=2.5),name="MA-5"))
            fb.add_trace(go.Scatter(x=dr["cycle"],y=dr["anomaly"],
                line=dict(color=BORDER,width=1),name="Raw",opacity=.7))
            fb.update_layout(**plo(320,"Anomaly Rolling Mean ± Std Dev"),
                xaxis=ax("Cycle"),yaxis=dict(**ax("Anomaly"),range=[0,1.15]))
            st.plotly_chart(fb, use_container_width=True)
        elif _has: st.info("Need ≥ 5 cycles for rolling analysis.")
        else: empty()


# ── TAB 8: AI ANALYST (GEMMA CHATBOT) ────────────────────────────────────────
with tabs[8]:
    _on = ollama_ok()

    st.markdown(
        f"<div style='background:{SURF};border:1px solid {BORDER};border-radius:12px;"
        f"padding:16px 20px;margin-bottom:16px;"
        f"display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px'>"
        f"<div style='display:flex;align-items:center;gap:12px'>"
        f"<div style='width:44px;height:44px;"
        f"background:linear-gradient(135deg,{ACCENT},{BLUE});"
        f"border-radius:12px;display:flex;align-items:center;"
        f"justify-content:center;font-size:1.4rem'>🤖</div>"
        f"<div>"
        f"<div style='font-size:.95rem;font-weight:700;color:{TEXT}'>Gemma AI Analyst</div>"
        f"<div style='font-size:.74rem;color:{TSUB};margin-top:2px'>"
        f"Powered by {OLLAMA_MODEL} · Full session context injected · Ask anything</div>"
        f"</div></div>"
        f"<div style='padding:5px 14px;border-radius:20px;font-family:JetBrains Mono,monospace;"
        f"font-size:.72rem;font-weight:700;"
        f"background:{rgba(ACCENT if _on else RED,.12)};"
        f"border:1px solid {ACCENT if _on else RED};"
        f"color:{ACCENT if _on else RED}'>"
        f"{'● ONLINE' if _on else '● OFFLINE'}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not _on:
        st.warning("**Gemma3:1b offline** — run `ollama serve` then `ollama pull gemma3:1b`")

    _gs1,_gs2,_gs3,_gs4 = st.columns(4)
    _gs1.metric("Gemma Status",  "Online" if _on else "Offline")
    _gs2.metric("Model",          OLLAMA_MODEL)
    _gs3.metric("Messages",       len(st.session_state.chat_messages))
    _gs4.metric("Cycles Run",     st.session_state.cycle)

    sep()

    _chat, _side = st.columns([3,1], gap="medium")

    with _side:
        section("🔍","Last Decision")
        if _has and lr:
            _dc = RISK_COL.get(lr.get("risk","Low"), ACCENT)
            rows = [
                ("Risk",       f"<b style='color:{_dc}'>{lr.get('risk','—')}</b>"),
                ("Action",     lr.get("action","—")),
                ("Anomaly",    f"{lr.get('anomaly',0):.3f}"),
                ("Gas (ppm)",  f"{lr.get('gas_conc',0):.1f}"),
                ("Confidence", f"{lr.get('confidence',0)*100:.0f}%"),
                ("Reward",     f"{lr.get('reward',0):.3f}"),
            ]
            tbl = "".join(
                f"<tr>"
                f"<td style='color:{TSUB};padding:5px 0;border-bottom:1px solid {BORDER};"
                f"font-size:.8rem;width:45%'>{k}</td>"
                f"<td style='color:{TEXT};font-weight:600;padding:5px 0;"
                f"border-bottom:1px solid {BORDER};text-align:right;font-size:.8rem'>{v}</td>"
                f"</tr>"
                for k,v in rows
            )
            st.markdown(
                f"<div style='background:{SURF};border:1px solid {BORDER};"
                f"border-left:3px solid {_dc};border-radius:10px;padding:12px 14px'>"
                f"<table style='width:100%;border-collapse:collapse'>{tbl}</table></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="btn-explain">', unsafe_allow_html=True)
            if st.button("🧠 Explain This Decision", key="explain_btn", use_container_width=True):
                lr2 = df.iloc[-1]
                prompt = (
                    f"Explain this pipeline monitoring decision in plain English for an operator. "
                    f"Be specific: what was detected, why the risk is {lr2['risk']}, "
                    f"what the operator should do now.\n\n"
                    f"Cycle {int(lr2['cycle'])}: Anomaly={lr2['anomaly']:.3f} (0-1 scale, >0.5 concerning), "
                    f"Gas={lr2['gas_conc']:.1f}ppm (danger>80), Risk={lr2['risk']}, "
                    f"Action={lr2['action']}, Confidence={lr2['confidence']*100:.0f}%."
                )
                with st.spinner("Gemma is analysing..."):
                    reply = ask_gemma(prompt, [], build_ctx(df, st.session_state.incidents))
                st.session_state.chat_messages.append(
                    {"role":"user","content":f"Explain the last decision (Cycle {int(lr2['cycle'])})."})
                st.session_state.chat_messages.append(
                    {"role":"assistant","content":reply})
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Run a cycle first.")

        sep()

        section("📊","Session Summary")
        if _has:
            _peak = df["gas_conc"].max()
            _avgA = df["anomaly"].mean()
            _topA = df["action"].value_counts().index[0]
            for k,v in [
                ("Cycles",     st.session_state.cycle),
                ("Incidents",  len(st.session_state.incidents)),
                ("Peak Gas",   f"{_peak:.1f} ppm"),
                ("Avg Anomaly",f"{_avgA:.3f}"),
                ("Top Action", _topA),
                ("Risk Now",   cur_risk),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:4px 0;border-bottom:1px solid {BORDER};font-size:.8rem'>"
                    f"<span style='color:{TSUB}'>{k}</span>"
                    f"<span style='color:{TEXT};font-weight:600'>{v}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No data yet.")

        sep()
        if st.button("🗑 Clear Chat", key="clear_chat_btn", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    with _chat:
        section("💬","Conversation","Ask about decisions, trends, incidents, safety")

        if not st.session_state.chat_messages:
            st.markdown(
                f"<div style='background:{SURF};border:1.5px dashed {BORDER};"
                f"border-radius:12px;padding:40px 24px;text-align:center;color:{TSUB}'>"
                f"<div style='font-size:2.5rem;margin-bottom:12px'>💬</div>"
                f"<div style='font-size:.95rem;font-weight:600;color:{TEXT};margin-bottom:8px'>"
                f"Ask Gemma anything about your pipeline</div>"
                f"<div style='font-size:.82rem;line-height:1.75'>"
                f"Why was an alarm raised? &nbsp;·&nbsp; Is the pipeline safe? "
                f"&nbsp;·&nbsp; What caused the anomaly? &nbsp;·&nbsp; Summarise the session"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_messages:
                body = msg["content"]
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(body)
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(body)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if _has:
            st.caption("QUICK QUESTIONS")
            _q1,_q2,_q3 = st.columns(3,gap="small")
            _chips = [
                ("Why was an alarm raised?",       "q1"),
                ("Is the pipeline safe now?",      "q2"),
                ("Summarise the last 10 cycles",   "q3"),
                ("What caused the last anomaly?",  "q4"),
                ("What action is recommended?",    "q5"),
                ("Show gas concentration trends",  "q6"),
            ]
            for i,(qtxt,qkey) in enumerate(_chips):
                with [_q1,_q2,_q3][i%3]:
                    st.markdown('<div class="btn-chip">', unsafe_allow_html=True)
                    if st.button(qtxt, key=f"chip_{qkey}", use_container_width=True):
                        st.session_state.chat_messages.append({"role":"user","content":qtxt})
                        with st.spinner("Gemma thinking..."):
                            r = ask_gemma(qtxt, st.session_state.chat_messages[:-1],
                                          build_ctx(df, st.session_state.incidents))
                        st.session_state.chat_messages.append({"role":"assistant","content":r})
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        sep()

        _ic, _sc, _cc = st.columns([7,1,1], gap="small")
        with _ic:
            user_input = st.text_input(
                label="chat", label_visibility="collapsed",
                placeholder="Ask about anomalies, gas readings, sensor behaviour, RL decisions...",
                key="gemma_input",
            )
        with _sc:
            st.markdown('<div class="btn-send">', unsafe_allow_html=True)
            send_btn = st.button("⚡ Send", key="gemma_send", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with _cc:
            st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
            if st.button("✕", key="gemma_clear", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        if send_btn and user_input.strip():
            st.session_state.chat_messages.append(
                {"role":"user","content":user_input.strip()})
            with st.spinner("Gemma3:1b analysing pipeline data..."):
                reply = ask_gemma(user_input.strip(),
                                  st.session_state.chat_messages[:-1],
                                  build_ctx(df, st.session_state.incidents))
            st.session_state.chat_messages.append(
                {"role":"assistant","content":reply})
            st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='margin-top:32px;padding:12px 20px;background:{SURF};"
    f"border-top:1px solid {BORDER};border-radius:12px 12px 0 0;"
    f"display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>"
    f"<span style='font-size:.72rem;color:{TSUB}'>"
    f"Methane AI Control Room v8.0 · YOLO · LSTM · RL · GEMMA3:1B</span>"
    f"<span style='font-family:JetBrains Mono,monospace;font-size:.7rem;color:{TSUB}'>"
    f"{datetime.now().strftime('%d %b %Y · %H:%M:%S')} · "
    f"{'DARK' if DARK else 'LIGHT'} MODE</span>"
    f"</div>",
    unsafe_allow_html=True,
)
