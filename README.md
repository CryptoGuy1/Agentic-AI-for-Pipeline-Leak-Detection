"""
GasSafe AI — Control Dashboard v4.0
Central Multimodal Agent · Folder-Driven Live Monitor
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config MUST be first ──────────────────────────────────────
st.set_page_config(
    page_title="GasSafe AI — Control Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ──────────────────────────────────────────────────
try:
    from src.tools.anomaly_tool import AnomalyTool
    from src.tools.decision_tool import DecisionTool
    from src.tools.explanation_tool import ExplanationTool
    from src.tools.vision_tool import VisionTool
    from src.agent.agent_core import MultimodalAgent
    from src.agent.memory import ShortTermMemory
    from src.agent.goal_manager import GoalManager
    IMPORTS_OK = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# =========================================================
# CONSTANTS
# =========================================================
ACTIONS = {
    0: "Monitor",
    1: "Increase Sampling",
    2: "Request Verification",
    3: "Raise Alarm",
    4: "Emergency Shutdown",
}

GAS_MAP = {"NoGas": 0, "Smoke": 1, "Mixture": 2, "Perfume": 3}
GAS_NAMES = {v: k for k, v in GAS_MAP.items()}
CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
RAW_SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]

IMAGE_NAME_COL_CANDIDATES = [
    "Corresponding Image Name", "corresponding_image_name", "image_name", "Image Name",
]
LABEL_COL_CANDIDATES = ["Gas", "label", "Label", "class", "Class"]

ACTION_COLORS = {
    0: "#00ff9d",
    1: "#3b82f6",
    2: "#f59e0b",
    3: "#f97316",
    4: "#ff2d55",
}

GAS_COLORS = {
    "NoGas":   "#00ff9d",
    "Smoke":   "#f97316",
    "Mixture": "#ff2d55",
    "Perfume": "#a78bfa",
}

ACTION_ICONS = {0: "👁", 1: "📡", 2: "🔍", 3: "🚨", 4: "🔴"}
GAS_ICONS = {"NoGas": "✅", "Smoke": "💨", "Mixture": "☠️", "Perfume": "🌸"}


# =========================================================
# CSS — Hardened Industrial HUD
# =========================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg0:     #04080f;
        --bg1:     #080f1a;
        --bg2:     #0c1524;
        --bg3:     #101d2e;
        --border:  #1a3354;
        --border2: #243d5c;
        --amber:   #f59e0b;
        --amber2:  #fbbf24;
        --green:   #00ff9d;
        --green2:  #10b981;
        --red:     #ff2d55;
        --orange:  #f97316;
        --blue:    #3b82f6;
        --blue2:   #60a5fa;
        --violet:  #a78bfa;
        --cyan:    #22d3ee;
        --text:    #e2e8f0;
        --text2:   #94a3b8;
        --muted:   #475569;
        --mono:    'IBM Plex Mono', monospace;
        --display: 'Orbitron', sans-serif;
        --body:    'DM Sans', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: var(--body) !important;
        background-color: var(--bg0) !important;
        color: var(--text) !important;
    }

    .stApp { background: var(--bg0); }

    /* scanline overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,255,157,0.015) 2px,
            rgba(0,255,157,0.015) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }

    .block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1600px; }

    [data-testid="stSidebar"] {
        background: var(--bg1) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: var(--bg2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--text) !important;
        font-family: var(--mono) !important;
        font-size: 11px !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stSlider { color: var(--text) !important; }

    h1,h2,h3,h4 { font-family: var(--display) !important; letter-spacing: 0.06em; }

    /* ── MASTER HEADER ────────────────────────────────── */
    .master-header {
        background: linear-gradient(135deg, var(--bg1) 0%, #0a1428 60%, #060d1a 100%);
        border: 1px solid var(--border);
        border-top: 3px solid var(--amber);
        border-radius: 10px;
        padding: 1.8rem 2.4rem;
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .master-header::after {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(245,158,11,0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .hud-title {
        font-family: var(--display);
        font-size: 2rem;
        font-weight: 900;
        color: var(--amber);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0;
        text-shadow: 0 0 30px rgba(245,158,11,0.4);
    }
    .hud-subtitle {
        font-family: var(--mono);
        font-size: 0.68rem;
        color: var(--muted);
        letter-spacing: 0.2em;
        margin-top: 6px;
    }
    .hud-badge {
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: 0.15em;
        padding: 3px 10px;
        border-radius: 3px;
        display: inline-block;
        margin-right: 6px;
        margin-top: 4px;
    }
    .badge-online  { background: rgba(0,255,157,0.12); border: 1px solid rgba(0,255,157,0.3); color: var(--green); }
    .badge-offline { background: rgba(255,45,85,0.12); border: 1px solid rgba(255,45,85,0.3);  color: var(--red); }
    .badge-warn    { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: var(--amber); }

    /* ── SECTION HEADERS ─────────────────────────────── */
    .section-header {
        font-family: var(--mono);
        font-size: 0.65rem;
        color: var(--muted);
        letter-spacing: 0.25em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border);
        padding-bottom: 8px;
        margin: 1.2rem 0 1rem;
    }
    .section-header-accent {
        border-bottom-color: var(--amber);
        color: var(--amber2);
    }

    /* ── METRIC CARDS ────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--amber);
    }

    .kpi-card {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-top: 2px solid var(--amber);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        bottom: -20px; right: -20px;
        width: 80px; height: 80px;
        border-radius: 50%;
        background: rgba(245,158,11,0.04);
    }
    .kpi-label {
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.2em;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: var(--display);
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
    }
    .kpi-sub {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--muted);
        margin-top: 6px;
    }

    /* ── CASE CARD ───────────────────────────────────── */
    .case-card {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        position: relative;
    }
    .case-card-correct   { border-left: 4px solid var(--green); }
    .case-card-incorrect { border-left: 4px solid var(--red); }

    .case-title {
        font-family: var(--display);
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .case-action-chain {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
        flex-wrap: wrap;
    }
    .action-chip {
        font-family: var(--mono);
        font-size: 10px;
        padding: 3px 8px;
        border-radius: 3px;
        letter-spacing: 0.1em;
    }
    .chip-raw      { background: rgba(100,116,139,0.2); border: 1px solid #334155; color: var(--text2); }
    .chip-safety   { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.3); color: var(--blue2); }
    .chip-final-ok { background: rgba(0,255,157,0.12);  border: 1px solid rgba(0,255,157,0.35); color: var(--green); }
    .chip-final-er { background: rgba(255,45,85,0.12);  border: 1px solid rgba(255,45,85,0.35);  color: var(--red); }
    .chip-arrow    { color: var(--muted); font-size: 12px; }

    /* ── INFO BLOCK ──────────────────────────────────── */
    .info-block {
        background: var(--bg1);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        font-family: var(--mono);
        font-size: 11px;
        color: var(--text2);
        line-height: 1.9;
        white-space: pre-wrap;
    }

    /* ── EXPLANATION BOX ─────────────────────────────── */
    .expl-wrapper {
        background: linear-gradient(135deg, #060f1e 0%, #08142a 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--cyan);
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.4rem;
        margin-top: 0.6rem;
        position: relative;
    }
    .expl-wrapper::before {
        content: 'EXPLANATION';
        position: absolute;
        top: 8px; right: 12px;
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.2em;
        color: rgba(34,211,238,0.4);
    }
    .expl-content {
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.8;
        color: #a0b4cc;
        white-space: pre-wrap;
    }
    .expl-empty {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        font-style: italic;
    }

    /* ── CRITIQUE BOX ────────────────────────────────── */
    .crit-wrapper {
        background: linear-gradient(135deg, #100d00 0%, #1a1200 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--amber);
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.4rem;
        margin-top: 0.6rem;
        position: relative;
    }
    .crit-wrapper::before {
        content: 'CRITIQUE';
        position: absolute;
        top: 8px; right: 12px;
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.2em;
        color: rgba(245,158,11,0.4);
    }
    .crit-content {
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.8;
        color: #b8a070;
        white-space: pre-wrap;
    }
    .crit-empty {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        font-style: italic;
    }

    /* ── VISION BOX ──────────────────────────────────── */
    .vision-wrapper {
        background: linear-gradient(135deg, #06100a 0%, #0a1a12 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--green);
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-top: 0.6rem;
        position: relative;
    }
    .vision-wrapper::before {
        content: 'VISION REPORT';
        position: absolute;
        top: 8px; right: 12px;
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.2em;
        color: rgba(0,255,157,0.35);
    }
    .vision-error-wrapper {
        border-left-color: var(--red);
        background: linear-gradient(135deg, #10060a 0%, #1a0a0f 100%);
    }

    /* ── STATUS STRIP ────────────────────────────────── */
    .status-strip {
        background: var(--bg1);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem 1.4rem;
        display: flex;
        gap: 1.5rem;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .status-item {
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: 0.1em;
        color: var(--muted);
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .dot-on  { background: var(--green); box-shadow: 0 0 6px var(--green); animation: blink 2s infinite; }
    .dot-off { background: var(--red);   box-shadow: 0 0 6px var(--red); }
    .dot-warn { background: var(--amber); box-shadow: 0 0 6px var(--amber); }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.3; }
    }

    /* ── STALE WARNING ───────────────────────────────── */
    .stale-banner {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        font-family: var(--mono);
        font-size: 11px;
        color: var(--amber2);
        margin-bottom: 1rem;
    }

    /* ── TABS ────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg1) !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        font-family: var(--mono) !important;
        font-size: 11px !important;
        letter-spacing: 0.1em !important;
        padding: 0.6rem 1.4rem !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--amber) !important;
        border-bottom-color: var(--amber) !important;
    }

    /* ── EXPANDER ────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-family: var(--mono) !important;
        font-size: 12px !important;
        color: var(--text) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg1) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* ── BUTTON ──────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1a3a1a 0%, #0f2d0f 100%) !important;
        border: 1px solid var(--green) !important;
        color: var(--green) !important;
        font-family: var(--display) !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.15em !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1f4a1f 0%, #143514 100%) !important;
        box-shadow: 0 0 20px rgba(0,255,157,0.2) !important;
    }
    .stButton > button:disabled {
        background: var(--bg2) !important;
        border-color: var(--border) !important;
        color: var(--muted) !important;
        box-shadow: none !important;
    }

    /* ── DATAFRAME ───────────────────────────────────── */
    .stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; }

    /* ── PROGRESS ────────────────────────────────────── */
    .stProgress > div > div { background: var(--green) !important; }

    /* ── MISC ─────────────────────────────────────────── */
    .divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.2rem 0;
    }
    .mono-tag {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--muted);
        background: var(--bg2);
        border: 1px solid var(--border);
        padding: 2px 7px;
        border-radius: 3px;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def settings_hash(dqn, ae, yolo, csv, img, win, mc, expl, crit):
    """Generate a hash of all run-affecting settings to detect changes."""
    key = f"{dqn}|{ae}|{yolo}|{csv}|{img}|{win}|{mc}|{expl}|{crit}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def infer_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def row_to_sensor_array(row):
    return [float(row[c]) for c in RAW_SENSOR_COLS]


def get_true_gas_id(label: str) -> int:
    if label not in GAS_MAP:
        raise ValueError(f"Unknown label '{label}'. Must be one of: {list(GAS_MAP.keys())}")
    return GAS_MAP[label]


def load_raw_dataframe(path: str):
    df = pd.read_csv(path)
    missing = [c for c in RAW_SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing sensor columns: {missing}")
    label_col = infer_column(df, LABEL_COL_CANDIDATES)
    image_col = infer_column(df, IMAGE_NAME_COL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"No label column found. Tried: {LABEL_COL_CANDIDATES}")
    if image_col is None:
        raise ValueError(f"No image-name column found. Tried: {IMAGE_NAME_COL_CANDIDATES}")
    return df, label_col, image_col


def pick_one_image_per_folder(df, label_col, image_col, image_base_path, window_size):
    image_base = Path(image_base_path)
    if not image_base.exists():
        raise FileNotFoundError(f"Image base path missing: {image_base_path}")
    selected = {}
    for label in ["NoGas", "Smoke", "Mixture", "Perfume"]:
        label_dir = image_base / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {label_dir}")
        images = sorted([
            p for p in label_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])
        if not images:
            raise FileNotFoundError(f"No images in folder: {label_dir}")
        chosen = None
        for image_path in images:
            matches = df[
                (df[label_col].astype(str) == str(label)) &
                (df[image_col].astype(str) == str(image_path.stem))
            ]
            if not matches.empty:
                valid = [i for i in matches.index.tolist() if i >= window_size - 1]
                if valid:
                    chosen = image_path
                    break
        if chosen is None:
            raise ValueError(
                f"No usable image for '{label}' with ≥{window_size} rows of prior history."
            )
        selected[label] = chosen
    return selected


def find_matching_target_row(df, label_col, image_col, label, image_path, window_size):
    matches = df[
        (df[label_col].astype(str) == str(label)) &
        (df[image_col].astype(str) == str(image_path.stem))
    ].copy()
    if matches.empty:
        raise ValueError(f"No CSV row for label='{label}', image='{image_path.stem}'.")
    valid = [i for i in matches.index.tolist() if i >= window_size - 1]
    if not valid:
        raise ValueError(f"CSV rows found but none have enough history for label='{label}'.")
    return valid[0], image_path.stem


def build_window_rows(df, target_idx, window_size=20):
    start_idx = target_idx - window_size + 1
    if start_idx < 0:
        raise ValueError(f"Not enough rows for window_size={window_size} at target_idx={target_idx}.")
    window_df = df.iloc[start_idx:target_idx + 1].copy()
    if len(window_df) != window_size:
        raise ValueError(f"Expected {window_size} rows, got {len(window_df)}.")
    return window_df


def safe_str(val, fallback="—") -> str:
    if val is None:
        return fallback
    s = str(val).strip()
    return s if s else fallback


def safe_num(val, fmt=".4f", fallback="—") -> str:
    """
    Safely format a numeric value that may be:
    - None                → fallback
    - a list or ndarray   → mean of the values, then formatted
    - a scalar int/float  → formatted directly
    Prevents 'unsupported format string passed to list.__format__'.
    """
    if val is None:
        return fallback
    try:
        if isinstance(val, (list, tuple)):
            val = float(np.mean(val)) if len(val) > 0 else None
            if val is None:
                return fallback
        elif isinstance(val, np.ndarray):
            val = float(np.mean(val))
        else:
            val = float(val)
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


def render_explanation(text, enabled: bool):
    """Always render the explanation block — shows state clearly."""
    if not enabled:
        content = '<span class="expl-empty">⚙ Explanation disabled — enable "Show Explanations" and re-run.</span>'
    elif text is None:
        content = '<span class="expl-empty">⚠ Explanation tool returned no output. Check that the Ollama model is running and the ExplanationTool completed without error.</span>'
    elif str(text).strip() in ("", "disabled", "None"):
        content = f'<span class="expl-empty">⚠ Explanation returned an empty or disabled string: "{text}"</span>'
    else:
        escaped = str(text).replace("<", "&lt;").replace(">", "&gt;")
        content = f'<span class="expl-content">{escaped}</span>'

    st.markdown(f'<div class="expl-wrapper">{content}</div>', unsafe_allow_html=True)


def render_critique(text, enabled: bool):
    """Always render the critique block — shows state clearly."""
    if not enabled:
        content = '<span class="crit-empty">⚙ Critique disabled — enable "Show Critique" and re-run.</span>'
    elif text is None:
        content = '<span class="crit-empty">⚠ Critique tool returned no output. Check that the ExplanationTool critique pipeline completed without error.</span>'
    elif str(text).strip() in ("", "disabled", "None"):
        content = f'<span class="crit-empty">⚠ Critique returned an empty or disabled string: "{text}"</span>'
    else:
        escaped = str(text).replace("<", "&lt;").replace(">", "&gt;")
        content = f'<span class="crit-content">{escaped}</span>'

    st.markdown(f'<div class="crit-wrapper">{content}</div>', unsafe_allow_html=True)


def render_vision_block(r):
    if r.get("vision_error"):
        err = safe_str(r["vision_error"])
        st.markdown(
            f'<div class="vision-wrapper vision-error-wrapper">'
            f'<span style="font-family:var(--mono);font-size:11px;color:var(--red);">VISION ERROR: {err}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        reason = safe_str(r.get("vision_reason"), "No vision reason returned.")
        escaped = reason.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="vision-wrapper">'
            f'<span style="font-family:var(--mono);font-size:11px;color:#7ab89a;">{escaped}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# AGENT LOADING
# =========================================================
@st.cache_resource(show_spinner=False)
def load_agent(dqn_path, ae_path, yolo_path, window_size, enable_explanations, enable_critique):
    """
    Cache key includes enable_explanations AND enable_critique so toggling
    either flag forces a fresh agent load with the correct tool configuration.
    """
    decision = DecisionTool(dqn_path, device="cpu", mc_dropout_samples=5, window_size=window_size)
    anomaly = AnomalyTool(ae_path)
    vision = VisionTool(yolo_path)

    # Load explanation tool only when at least one of the two output types is needed
    explainer = None
    if enable_explanations or enable_critique:
        explainer = ExplanationTool("gemma3:1b")

    memory = ShortTermMemory(max_size=200)
    goal_manager = GoalManager()

    agent = MultimodalAgent(
        anomaly_tool=anomaly,
        decision_tool=decision,
        explanation_tool=explainer,
        memory=memory,
        goal_manager=goal_manager,
        critic=None,
        window_size=window_size,
        vision_tool=vision,
    )
    return agent


# =========================================================
# CHARTS
# =========================================================
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,15,26,0.9)",
    font=dict(family="IBM Plex Mono, monospace", color="#64748b", size=11),
    margin=dict(l=48, r=20, t=48, b=40),
    xaxis=dict(gridcolor="#1a3354", linecolor="#1a3354", zeroline=False),
    yaxis=dict(gridcolor="#1a3354", linecolor="#1a3354", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a3354", borderwidth=1),
    title_font=dict(family="Orbitron, sans-serif", size=13, color="#f59e0b"),
)


def q_bar_chart(q_values, final_action, label):
    colors_full = [ACTION_COLORS.get(i, "#64748b") for i in range(5)]

    def hex_to_rgba(h, alpha):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"

    bar_colors = [
        hex_to_rgba(c, 1.0) if i == final_action else hex_to_rgba(c, 0.25)
        for i, c in enumerate(colors_full)
    ]

    fig = go.Figure(go.Bar(
        x=[f"A{i} {ACTIONS[i]}" for i in range(5)],
        y=q_values,
        marker_color=bar_colors,
        marker_line_color=colors_full,
        marker_line_width=1.5,
        text=[f"{v:.3f}" for v in q_values],
        textposition="outside",
        textfont=dict(size=10, color="#64748b"),
    ))
    fig.update_layout(
        **PLOT_BASE,
        title=f"Q-Values — {label}",
        height=270,
        yaxis_title="Q",
    )
    return fig


def action_dist_chart(logs):
    counts = pd.Series([r["action"] for r in logs]).value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[f"{i}: {ACTIONS[i]}" for i in counts.index],
        y=counts.values,
        marker_color=[ACTION_COLORS.get(i, "#64748b") for i in counts.index],
        marker_line_color="#1a3354",
        marker_line_width=1,
        text=counts.values,
        textposition="outside",
        textfont=dict(color="#64748b", size=11),
    ))
    fig.update_layout(**PLOT_BASE, title="Final Action Distribution", height=290)
    return fig


def confidence_hist(logs):
    df_p = pd.DataFrame(logs)
    fig = go.Figure()
    for correct, color, name in [(True, "#00ff9d", "Correct"), (False, "#ff2d55", "Wrong")]:
        sub = df_p[df_p["is_correct"] == correct]["policy_confidence"]
        if sub.empty:
            continue
        fig.add_trace(go.Histogram(
            x=sub, name=name,
            marker_color=color, opacity=0.65, nbinsx=20,
            marker_line_color="#1a3354", marker_line_width=0.5,
        ))
    fig.update_layout(**PLOT_BASE, title="Policy Confidence Distribution", height=270, barmode="overlay")
    return fig


def anomaly_scatter(logs):
    df_p = pd.DataFrame(logs)
    fig = go.Figure()
    for lbl, color in GAS_COLORS.items():
        sub = df_p[df_p["label"] == lbl]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["anomaly_normalized"], y=sub["action"],
            mode="markers", name=lbl,
            marker=dict(color=color, size=10, opacity=0.8, line=dict(color=color, width=1)),
        ))
    fig.update_layout(
        **PLOT_BASE, title="Anomaly Score vs Final Action", height=300,
    )
    fig.update_xaxes(title_text="Normalized Anomaly")
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(5)),
        ticktext=[f"{i}: {ACTIONS[i]}" for i in range(5)],
    )
    return fig


def action_pipeline_chart(r):
    """Small 3-node pipeline chart showing raw → safety → final."""
    stages = ["Raw (DQN)", "After Safety", "Final"]
    values = [r["action_raw"], r["action_after_safety"], r["action"]]
    labels = [r["action_raw_name"], r["action_after_safety_name"], r["action_name"]]
    colors = [ACTION_COLORS.get(v, "#64748b") for v in values]

    fig = go.Figure()
    for i, (stage, label_, color) in enumerate(zip(stages, labels, colors)):
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode="markers+text",
            marker=dict(size=28, color=color, opacity=0.85, line=dict(color=color, width=2)),
            text=[f"{ACTION_ICONS.get(values[i], '')}\n{label_}"],
            textposition="top center",
            textfont=dict(size=10, color=color),
            showlegend=False,
        ))
        if i < 2:
            fig.add_annotation(
                x=i + 0.5, y=0,
                text="→", showarrow=False,
                font=dict(size=18, color="#334155"),
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=120,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False, range=[-0.5, 2.5]),
        yaxis=dict(visible=False, range=[-0.8, 0.8]),
        font=dict(family="IBM Plex Mono, monospace", color="#94a3b8"),
    )
    for i, stage in enumerate(stages):
        fig.add_annotation(x=i, y=-0.5, text=stage, showarrow=False,
                           font=dict(size=9, color="#475569"),
                           font_family="IBM Plex Mono, monospace")
    return fig


# =========================================================
# RUNNER
# =========================================================
def run_single_folder_case(
    agent, df, label_col, image_col, label, image_path,
    window_size, use_mc_dropout, enable_explanations, enable_critique,
):
    target_idx, image_name = find_matching_target_row(
        df=df, label_col=label_col, image_col=image_col,
        label=label, image_path=image_path, window_size=window_size,
    )
    window_df = build_window_rows(df, target_idx, window_size=window_size)
    true_gas_id = get_true_gas_id(label)

    agent.reset_window()
    final_result = None

    for i, (_, row) in enumerate(window_df.iterrows()):
        sensor_array = row_to_sensor_array(row)
        final_image_path = str(image_path) if i == (window_size - 1) else None

        result = agent.run_once(
            sensor_row=sensor_array,
            step=i,
            gas_id=true_gas_id,
            image_path=final_image_path,
            use_mc_dropout=use_mc_dropout,
            enable_explanations=enable_explanations,
            enable_critique=enable_critique,
        )
        final_result = result

    if final_result is None or not final_result.get("ready", False):
        raise RuntimeError(f"Agent did not produce a ready result for '{label}'.")

    return {
        "label":            label,
        "image_name":       image_name,
        "image_path":       str(image_path),
        "target_idx":       int(target_idx),

        "action_raw":             final_result.get("action_raw"),
        "action_raw_name":        final_result.get("action_raw_name"),
        "action_after_safety":    final_result.get("action_after_safety"),
        "action_after_safety_name": final_result.get("action_after_safety_name"),
        "action":                 final_result.get("action"),
        "action_name":            final_result.get("action_name"),

        "gas_id":           final_result.get("gas_id"),
        "is_correct":       final_result.get("is_correct"),
        "expected_actions": final_result.get("expected_actions"),

        "anomaly_raw":         final_result.get("anomaly_raw"),
        "anomaly_normalized":  final_result.get("anomaly_normalized"),
        "policy_confidence":   final_result.get("policy_confidence"),
        "reward":              final_result.get("reward"),
        "latency_ms":          float(final_result.get("latency", 0.0) * 1000.0),

        "yolo_class_id":       final_result.get("yolo_class_id"),
        "yolo_class_label":    final_result.get("yolo_class_label"),
        "yolo_confidence":     final_result.get("yolo_confidence"),
        "yolo_semantic_gas_id": final_result.get("yolo_semantic_gas_id"),
        "yolo_gas_name":       final_result.get("yolo_gas_name"),
        "vision_action_support": final_result.get("vision_action_support"),
        "vision_danger_flag":  final_result.get("vision_danger_flag"),
        "vision_reason":       final_result.get("vision_reason"),
        "vision_error":        final_result.get("vision_error"),

        "safety_changed_action":   final_result.get("safety_changed_action"),
        "vision_escalated_action": final_result.get("vision_escalated_action"),

        "q_values":   final_result.get("q_values"),
        "q_std":      final_result.get("q_std"),

        # ── These are the fields that were silently hidden before ──
        "explanation":  final_result.get("explanation"),
        "critique":     final_result.get("critique"),

        # Track what was enabled during this run so the UI can
        # show accurate "not enabled" messages vs actual None failures
        "_ran_with_explanations": enable_explanations,
        "_ran_with_critique":     enable_critique,
    }


# =========================================================
# MAIN UI
# =========================================================
inject_css()

# ── HEADER ─────────────────────────────────────────────
online = IMPORTS_OK
status_html = (
    '<span class="hud-badge badge-online">● ONLINE</span>'
    if online else
    '<span class="hud-badge badge-offline">✕ IMPORT ERROR</span>'
)

st.markdown(f"""
<div class="master-header">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
    <div>
      <p class="hud-title">⚡ GasSafe AI</p>
      <p class="hud-subtitle">CENTRAL MULTIMODAL AGENT · FOLDER-DRIVEN LIVE TEST · SENSOR + VISION VERIFICATION</p>
      <div style="margin-top:10px;">
        {status_html}
        <span class="hud-badge badge-warn">v4.0</span>
        <span class="hud-badge" style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);color:#60a5fa;">DUELING DQN</span>
        <span class="hud-badge" style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);color:#a78bfa;">LSTM AE</span>
        <span class="hud-badge" style="background:rgba(34,211,238,0.1);border:1px solid rgba(34,211,238,0.3);color:#22d3ee;">YOLOv8</span>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; line-height:2;">
        GAS STATION SAFETY INTELLIGENCE<br>
        22-FEATURE RL STATE · 5-ACTION POLICY<br>
        MC DROPOUT · VISUAL VERIFICATION
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not IMPORTS_OK:
    st.error(f"**Import failed:** {IMPORT_ERROR}")
    st.info("Run from the project root with your virtual environment activated.")
    st.stop()


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙ Model Paths</p>', unsafe_allow_html=True)
    dqn_model_path  = st.text_input("DQN Model Path",  value="models/DeepQnet.pth")
    ae_model_path   = st.text_input("AE Model Path",   value="models/lstm_autoencoder_weights.pth")
    yolo_model_path = st.text_input("YOLO Model Path", value="models/yolov8_gas_classifier.pt")
    raw_csv_path    = st.text_input(
        "Raw CSV Path",
        value=r"C:\Users\HP\Downloads\archive (7)\Multimodal Dataset for Gas Detection and Classification\Gas Sensors Measurements\Gas_Sensors_Measurements.csv",
    )
    image_base_path = st.text_input(
        "Image Folder Path",
        value=r"C:\Users\HP\Downloads\archive (7)\Multimodal Dataset for Gas Detection and Classification\Thermal Camera Images",
    )

    st.markdown('<p class="section-header">▶ Run Settings</p>', unsafe_allow_html=True)
    window_size = st.slider("Window Size", 5, 50, 20, 1)
    use_mc      = st.checkbox("MC Dropout (slower)", value=True)
    show_expl   = st.checkbox("Show Explanations", value=True,
                              help="Requires Ollama running with gemma3:1b. Disabling speeds up the run.")
    show_crit   = st.checkbox("Show Critique", value=True,
                              help="Requires ExplanationTool critique pipeline. Disabling speeds up the run.")

    if not show_expl and not show_crit:
        st.markdown(
            '<div class="hud-badge badge-warn" style="display:block;margin:4px 0;">FAST MODE — no LLM calls</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-header">◈ Display Filter</p>', unsafe_allow_html=True)
    filter_label    = st.selectbox("Filter by label", ["All", "NoGas", "Smoke", "Mixture", "Perfume"])
    show_only_wrong = st.checkbox("Show only wrong predictions", value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; line-height:2.2;">
    <span style="color:#475569;">CANONICAL GAS MAP</span><br>
    <span style="color:#00ff9d;">■</span> NoGas   → 0 → Monitor<br>
    <span style="color:#f97316;">■</span> Smoke   → 1 → Raise Alarm<br>
    <span style="color:#ff2d55;">■</span> Mixture → 2 → Emergency Shutdown<br>
    <span style="color:#a78bfa;">■</span> Perfume → 3 → Increase Sampling / Request Verification
    </div>
    """, unsafe_allow_html=True)


# ── PATH VALIDATION ─────────────────────────────────────
path_checks = {
    "DQN model":    Path(dqn_model_path).exists(),
    "AE model":     Path(ae_model_path).exists(),
    "YOLO model":   Path(yolo_model_path).exists(),
    "Raw CSV":      Path(raw_csv_path).exists(),
    "Image folder": Path(image_base_path).exists(),
}
missing_items = [name for name, ok in path_checks.items() if not ok]

# ── SYSTEM STATUS STRIP ─────────────────────────────────
tools_html = ""
for name, ok in path_checks.items():
    dot_cls = "dot-on" if ok else "dot-off"
    tools_html += f'<span class="status-item"><span class="dot {dot_cls}"></span>{name}</span>'

expl_dot = "dot-on" if show_expl else "dot-warn"
crit_dot  = "dot-on" if show_crit else "dot-warn"
tools_html += f'<span class="status-item"><span class="dot {expl_dot}"></span>Explanations</span>'
tools_html += f'<span class="status-item"><span class="dot {crit_dot}"></span>Critique</span>'
mc_dot = "dot-on" if use_mc else "dot-warn"
tools_html += f'<span class="status-item"><span class="dot {mc_dot}"></span>MC Dropout</span>'

st.markdown(f'<div class="status-strip">{tools_html}</div>', unsafe_allow_html=True)

for item in missing_items:
    st.error(f"⚠ {item} not found — check the path in the sidebar.")

run_clicked = st.button(
    "▶ RUN FOLDER LIVE TEST" if not missing_items else "⚠ FIX FILE PATHS ABOVE",
    disabled=bool(missing_items),
    use_container_width=True,
)

# ── RAW DATASET QUICK VIEW ──────────────────────────────
if Path(raw_csv_path).exists():
    try:
        df_preview, lc_preview, _ = load_raw_dataframe(raw_csv_path)
        counts = df_preview[lc_preview].value_counts()
        cols_p = st.columns(min(4, len(counts)))
        for col_, (lbl, cnt) in zip(cols_p, counts.items()):
            color = GAS_COLORS.get(str(lbl), "#64748b")
            with col_:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{GAS_ICONS.get(str(lbl), '●')} {lbl}</div>
                  <div class="kpi-value" style="color:{color};">{cnt}</div>
                  <div class="kpi-sub">raw rows</div>
                </div>""", unsafe_allow_html=True)
    except Exception:
        pass

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# =========================================================
# RUN LOGIC — with stale-detection
# =========================================================
current_hash = settings_hash(
    dqn_model_path, ae_model_path, yolo_model_path,
    raw_csv_path, image_base_path,
    window_size, use_mc, show_expl, show_crit,
)

results_exist = "dashboard_results" in st.session_state
stored_hash   = st.session_state.get("settings_hash", None)
settings_changed = results_exist and (stored_hash != current_hash)

if settings_changed:
    st.markdown("""
    <div class="stale-banner">
    ⚠ Settings have changed since the last run. Results below are from the previous configuration.
    Click <strong>RUN FOLDER LIVE TEST</strong> to generate updated results.
    </div>
    """, unsafe_allow_html=True)

if run_clicked or results_exist:
    if run_clicked:
        # Clear stale results so we start fresh
        st.session_state.pop("dashboard_results", None)

        with st.spinner("Loading central multimodal agent..."):
            agent = load_agent(
                dqn_model_path, ae_model_path, yolo_model_path,
                window_size, show_expl, show_crit,
            )

        with st.spinner("Loading raw CSV and matching class images..."):
            df, label_col, image_col = load_raw_dataframe(raw_csv_path)
            chosen_images = pick_one_image_per_folder(
                df=df, label_col=label_col, image_col=image_col,
                image_base_path=image_base_path, window_size=window_size,
            )

        all_results = []
        progress = st.progress(0, text="Running folder-driven live test...")
        items = list(chosen_images.items())

        for i, (label, image_path) in enumerate(items):
            progress.progress((i + 0.1) / len(items), text=f"Processing {label}...")
            try:
                result = run_single_folder_case(
                    agent=agent, df=df, label_col=label_col, image_col=image_col,
                    label=label, image_path=image_path, window_size=window_size,
                    use_mc_dropout=use_mc,
                    enable_explanations=show_expl,
                    enable_critique=show_crit,
                )
                all_results.append(result)
            except Exception as e:
                st.error(f"Error processing {label}: {e}")

            progress.progress((i + 1) / len(items), text=f"✓ {label} complete")

        progress.empty()
        st.session_state["dashboard_results"] = all_results
        st.session_state["settings_hash"] = current_hash

    # ── APPLY FILTERS ────────────────────────────────────
    logs = st.session_state["dashboard_results"]
    if filter_label != "All":
        logs = [r for r in logs if r["label"] == filter_label]
    if show_only_wrong:
        logs = [r for r in logs if not r["is_correct"]]

    if not logs:
        st.warning("No results match the current filters.")
        st.stop()

    df_logs = pd.DataFrame(logs)

    # ── KPI STRIP ────────────────────────────────────────
    total     = len(df_logs)
    correct   = int(df_logs["is_correct"].sum())
    accuracy  = correct / total if total else 0.0
    avg_conf  = float(df_logs["policy_confidence"].mean()) if total else 0.0
    avg_lat   = float(df_logs["latency_ms"].mean()) if total else 0.0
    expl_rate = sum(1 for r in logs if r.get("explanation") not in [None, "", "disabled", "None"]) / total if total else 0.0
    crit_rate = sum(1 for r in logs if r.get("critique") not in [None, "", "disabled", "None"]) / total if total else 0.0

    kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
    kpi_data = [
        (kc1, "Cases",             str(total),               "", "#f59e0b"),
        (kc2, "Correct",           f"{correct}/{total}",     f"{accuracy*100:.1f}%", "#00ff9d" if accuracy == 1 else "#f97316"),
        (kc3, "Avg Confidence",    f"{avg_conf:.4f}",        "",  "#3b82f6"),
        (kc4, "Avg Latency",       f"{avg_lat:.1f}",         "ms", "#a78bfa"),
        (kc5, "Explanation Rate",  f"{expl_rate*100:.0f}%",  "of cases", "#22d3ee"),
        (kc6, "Critique Rate",     f"{crit_rate*100:.0f}%",  "of cases", "#f59e0b"),
    ]
    for col_, label_, val_, sub_, color_ in kpi_data:
        with col_:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label_}</div>
              <div class="kpi-value" style="color:{color_};">{val_}</div>
              <div class="kpi-sub">{sub_}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "🔬 Case Details",
        "📋 Structured Table",
    ])


    # ─────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ─────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(action_dist_chart(logs),  use_container_width=True)
            st.plotly_chart(confidence_hist(logs),    use_container_width=True)
        with c2:
            st.plotly_chart(anomaly_scatter(logs),    use_container_width=True)

        st.markdown('<p class="section-header section-header-accent">Decision Summary — One Card Per Class</p>',
                    unsafe_allow_html=True)

        for r in logs:
            gc = GAS_COLORS.get(r["label"], "#94a3b8")
            correct_class = "case-card-correct" if r["is_correct"] else "case-card-incorrect"
            final_chip    = "chip-final-ok" if r["is_correct"] else "chip-final-er"
            icon = GAS_ICONS.get(r["label"], "●")

            raw_act   = safe_str(r.get("action_raw_name"))
            safe_act  = safe_str(r.get("action_after_safety_name"))
            final_act = safe_str(r.get("action_name"))
            vis_esc   = "✓ vision escalated" if r.get("vision_escalated_action") else "— no escalation"
            saf_chg   = "✓ safety overrode"  if r.get("safety_changed_action")   else "— no override"
            yolo_lbl  = safe_str(r.get("yolo_gas_name"))
            yolo_conf_str = safe_num(r.get("yolo_confidence"), ".3f")
            anom_str      = safe_num(r.get("anomaly_normalized"), ".4f")
            conf_str      = safe_num(r.get("policy_confidence"),  ".4f")

            st.markdown(f"""
            <div class="case-card {correct_class}">
              <div class="case-title" style="color:{gc};">{icon} {r['label']} — {r['image_name']}</div>
              <div class="case-action-chain" style="margin:10px 0;">
                <span class="action-chip chip-raw">DQN: {raw_act}</span>
                <span class="chip-arrow">→</span>
                <span class="action-chip chip-safety">Safety: {safe_act}</span>
                <span class="chip-arrow">→</span>
                <span class="action-chip {final_chip}">Final: {final_act}</span>
              </div>
              <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#475569; line-height:2;">
                YOLO: {yolo_lbl} ({yolo_conf_str}) &nbsp;|&nbsp;
                {vis_esc} &nbsp;|&nbsp; {saf_chg} &nbsp;|&nbsp;
                Anomaly: {anom_str} &nbsp;|&nbsp;
                Conf: {conf_str}
              </div>
            </div>
            """, unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────
    # TAB 2 — CASE DETAILS
    # ─────────────────────────────────────────────────────
    with tab2:
        st.markdown('<p class="section-header section-header-accent">Detailed Per-Case Breakdown</p>',
                    unsafe_allow_html=True)

        for r in logs:
            gc      = GAS_COLORS.get(r["label"], "#94a3b8")
            icon    = GAS_ICONS.get(r["label"], "●")
            correct = "✅ CORRECT" if r["is_correct"] else "❌ WRONG"

            # Did this run actually have explanations/critique enabled?
            ran_expl = r.get("_ran_with_explanations", show_expl)
            ran_crit = r.get("_ran_with_critique",     show_crit)

            with st.expander(
                f"{icon}  {r['label']}  ·  {r['image_name']}  ·  Final: {safe_str(r.get('action_name'))}  ·  {correct}",
                expanded=True,
            ):
                # ── Row 1: pipeline chart + Q-values ──────────────
                pipeline_col, qval_col = st.columns([1, 1.4])

                with pipeline_col:
                    st.markdown('<p class="section-header">Action Pipeline</p>', unsafe_allow_html=True)
                    st.plotly_chart(action_pipeline_chart(r), use_container_width=True)

                with qval_col:
                    if r.get("q_values") is not None:
                        st.plotly_chart(
                            q_bar_chart(r["q_values"], r["action"], r["label"]),
                            use_container_width=True,
                        )
                    else:
                        st.info("Q-values not available.")

                # ── Row 2: structured info + vision ───────────────
                info_col, vision_col = st.columns(2)

                with info_col:
                    st.markdown('<p class="section-header">Structured Output</p>', unsafe_allow_html=True)

                    # Pull raw values — any of these may be None, list, ndarray, or scalar
                    anom_n  = r.get("anomaly_normalized")
                    anom_r  = r.get("anomaly_raw")
                    conf    = r.get("policy_confidence")
                    rew     = r.get("reward")
                    lat     = r.get("latency_ms")
                    q_std   = r.get("q_std")   # list[float] from MC Dropout — one std per action
                    exp_act = r.get("expected_actions")

                    # q_std is a per-action list; show mean spread as the summary value
                    q_std_label = (
                        f"{safe_num(q_std, '.4f')} (mean over actions)"
                        if isinstance(q_std, (list, tuple, np.ndarray))
                        else safe_num(q_std, ".4f")
                    )

                    exp_act_str = (
                        ", ".join(f"{a}={ACTIONS.get(a, a)}" for a in exp_act)
                        if isinstance(exp_act, (list, tuple)) else safe_str(exp_act)
                    )

                    st.markdown(f"""<div class="info-block">
Label             : {r['label']}
Image             : {r['image_name']}
CSV target index  : {r['target_idx']}

── Decision Chain ──────────────────────
Raw action        : {safe_str(r.get('action_raw_name'))}
After safety      : {safe_str(r.get('action_after_safety_name'))}
Final action      : {safe_str(r.get('action_name'))}

Correct           : {r['is_correct']}
Expected actions  : {exp_act_str}
Safety changed    : {safe_str(r.get('safety_changed_action'))}
Vision escalated  : {safe_str(r.get('vision_escalated_action'))}

── Sensor / Anomaly ────────────────────
Anomaly (norm)    : {safe_num(anom_n, '.6f')}
Anomaly (raw)     : {safe_num(anom_r, '.6f')}
Policy confidence : {safe_num(conf,   '.6f')}
Q-value spread    : {q_std_label}
Reward            : {safe_num(rew,    '.4f')}
Latency           : {safe_num(lat,    '.2f')} ms

── YOLO Vision ─────────────────────────
Raw class id      : {safe_str(r.get('yolo_class_id'))}
Raw class label   : {safe_str(r.get('yolo_class_label'))}
YOLO confidence   : {safe_num(r.get('yolo_confidence'), '.4f')}
Mapped gas id     : {safe_str(r.get('yolo_semantic_gas_id'))}
Mapped gas name   : {safe_str(r.get('yolo_gas_name'))}
Action support    : {safe_str(r.get('vision_action_support'))}
Danger flag       : {safe_str(r.get('vision_danger_flag'))}</div>""",
                        unsafe_allow_html=True,
                    )

                with vision_col:
                    st.markdown('<p class="section-header">Vision Report</p>', unsafe_allow_html=True)
                    render_vision_block(r)

                # ── Row 3: Explanation ─────────────────────────────
                st.markdown('<p class="section-header section-header-accent">🧠 Agent Explanation</p>',
                            unsafe_allow_html=True)
                render_explanation(r.get("explanation"), ran_expl)

                # ── Row 4: Critique ───────────────────────────────
                st.markdown('<p class="section-header section-header-accent">⚖ Decision Critique</p>',
                            unsafe_allow_html=True)
                render_critique(r.get("critique"), ran_crit)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────
    # TAB 3 — STRUCTURED TABLE
    # ─────────────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="section-header section-header-accent">Export-Ready Results</p>',
                    unsafe_allow_html=True)

        keep_cols = [
            "label", "image_name", "target_idx",
            "action_raw_name", "action_after_safety_name", "action_name",
            "is_correct", "policy_confidence", "anomaly_normalized", "reward",
            "yolo_class_label", "yolo_confidence", "yolo_gas_name",
            "vision_action_support", "vision_danger_flag",
            "safety_changed_action", "vision_escalated_action", "latency_ms",
        ]
        table_df = df_logs[[c for c in keep_cols if c in df_logs.columns]].copy()
        st.dataframe(table_df, use_container_width=True, height=400)

        # ── Explanation / Critique inline review ──────────
        st.markdown('<p class="section-header section-header-accent">Explanation & Critique Review</p>',
                    unsafe_allow_html=True)

        for r in logs:
            gc   = GAS_COLORS.get(r["label"], "#94a3b8")
            icon = GAS_ICONS.get(r["label"], "●")
            ran_expl = r.get("_ran_with_explanations", show_expl)
            ran_crit = r.get("_ran_with_critique",     show_crit)

            st.markdown(
                f'<div style="font-family:\'Orbitron\',sans-serif;font-size:0.85rem;'
                f'font-weight:700;letter-spacing:0.1em;color:{gc};margin:1rem 0 0.3rem;">'
                f'{icon} {r["label"]} — {r["image_name"]}</div>',
                unsafe_allow_html=True,
            )
            render_explanation(r.get("explanation"), ran_expl)
            render_critique(r.get("critique"),     ran_crit)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Results CSV",
            data=csv_bytes,
            file_name="dashboard_folder_live_test_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
