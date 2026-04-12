"""
Gas Station Safety Intelligence Dashboard
Central Multimodal Agent — Folder-Driven Live Monitor
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    from src.tools.anomaly_tool import AnomalyTool
    from src.tools.decision_tool import DecisionTool
    from src.tools.explanation_tool import ExplanationTool
    from src.tools.vision_tool import VisionTool
    from src.agent.agent_core import MultimodalAgent
    from src.agent.memory import ShortTermMemory
    from src.agent.goal_manager import GoalManager
    IMPORTS_OK = True
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

GAS_MAP = {
    "NoGas": 0,
    "Smoke": 1,
    "Mixture": 2,
    "Perfume": 3,
}

GAS_NAMES = {v: k for k, v in GAS_MAP.items()}
CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
RAW_SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]

IMAGE_NAME_COL_CANDIDATES = [
    "Corresponding Image Name",
    "corresponding_image_name",
    "image_name",
    "Image Name",
]

LABEL_COL_CANDIDATES = [
    "Gas",
    "label",
    "Label",
    "class",
    "Class",
]

ACTION_COLORS = {
    0: "#22c55e",
    1: "#3b82f6",
    2: "#f59e0b",
    3: "#f97316",
    4: "#ef4444",
}

GAS_COLORS = {
    "NoGas": "#22c55e",
    "Smoke": "#f97316",
    "Mixture": "#ef4444",
    "Perfume": "#3b82f6",
}


# =========================================================
# CSS
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
        --amber:  #f59e0b;
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
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1450px; }

    [data-testid="stSidebar"] {
        background: var(--bg1) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    h1,h2,h3,h4 { font-family: var(--display) !important; letter-spacing: 0.05em; }

    [data-testid="stMetric"] {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

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
        white-space: pre-wrap;
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
        white-space: pre-wrap;
    }

    .info-box {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-family: var(--mono);
        font-size: 12px;
        color: #94a3b8;
        white-space: pre-wrap;
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
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
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
        raise ValueError(f"Raw CSV missing required sensor columns: {missing}")

    label_col = infer_column(df, LABEL_COL_CANDIDATES)
    image_col = infer_column(df, IMAGE_NAME_COL_CANDIDATES)

    if label_col is None:
        raise ValueError(f"Could not find label column. Tried: {LABEL_COL_CANDIDATES}")
    if image_col is None:
        raise ValueError(f"Could not find image-name column. Tried: {IMAGE_NAME_COL_CANDIDATES}")

    return df, label_col, image_col


def pick_one_image_per_folder(df, label_col, image_col, image_base_path, window_size):
    image_base = Path(image_base_path)
    if not image_base.exists():
        raise FileNotFoundError(f"Image base path does not exist: {image_base_path}")

    selected = {}

    for label in ["NoGas", "Smoke", "Mixture", "Perfume"]:
        label_dir = image_base / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing folder: {label_dir}")

        images = sorted(
            [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )

        if not images:
            raise FileNotFoundError(f"No images found in folder: {label_dir}")

        chosen = None
        for image_path in images:
            image_name = image_path.stem
            matches = df[
                (df[label_col].astype(str) == str(label)) &
                (df[image_col].astype(str) == str(image_name))
            ]

            if matches.empty:
                continue

            valid_matches = [idx for idx in matches.index.tolist() if idx >= window_size - 1]
            if valid_matches:
                chosen = image_path
                break

        if chosen is None:
            raise ValueError(
                f"Could not find any usable image in folder '{label}' with at least "
                f"{window_size} rows of prior sensor history in the CSV."
            )

        selected[label] = chosen

    return selected


def find_matching_target_row(df, label_col, image_col, label, image_path, window_size):
    image_name = image_path.stem

    matches = df[
        (df[label_col].astype(str) == str(label)) &
        (df[image_col].astype(str) == str(image_name))
    ].copy()

    if matches.empty:
        raise ValueError(f"No CSV row found for label='{label}', image_name='{image_name}'.")

    valid_indices = [idx for idx in matches.index.tolist() if idx >= window_size - 1]
    if not valid_indices:
        raise ValueError(
            f"CSV rows were found for label='{label}', image_name='{image_name}', "
            f"but none have enough earlier history for a {window_size}-step window."
        )

    target_idx = valid_indices[0]
    return target_idx, image_name


def build_window_rows(df, target_idx, window_size=20):
    start_idx = target_idx - window_size + 1
    if start_idx < 0:
        raise ValueError(
            f"Not enough earlier rows to build a {window_size}-step window for target_idx={target_idx}."
        )

    window_df = df.iloc[start_idx:target_idx + 1].copy()
    if len(window_df) != window_size:
        raise ValueError(f"Expected window size {window_size}, got {len(window_df)}.")
    return window_df


# =========================================================
# MODEL / AGENT LOADING
# =========================================================
@st.cache_resource(show_spinner=False)
def load_agent(dqn_model_path, ae_model_path, yolo_model_path, window_size, enable_explanations):
    decision = DecisionTool(
        dqn_model_path,
        device="cpu",
        mc_dropout_samples=5,
        window_size=window_size,
    )

    anomaly = AnomalyTool(ae_model_path)
    vision = VisionTool(yolo_model_path)
    explainer = ExplanationTool("gemma3:1b") if enable_explanations else None
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

    marker_colors = []
    for c, op in zip(colors, opacities):
        marker_colors.append(
            f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},{op})"
        )

    fig = go.Figure(go.Bar(
        x=[f"A{i}\n{ACTIONS[i][:8]}" for i in range(5)],
        y=q_values,
        marker_color=marker_colors,
        marker_line_color=colors,
        marker_line_width=1.5,
        text=[f"{v:.2f}" for v in q_values],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"Q-Values — {label}", font=dict(size=13, color="#f59e0b")),
        height=260,
        yaxis_title="Q-Value",
    )
    return fig


def action_distribution_chart(logs):
    counts = pd.Series([r["action"] for r in logs]).value_counts().sort_index()

    fig = go.Figure(go.Bar(
        x=[f"{i}: {ACTIONS[i]}" for i in counts.index],
        y=counts.values,
        marker_color=[ACTION_COLORS.get(i, "#64748b") for i in counts.index],
        marker_line_color="#1e3a5f",
        marker_line_width=1,
        text=counts.values,
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Final Action Distribution", font=dict(size=13, color="#f59e0b")),
        height=280,
    )
    return fig


def confidence_histogram(logs):
    df_plot = pd.DataFrame(logs)
    fig = go.Figure()

    for is_corr, color, name in [(True, "#22c55e", "Correct"), (False, "#ef4444", "Wrong")]:
        sub = df_plot[df_plot["is_correct"] == is_corr]["policy_confidence"]
        if sub.empty:
            continue
        fig.add_trace(go.Histogram(
            x=sub,
            name=name,
            marker_color=color,
            opacity=0.7,
            nbinsx=20,
            marker_line_color="#1e3a5f",
            marker_line_width=0.5,
        ))

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Policy Confidence Distribution", font=dict(size=13, color="#f59e0b")),
        height=260,
        barmode="overlay",
    )
    return fig


def anomaly_scatter(logs):
    df_plot = pd.DataFrame(logs)
    fig = go.Figure()

    for lbl, color in GAS_COLORS.items():
        sub = df_plot[df_plot["label"] == lbl]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["anomaly_normalized"],
            y=sub["action"],
            mode="markers",
            name=lbl,
            marker=dict(color=color, size=8, opacity=0.75, line=dict(color=color, width=0.5)),
        ))

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Anomaly vs Final Action", font=dict(size=13, color="#f59e0b")),
        height=300,
    )
    fig.update_xaxes(title_text="Normalized Anomaly")
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(5)),
        ticktext=[f"{i}:{ACTIONS[i][:10]}" for i in range(5)],
    )
    return fig


# =========================================================
# RUNNER
# =========================================================
def run_single_folder_case(
    agent,
    df,
    label_col,
    image_col,
    label,
    image_path,
    window_size,
    use_mc_dropout,
    enable_explanations,
    enable_critique,
):
    target_idx, image_name = find_matching_target_row(
        df=df,
        label_col=label_col,
        image_col=image_col,
        label=label,
        image_path=image_path,
        window_size=window_size,
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
        raise RuntimeError(f"Agent did not produce a ready final result for {label}.")

    return {
        "label": label,
        "image_name": image_name,
        "image_path": str(image_path),
        "target_idx": int(target_idx),

        "action_raw": final_result.get("action_raw"),
        "action_raw_name": final_result.get("action_raw_name"),

        "action_after_safety": final_result.get("action_after_safety"),
        "action_after_safety_name": final_result.get("action_after_safety_name"),

        "action": final_result.get("action"),
        "action_name": final_result.get("action_name"),

        "gas_id": final_result.get("gas_id"),
        "is_correct": final_result.get("is_correct"),
        "expected_actions": final_result.get("expected_actions"),

        "anomaly_raw": final_result.get("anomaly_raw"),
        "anomaly_normalized": final_result.get("anomaly_normalized"),
        "policy_confidence": final_result.get("policy_confidence"),
        "reward": final_result.get("reward"),
        "latency_ms": float(final_result.get("latency", 0.0) * 1000.0),

        "yolo_class_id": final_result.get("yolo_class_id"),
        "yolo_class_label": final_result.get("yolo_class_label"),
        "yolo_confidence": final_result.get("yolo_confidence"),
        "yolo_semantic_gas_id": final_result.get("yolo_semantic_gas_id"),
        "yolo_gas_name": final_result.get("yolo_gas_name"),
        "vision_action_support": final_result.get("vision_action_support"),
        "vision_danger_flag": final_result.get("vision_danger_flag"),
        "vision_reason": final_result.get("vision_reason"),
        "vision_error": final_result.get("vision_error"),

        "safety_changed_action": final_result.get("safety_changed_action"),
        "vision_escalated_action": final_result.get("vision_escalated_action"),

        "q_values": final_result.get("q_values"),
        "q_std": final_result.get("q_std"),
        "explanation": final_result.get("explanation"),
        "critique": final_result.get("critique"),
    }


# =========================================================
# UI
# =========================================================
inject_css()

status_dot = '<span class="status-dot status-online"></span>' if IMPORTS_OK else '<span class="status-dot status-offline"></span>'
status_text = "SYSTEM ONLINE" if IMPORTS_OK else "IMPORT ERROR"

st.markdown(f"""
<div class="hud-header">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <p class="hud-title">⚡ GasSafe AI — Control Dashboard</p>
      <p class="hud-subtitle">CENTRAL MULTIMODAL AGENT · FOLDER-DRIVEN LIVE TEST · SENSOR + VISION VERIFICATION</p>
    </div>
    <div style="text-align:right;">
      <div style="font-family:'Share Tech Mono',monospace; font-size:11px; color:#64748b; letter-spacing:0.15em;">
        {status_dot}<span style="color:#{'22c55e' if IMPORTS_OK else 'ef4444'};">{status_text}</span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#475569; margin-top:4px;">
        GAS STATION SAFETY INTELLIGENCE v3.0
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not IMPORTS_OK:
    st.error(f"Failed to import model tools: {IMPORT_ERROR}")
    st.info("Run this from the project root with the environment activated.")
    st.stop()

with st.sidebar:
    st.markdown('<p class="section-header">⚙ Configuration</p>', unsafe_allow_html=True)

    dqn_model_path = st.text_input("DQN Model Path", value="models/DeepQnet.pth")
    ae_model_path = st.text_input("AE Model Path", value="models/lstm_autoencoder_weights.pth")
    yolo_model_path = st.text_input("YOLO Model Path", value="models/yolov8_gas_classifier.pt")

    raw_csv_path = st.text_input(
        "Raw CSV Path",
        value=r"C:\Users\HP\Downloads\archive (7)\Multimodal Dataset for Gas Detection and Classification\Gas Sensors Measurements\Gas_Sensors_Measurements.csv",
    )
    image_base_path = st.text_input(
        "Image Folder Path",
        value=r"C:\Users\HP\Downloads\archive (7)\Multimodal Dataset for Gas Detection and Classification\Thermal Camera Images",
    )

    st.markdown('<p class="section-header">▶ Run Settings</p>', unsafe_allow_html=True)
    window_size = st.slider("Window Size", 5, 50, 20, 1)
    use_mc = st.checkbox("MC Dropout (slower)", value=True)
    show_expl = st.checkbox("Show Explanations", value=True)
    show_crit = st.checkbox("Show Critique", value=True)

    st.markdown('<p class="section-header">◈ Filter</p>', unsafe_allow_html=True)
    filter_label = st.selectbox("Filter by label", ["All", "NoGas", "Smoke", "Mixture", "Perfume"])
    show_only_wrong = st.checkbox("Show only wrong predictions", value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#475569; line-height:1.8;">
    GAS_MAP<br>
    ├ NoGas   → 0 → Monitor<br>
    ├ Smoke   → 1 → Raise Alarm<br>
    ├ Mixture → 2 → Emergency Shutdown<br>
    └ Perfume → 3 → Increase Sampling / Request Verification
    </div>
    """, unsafe_allow_html=True)

path_checks = {
    "DQN model": Path(dqn_model_path).exists(),
    "AE model": Path(ae_model_path).exists(),
    "YOLO model": Path(yolo_model_path).exists(),
    "Raw CSV": Path(raw_csv_path).exists(),
    "Image folder": Path(image_base_path).exists(),
}

missing_items = [name for name, ok in path_checks.items() if not ok]

if missing_items:
    for item in missing_items:
        st.error(f"{item} not found.")
run_clicked = st.button(
    "▶ RUN FOLDER LIVE TEST" if not missing_items else "⚠ FIX FILE PATHS",
    disabled=bool(missing_items),
    use_container_width=True,
)

# quick static overview
if Path(raw_csv_path).exists():
    try:
        df_raw_preview, label_col_preview, _ = load_raw_dataframe(raw_csv_path)
        if label_col_preview:
            counts = df_raw_preview[label_col_preview].value_counts()
            cols = st.columns(min(4, len(counts)))
            for col_, (lbl, cnt) in zip(cols, counts.items()):
                color = GAS_COLORS.get(lbl, "#64748b")
                with col_:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-card-label">{lbl}</div>
                      <div class="metric-card-value" style="color:{color};">{cnt}</div>
                      <div class="metric-card-sub">raw rows</div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception:
        pass

st.markdown("---")

if run_clicked or ("dashboard_results" in st.session_state):
    if run_clicked or "dashboard_results" not in st.session_state:
        with st.spinner("Loading central multimodal agent..."):
            agent = load_agent(
                dqn_model_path,
                ae_model_path,
                yolo_model_path,
                window_size,
                show_expl,
            )

        with st.spinner("Loading raw CSV and matching images..."):
            df, label_col, image_col = load_raw_dataframe(raw_csv_path)
            chosen_images = pick_one_image_per_folder(
                df=df,
                label_col=label_col,
                image_col=image_col,
                image_base_path=image_base_path,
                window_size=window_size,
            )

        all_results = []
        progress = st.progress(0, text="Running folder-driven live test...")

        items = list(chosen_images.items())
        for i, (label, image_path) in enumerate(items):
            result = run_single_folder_case(
                agent=agent,
                df=df,
                label_col=label_col,
                image_col=image_col,
                label=label,
                image_path=image_path,
                window_size=window_size,
                use_mc_dropout=use_mc,
                enable_explanations=show_expl,
                enable_critique=show_crit,
            )
            all_results.append(result)
            progress.progress((i + 1) / len(items), text=f"Processed {label}")

        progress.empty()
        st.session_state["dashboard_results"] = all_results

    logs = st.session_state["dashboard_results"]

    if filter_label != "All":
        logs = [r for r in logs if r["label"] == filter_label]
    if show_only_wrong:
        logs = [r for r in logs if not r["is_correct"]]

    if not logs:
        st.warning("No results match the current filters.")
        st.stop()

    df_logs = pd.DataFrame(logs)

    total = len(df_logs)
    correct = int(df_logs["is_correct"].sum())
    accuracy = correct / total if total else 0.0
    avg_conf = float(df_logs["policy_confidence"].mean()) if total else 0.0
    avg_latency = float(df_logs["latency_ms"].mean()) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Cases", total)
    with c2:
        st.metric("Correct", correct, f"{accuracy*100:.2f}%")
    with c3:
        st.metric("Avg Policy Confidence", f"{avg_conf:.4f}")
    with c4:
        st.metric("Avg Latency", f"{avg_latency:.2f} ms")

    tab1, tab2, tab3 = st.tabs(["Overview", "Case Details", "Structured Table"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(action_distribution_chart(logs), use_container_width=True)
            st.plotly_chart(confidence_histogram(logs), use_container_width=True)
        with c2:
            st.plotly_chart(anomaly_scatter(logs), use_container_width=True)

        st.markdown('<p class="section-header">Selected Images</p>', unsafe_allow_html=True)
        for r in logs:
            st.markdown(f"""
            <div class="label-card">
              <div class="label-card-title" style="color:{GAS_COLORS.get(r['label'], '#94a3b8')};">
                {r['label']} — {r['image_name']}
              </div>
              <div style="font-family:'Share Tech Mono',monospace; font-size:12px; color:#94a3b8; margin-top:8px;">
                Raw action: {r['action_raw_name']}<br>
                After safety: {r['action_after_safety_name']}<br>
                Final action: {r['action_name']}<br>
                Correct: {'YES' if r['is_correct'] else 'NO'}<br>
                Vision escalated: {r['vision_escalated_action']}<br>
                Safety changed: {r['safety_changed_action']}
              </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<p class="section-header">Per-Case Breakdown</p>', unsafe_allow_html=True)

        for r in logs:
            label_color = GAS_COLORS.get(r["label"], "#94a3b8")
            correct_symbol = "✅" if r["is_correct"] else "❌"

            with st.expander(f"{correct_symbol} {r['label']} — {r['image_name']} — Final: {r['action_name']}"):
                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown(f"""
                    <div class="info-box">
Label: {r['label']}
Image: {r['image_name']}
CSV target index: {r['target_idx']}

Raw action: {r['action_raw_name']}
Action after safety: {r['action_after_safety_name']}
Final action: {r['action_name']}

Correct: {r['is_correct']}
Expected actions: {r['expected_actions']}

Normalized anomaly: {r['anomaly_normalized']:.6f}
Raw anomaly: {r['anomaly_raw']}
Policy confidence: {r['policy_confidence']:.6f}
Reward: {r['reward']:.4f}
Latency: {r['latency_ms']:.3f} ms

YOLO raw class id: {r['yolo_class_id']}
YOLO raw class label: {r['yolo_class_label']}
YOLO confidence: {r['yolo_confidence']}
YOLO mapped gas id: {r['yolo_semantic_gas_id']}
YOLO mapped gas name: {r['yolo_gas_name']}

Vision support for raw action: {r['vision_action_support']}
Vision danger flag: {r['vision_danger_flag']}
Safety changed action: {r['safety_changed_action']}
Vision escalated action: {r['vision_escalated_action']}
                    </div>
                    """, unsafe_allow_html=True)

                    if r["vision_error"]:
                        st.error(f"Vision error: {r['vision_error']}")
                    else:
                        st.markdown(f"""
                        <div class="info-box">
Vision reason:
{r['vision_reason']}
                        </div>
                        """, unsafe_allow_html=True)

                with right:
                    if r["q_values"] is not None:
                        st.plotly_chart(
                            q_bar_chart(r["q_values"], r["action"], r["label"]),
                            use_container_width=True,
                        )

                if show_expl and r["explanation"]:
                    st.markdown('<p class="section-header">Explanation</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="expl-box">{r["explanation"]}</div>', unsafe_allow_html=True)

                if show_crit and r["critique"]:
                    st.markdown('<p class="section-header">Critique</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="crit-box">{r["critique"]}</div>', unsafe_allow_html=True)

    with tab3:
        table_df = df_logs.copy()

        keep_cols = [
            "label", "image_name", "target_idx",
            "action_raw_name", "action_after_safety_name", "action_name",
            "is_correct", "policy_confidence", "anomaly_normalized", "reward",
            "yolo_class_label", "yolo_confidence", "yolo_gas_name",
            "vision_action_support", "vision_danger_flag",
            "safety_changed_action", "vision_escalated_action", "latency_ms"
        ]
        table_df = table_df[keep_cols]
        st.dataframe(table_df, use_container_width=True)

        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results CSV",
            data=csv_bytes,
            file_name="dashboard_folder_live_test_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
