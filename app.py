import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import altair as alt

from src.agent.agent_core import MultimodalAgent
from src.tools.vision_tool import VisionTool
from src.tools.anomaly_tool import AnomalyTool
from src.tools.decision_tool import DecisionTool
from src.tools.explanation_tool import ExplanationTool
from src.agent.memory import ShortTermMemory as AgentMemory
from src.agent.goal_manager import GoalManager


# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Methane Control Room",
    layout="wide"
)

# ---------- REMOVE STREAMLIT WHITESPACE ----------
st.markdown("""
<style>

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.4rem;
}

header {visibility: hidden;}

.stMetric {
    padding:0rem;
}

</style>
""", unsafe_allow_html=True)

st.title("Methane Industrial Control Room")


# ----------------------------------------------------
# LOAD AGENT
# ----------------------------------------------------

@st.cache_resource
def load_agent():

    vision = VisionTool("models/yolov8_gas_classifier.pt")
    anomaly = AnomalyTool("models/lstm_autoencoder_weights.pth")
    decision = DecisionTool("models/drf_gas_model.pth")
    explain = ExplanationTool("gemma:2b")

    memory = AgentMemory()
    goal_manager = GoalManager()

    return MultimodalAgent(
        vision_tool=vision,
        anomaly_tool=anomaly,
        decision_tool=decision,
        explanation_tool=explain,
        memory=memory,
        goal_manager=goal_manager
    )

agent = load_agent()


# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "incidents" not in st.session_state:
    st.session_state.incidents = []

if "cycle" not in st.session_state:
    st.session_state.cycle = 0


# ----------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------

st.sidebar.header("System Control")

run = st.sidebar.button("Start Monitoring")

temp = st.sidebar.number_input("Temperature °C", 32.0)
hum = st.sidebar.number_input("Humidity %", 55.0)


# ----------------------------------------------------
# DASHBOARD GRID (COMPACT)
# ----------------------------------------------------

df = pd.DataFrame(st.session_state.history)

# ---------- ROW 1 ----------
r1c1,r1c2,r1c3,r1c4 = st.columns(4)

with r1c1:
    st.metric("Agent","OK")

with r1c2:
    st.metric("Sensors","LIVE")

with r1c3:
    st.metric("Gemma","CONNECTED")

with r1c4:
    st.metric("Cycles",st.session_state.cycle)



# ---------- ROW 2 ----------
r2c1,r2c2,r2c3 = st.columns([1,1,1])

# THREAT GAUGE
with r2c1:

    threat_level = 0

    if st.session_state.history:
        threat_level = min(100, st.session_state.history[-1]["anomaly"]*100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=threat_level,
        gauge={"axis":{"range":[0,100]}}
    ))

    fig.update_layout(height=250,margin=dict(l=10,r=10,t=10,b=10))

    st.plotly_chart(fig,use_container_width=True)


# THREAT RADAR
with r2c2:

    if not df.empty:

        radar = go.Figure()

        radar.add_trace(go.Scatterpolar(
            r=[df.iloc[-1]["anomaly"]*100,50,30],
            theta=["Anomaly","Confidence","Risk"],
            fill='toself'
        ))

        radar.update_layout(height=250,margin=dict(l=10,r=10,t=10,b=10))

        st.plotly_chart(radar,use_container_width=True)


# CAMERA
with r2c3:

    st.image(
        "C:\\Users\\INTERNAL AUDIT\\Downloads\\999_Smoke.png",
        use_column_width=True
    )


# ---------- ROW 3 ----------
r3c1,r3c2 = st.columns(2)

# TELEMETRY
with r3c1:

    if not df.empty:

        chart = alt.Chart(df).mark_line().encode(
            x="cycle",
            y="anomaly"
        )

        st.altair_chart(chart,use_container_width=True)


# HEATMAP
with r3c2:

    sensor_array = np.random.rand(50,7)

    heatmap = go.Figure(data=go.Heatmap(
        z=sensor_array,
        colorscale="Inferno"
    ))

    heatmap.update_layout(height=250,margin=dict(l=10,r=10,t=10,b=10))

    st.plotly_chart(heatmap,use_container_width=True)



# ---------- ROW 4 ----------
r4c1,r4c2 = st.columns(2)

# AI DECISION
with r4c1:

    if not df.empty:

        last = df.iloc[-1]

        st.write("STATE:",last["state"])
        st.write("ACTION:",last["action"])
        st.write("REWARD:",last["reward"])



# GEMMA REASONING
with r4c2:

    if st.button("Explain Last Decision"):

        if not df.empty:

            last = df.iloc[-1]

            prompt=f"""
Explain methane monitoring decision.

state={last['state']}
action={last['action']}
reward={last['reward']}
"""

            explanation = agent.explainer.explain(prompt)

            st.info(explanation)



# ----------------------------------------------------
# RUN LOOP
# ----------------------------------------------------

if run:

    sensor_array = np.random.rand(50,7)

    image = "C:\\Users\\INTERNAL AUDIT\\Downloads\\999_Smoke.png"

    result = agent.run_once(
        image=image,
        sensor_array=sensor_array,
        temp=temp,
        hum=hum,
        step=st.session_state.cycle
    )

    anomaly = result["state"][0]

    row = {
        "cycle":st.session_state.cycle,
        "anomaly":anomaly,
        "confidence":result["state"][2],
        "state":result["state"],
        "action":result["action"],
        "reward":result["reward"]
    }

    st.session_state.history.append(row)

    if anomaly > 0.5:
        st.session_state.incidents.append({
            "cycle":st.session_state.cycle,
            "event":"Methane anomaly detected"
        })

    st.session_state.cycle += 1

    time.sleep(1)

    st.rerun()