import streamlit as st
import numpy as np
import time

from src.tools.vision_tool import VisionTool
from src.tools.anomaly_tool import AnomalyTool
from src.tools.decision_tool import DecisionTool
from src.tools.explanation_tool import ExplanationTool
from src.agent.agent_core import MultimodalAgent

st.set_page_config(layout="wide")

st.title("🧠 Multimodal Industrial Safety Agent")
st.markdown("Real-Time AI Agent with RL + Anomaly Detection + Vision + LLM Reasoning")

# -------------------------------------------------
# Load Models (Cached — very important for Pi)
# -------------------------------------------------
@st.cache_resource
def load_agent():
    vision = VisionTool("models/yolo_gas_classifier.pt")
    anomaly = AnomalyTool("models/lstm_autoencoder.pth")
    decision = DecisionTool("models/dqn_model.pth")
    explainer = ExplanationTool("gemma:2b")

    agent = MultimodalAgent(
        vision=vision,
        anomaly=anomaly,
        decision=decision,
        explainer=explainer,
        cycles=10
    )

    return agent

agent = load_agent()

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    image_path = st.text_input("Image Path", "test.jpg")
    temperature = st.number_input("Temperature (°C)", value=32.0)
    humidity = st.number_input("Humidity (%)", value=55.0)

with col2:
    st.markdown("### Sensor Simulation")
    sensor_data = np.random.rand(50, 7)
    st.write("Synthetic sensor matrix shape:", sensor_data.shape)

# -------------------------------------------------
# RUN AGENT
# -------------------------------------------------
if st.button("Run 10 Cycles"):

    start_time = time.time()

    try:
        results = agent.run_cycles(
            image=image_path,
            sensor_array=sensor_data,
            temp=temperature,
            hum=humidity
        )

        runtime = time.time() - start_time

        st.success(f"Agent Execution Completed in {runtime:.2f} seconds")

        for r in results:
            with st.expander(f"Cycle {r['step']}"):
                st.write("State:", r["state"])
                st.write("Action:", r["action"])
                st.write("Q-values:", r["q_values"])
                st.write("Explanation:")
                st.info(r["explanation"])

    except Exception as e:
        st.error(f"Agent crashed: {e}")

# -------------------------------------------------
# LIVE CHAT WITH GEMMA
# -------------------------------------------------
st.markdown("---")
st.header("💬 Ask the AI About Decisions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the system")

if st.button("Send Question"):

    if user_input.strip() != "":
        try:
            response = agent.explainer.explain(user_input)  # reuse loaded model

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", response))

        except Exception as e:
            st.error(f"Gemma error: {e}")

# Display Chat
for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")