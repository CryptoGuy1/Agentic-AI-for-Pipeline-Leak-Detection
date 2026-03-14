from src.tools.vision_tool import VisionTool
from src.tools.anomaly_tool import AnomalyTool
from src.tools.decision_tool import DecisionTool
from src.tools.explanation_tool import ExplanationTool
from src.agent.agent_core import MultimodalAgent
from src.agent.memory import ShortTermMemory
from src.agent.goal_manager import GoalManager
import numpy as np

vision = VisionTool("models/yolov8_gas_classifier.pt")
anomaly = AnomalyTool("models/lstm_autoencoder_weights.pth")
decision = DecisionTool("models/drf_gas_model.pth")
explainer = ExplanationTool("gemma:2b")

memory = ShortTermMemory(max_size=50)
goal_manager = GoalManager()

agent = MultimodalAgent(
    vision_tool=vision,
    anomaly_tool=anomaly,
    decision_tool=decision,
    explanation_tool=explainer,
    memory=memory,
    goal_manager=goal_manager
)

sensor_data = np.random.rand(50,7)

results = agent.run_cycles(
    image="C:\\Users\\INTERNAL AUDIT\\Downloads\\999_Smoke.png",
    sensor_array=sensor_data,
    temp=32,
    hum=55
)

actions = {
    0: "Monitor",
    1: "Increase Sampling",
    2: "Request Verification",
    3: "Raise Alarm",
    4: "Emergency Shutdown"
}

for r in results:

    print("="*60)

    print("STEP:", r["step"])

    print("STATE:", r["state"])

    action_id = r["action"]

    print("ACTION CHOSEN:", actions.get(action_id, "Unknown"))

    print("Q VALUES:", r["q_values"])

    print("REWARD:", r["reward"])

    print("\nEXPLANATION:")
    print(r["explanation"])

    print("\nCRITIC REVIEW:")
    print(r["critique"])