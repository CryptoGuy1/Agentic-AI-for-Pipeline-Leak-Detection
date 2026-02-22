from src.tools.vision_tool import VisionTool
from src.tools.anomaly_tool import AnomalyTool
from src.tools.decision_tool import DecisionTool
from src.tools.explanation_tool import ExplanationTool
from src.agent.agent_core import MultimodalAgent
import numpy as np

vision = VisionTool("models/yolo_gas_classifier.pt")
anomaly = AnomalyTool("models/lstm_autoencoder.pth", "models/minmax_scaler.save")
decision = DecisionTool("models/dqn_model.pth")
explainer = ExplanationTool("gemma:2b")

agent = MultimodalAgent(
    vision,
    anomaly,
    decision,
    explainer,
    cycles=10
)

sensor_data = np.random.rand(50,7)

results = agent.run_cycles(
    image="test.jpg",
    sensor_array=sensor_data,
    temp=32,
    hum=55
)

for r in results:
    print(r["step"], r["action"])
    print(r["explanation"])