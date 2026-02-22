import time
from src.agent.memory import ShortTermMemory
from src.agent.goal_manager import GoalManager
from src.agent.safety import safety_override


class MultimodalAgent:

    def __init__(self, vision_tool, anomaly_tool, decision_tool, explanation_tool, cycles=10):

        self.vision = vision_tool
        self.anomaly = anomaly_tool
        self.decision = decision_tool
        self.explainer = explanation_tool

        self.memory = ShortTermMemory(max_size=10)
        self.goal_manager = GoalManager()

        self.cycles = cycles


    def build_explanation_prompt(self, state, action, q_values, step):

        anomaly, gas_class, conf, temp, hum = state
        risk_level = self.goal_manager.evaluate_risk(state)

        prompt = f"""
        Industrial Safety Agent Report

        Cycle: {step}
        Goal: {self.goal_manager.get_goal()}
        Risk Level: {risk_level}

        State:
        - Anomaly Score: {anomaly:.3f}
        - Gas Class: {gas_class}
        - Detection Confidence: {conf:.3f}
        - Temperature: {temp*100:.1f}
        - Humidity: {hum*100:.1f}

        DQN Q-values: {q_values}
        Action Selected: {action}

        Explain why this action aligns with the safety objective.
        """

        return prompt


    def run_once(self, image, sensor_array, temp, hum, step):

        # 1. Perception
        gas_class, conf = self.vision.detect(image)
        anomaly_score = self.anomaly.compute(sensor_array)

        # 2. Build state
        state = [
            anomaly_score,
            gas_class,
            conf,
            temp / 100,
            hum / 100
        ]

        # 3. Decision (RL Brain)
        action, q_values = self.decision.decide(state)

        # 4. Safety Override
        action = safety_override(state, action)

        # 5. Explanation
        prompt = self.build_explanation_prompt(state, action, q_values, step)
        explanation = self.explainer.explain(prompt)

        # 6. Store in memory
        self.memory.add(state, action, explanation)

        return {
            "step": step,
            "state": state,
            "action": action,
            "q_values": q_values,
            "explanation": explanation
        }


    def run_cycles(self, image, sensor_array, temp, hum):

        results = []

        for step in range(self.cycles):

            result = self.run_once(image, sensor_array, temp, hum, step)
            results.append(result)

            time.sleep(1)  # Prevent CPU overload on Raspberry Pi

        return results