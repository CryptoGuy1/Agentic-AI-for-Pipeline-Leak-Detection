class CriticAgent:

    def __init__(self, explanation_tool):

        self.explainer = explanation_tool

    def critique(self, state, action, q_values):

        anomaly, gas_id, conf, temp, hum = state

        prompt = f"""
You are a safety auditor reviewing an industrial methane monitoring AI.

System readings:

Anomaly score: {anomaly}
Gas class id: {gas_id}
Detection confidence: {conf}
Temperature: {temp*100} °C
Humidity: {hum*100} %

RL policy output:

Action chosen: {action}
Q-values: {q_values}

Action meanings:
0 = Monitor
1 = Increase sampling
2 = Request verification
3 = Raise alarm
4 = Emergency shutdown

Your task:

1. Determine if the action is SAFE
2. Determine if the action matches the anomaly level
3. Suggest a safer action if needed

Return format:

Safety Verdict:
Recommended Action:
Reasoning:
"""

        critique = self.explainer.explain(prompt)

        return critique