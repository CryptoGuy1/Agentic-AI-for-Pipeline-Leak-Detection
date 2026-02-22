class GoalManager:
    def __init__(self):
        self.goal = "Maintain industrial safety while minimizing false shutdowns"

    def evaluate_risk(self, state):
        anomaly, gas_class, conf, temp, hum = state

        if anomaly > 0.85:
            return "HIGH_RISK"
        elif anomaly > 0.6:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"

    def get_goal(self):
        return self.goal