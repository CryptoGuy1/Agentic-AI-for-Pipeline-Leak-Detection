from src.agent.safety import safety_override
from src.agent.critic_agent import CriticAgent
from src.agent.reward_system import compute_reward
from utils.logger import log_run
import numpy as np
import time


class MultimodalAgent:

    def __init__(
        self,
        vision_tool,
        anomaly_tool,
        decision_tool,
        explanation_tool,
        memory,
        goal_manager
    ):

        self.vision = vision_tool
        self.anomaly = anomaly_tool
        self.decision = decision_tool
        self.explainer = explanation_tool

        self.memory = memory
        self.goal_manager = goal_manager

        self.critic = CriticAgent(self.explainer)

        # categorical encoding
        self.gas_map = {
            "NoGas": 0,
            "Smoke": 1,
            "Mixture": 2,
            "Perfume": 3
        }

    # --------------------------------------------------
    # SINGLE AGENT CYCLE
    # --------------------------------------------------
    def run_once(self, image, sensor_array, temp, hum, step):

        start_time = time.time()

        # 1️⃣ Perception
        gas_class, conf = self.vision.predict(image)
        anomaly_score = self.anomaly.compute(sensor_array)

        # 2️⃣ Encode categorical gas class
        gas_id = self.gas_map.get(gas_class, 0)

        # 3️⃣ Build RL state
        state = [
            anomaly_score,
            gas_id,
            conf,
            temp / 100,
            hum / 100
        ]

        # 4️⃣ RL Decision
        action, q_values = self.decision.decide(state)

        # 5️⃣ Safety Override
        action = safety_override(state, action)

        # 6️⃣ Reward
        reward = compute_reward(state, action)

        # 7️⃣ Explanation
        anomaly, gas_id, conf, temp, hum = state

        explanation = self.explainer.explain(f"""
You are an industrial methane safety AI.

System readings:

Anomaly score: {anomaly}
Gas class ID: {gas_id}
Detection confidence: {conf}
Temperature: {temp*100} °C
Humidity: {hum*100} %

RL policy output:
Action chosen: {action}
Q-values: {q_values}

Actions meaning:
0 = Monitor
1 = Increase sampling
2 = Request verification
3 = Raise alarm
4 = Emergency shutdown

Explain why the chosen action is appropriate.
""")

        # 8️⃣ Critic review
        critique = self.critic.critique(
            state,
            action,
            q_values
        )

        # 9️⃣ Store Memory
        self.memory.add({
            "step": step,
            "state": state,
            "action": action,
            "reward": reward,
            "explanation": explanation,
            "critique": critique
        })

        # 🔟 Self Training
        if len(self.memory.get_recent()) > 20:
            from src.agent.trainer import AgentTrainer

            trainer = AgentTrainer(self.decision)
            trainer.train(self.memory)

        # 11️⃣ Latency
        latency = time.time() - start_time

        # 12️⃣ Logging
        log_run({
            "step": step,
            "state": state,
            "action": action,
            "reward": reward,
            "latency": latency,
            "explanation": explanation,
            "critic_review": critique
        })

        # 13️⃣ Return results
        return {
            "step": step,
            "state": state,
            "action": action,
            "q_values": q_values,
            "reward": reward,
            "explanation": explanation,
            "critique": critique
        }

    # --------------------------------------------------
    # MULTIPLE AGENT CYCLES
    # --------------------------------------------------

    def run_cycles(self, image, sensor_array, temp, hum, cycles=10):

        results = []

        rewards = []
        latencies = []
        alerts = 0
        false_alerts = 0

        for step in range(cycles):

            result = self.run_once(
                image=image,
                sensor_array=sensor_array,
                temp=temp,
                hum=hum,
                step=step
            )

            results.append(result)

            rewards.append(result["reward"])
            latencies.append(result.get("latency", 0))

            # alert detection
            if result["action"] >= 3:
                alerts += 1

                anomaly = result["state"][0]

                if anomaly < 0.3:
                    false_alerts += 1

            time.sleep(1)

        # -------- Episode Metrics --------

        episode_reward = sum(rewards)

        average_latency = np.mean(latencies)

        false_alert_rate = (
            false_alerts / alerts if alerts > 0 else 0
        )

        # policy entropy
        action_counts = {}

        for r in results:
            a = r["action"]
            action_counts[a] = action_counts.get(a, 0) + 1

        probs = [c / len(results) for c in action_counts.values()]

        policy_entropy = -sum(p * np.log(p) for p in probs)

        episode_metrics = {
            "episode_reward": episode_reward,
            "average_latency": float(average_latency),
            "false_alert_rate": false_alert_rate,
            "policy_entropy": float(policy_entropy)
        }

        log_run({
            "episode_metrics": episode_metrics
        })

        return results