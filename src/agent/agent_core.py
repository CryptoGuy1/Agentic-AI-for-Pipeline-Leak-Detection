import time
from collections import deque
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.agent.safety import safety_override

# ── FIX: import corrected reward with gas_id support ─────────
from src.agent.reward_system import (
    compute_reward,
    is_correct_action,
    get_expected_action,
)


class MultimodalAgent:
    """
    Live inference agent aligned to the trained 22-feature DQN model.

    Flow:
      1. Keep a rolling 20-step window of 7 gas sensors
      2. Compute anomaly from current 7-sensor reading
      3. Build 22-feature state  [anomaly | current | delta | std]
      4. DQN chooses action
      5. Safety override can force safer action
      6. Compute reward using gas_id (FIXED — old code used anomaly-only)
      7. Optional explanation / critique

    BUGS FIXED vs old version:
      - run_once() now accepts gas_id parameter and passes it to compute_reward
      - reward is now computed with gas_id → gas-class semantics match training
      - result dict now includes is_correct and expected_actions fields
    """

    def __init__(
        self,
        anomaly_tool,
        decision_tool,
        explanation_tool,
        memory,
        goal_manager,
        critic=None,
        window_size: int = 20,
    ):
        self.anomaly      = anomaly_tool
        self.decision     = decision_tool
        self.explainer    = explanation_tool
        self.memory       = memory
        self.goal_manager = goal_manager
        self.critic       = critic

        self.window_size   = window_size
        self.sensor_buffer = deque(maxlen=window_size)

        self.actions = {
            0: "monitor",
            1: "increase_sampling",
            2: "request_verification",
            3: "raise_alert",
            4: "emergency_shutdown",
        }

    def reset_window(self) -> None:
        self.sensor_buffer.clear()

    def update_sensor_buffer(self, sensor_row: Iterable[float]) -> None:
        sensor_row = np.asarray(list(sensor_row), dtype=np.float32)
        if sensor_row.shape[0] != 7:
            raise ValueError(
                f"Expected 7 sensor values per row, got {sensor_row.shape[0]}"
            )
        self.sensor_buffer.append(sensor_row)

    def is_ready(self) -> bool:
        return len(self.sensor_buffer) == self.window_size

    def get_sensor_window(self) -> np.ndarray:
        if not self.is_ready():
            raise ValueError(
                f"Need {self.window_size} rows; have {len(self.sensor_buffer)}"
            )
        return np.stack(self.sensor_buffer, axis=0).astype(np.float32)

    def run_once(
        self,
        sensor_row,
        step: int,
        gas_id: Optional[int] = None,   # ← FIX: new parameter
        use_mc_dropout: bool = False,
        enable_explanations: bool = False,
        enable_critique: bool = False,
    ) -> Dict[str, Any]:
        """
        Process one new sensor reading and return the agent's decision.

        Args:
            sensor_row      : iterable of 7 float sensor values
                              [MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135]
            step            : current timestep index
            gas_id          : int ground-truth gas class (0-3) if known.
                              ALWAYS pass this in verify/evaluation mode.
                              In true blind live mode, leave as None.
            use_mc_dropout  : run MC-Dropout uncertainty estimation
            enable_explanations : generate text explanation via LLM
            enable_critique : run critic module

        Returns:
            dict with action, reward, q_values, is_correct, etc.
        """
        t0 = time.time()

        self.update_sensor_buffer(sensor_row)

        if not self.is_ready():
            return {
                "step"         : step,
                "ready"        : False,
                "message"      : f"Filling window: {len(self.sensor_buffer)}/{self.window_size}",
            }

        sensor_window  = self.get_sensor_window()
        current_row    = sensor_window[-1]

        anomaly_score_raw = float(self.anomaly.compute(current_row))

        state = self.decision.build_state_from_window(
            sensor_window=sensor_window,
            anomaly_score_raw=anomaly_score_raw,
        )

        action_raw, q_values, q_std, policy_conf = self.decision.decide(
            state=state,
            use_mc_dropout=use_mc_dropout,
        )

        action = safety_override(state.tolist(), action_raw)

        # ── FIX: use gas_id in reward computation ─────────────────────
        # Old code: compute_reward(state.tolist(), action)
        #   → used anomaly thresholds only
        #   → Mixture (anomaly≈0.10) got "Monitor" reward, not "Emergency"
        #
        # Fixed:    compute_reward(state, action, gas_id=gas_id, anomaly=state[0])
        #   → uses gas_id + action, matching training notebook exactly
        reward = compute_reward(
            state=state,
            action=int(action),
            gas_id=gas_id,              # None in blind live mode → falls back
            anomaly=float(state[0]),
        )

        # ── Correctness check ──────────────────────────────────────────
        is_correct      = is_correct_action(gas_id, action) if gas_id is not None else None
        expected_actions = get_expected_action(gas_id) if gas_id is not None else []

        # ── Optional explanation ───────────────────────────────────────
        explanation = "disabled"
        if enable_explanations and self.explainer is not None:
            correct_str = "✅ CORRECT" if is_correct else ("❌ WRONG" if is_correct is False else "unknown")
            explanation = self.explainer.explain(
                f"""
You are an industrial gas safety AI at a gas station.

Normalized anomaly: {state[0]:.4f}
Gas ID (ground truth): {gas_id} (0=NoGas, 1=Smoke, 2=Mixture, 3=Perfume)
Action selected by DQN: {action_raw} ({self.actions[int(action_raw)]})
Action after safety override: {action} ({self.actions[int(action)]})
Is action correct: {correct_str}
Policy confidence: {policy_conf:.4f}
Q-values: {np.round(q_values, 4).tolist()}

Actions:
0=Monitor  1=Increase Sampling  2=Request Verification
3=Raise Alert  4=Emergency Shutdown

Explain whether this action is appropriate and why.
""".strip()
            )

        # ── Optional critique ──────────────────────────────────────────
        critique = "disabled"
        if enable_critique and self.critic is not None:
            critique = self.critic.critique(
                state.tolist(), int(action), q_values.tolist()
            )

        latency = time.time() - t0

        result = {
            "step"              : step,
            "ready"             : True,
            "state"             : state.tolist(),
            "action_raw"        : int(action_raw),
            "action_raw_name"   : self.actions[int(action_raw)],
            "action"            : int(action),
            "action_name"       : self.actions[int(action)],
            "gas_id"            : gas_id,
            "is_correct"        : is_correct,          # ← NEW field
            "expected_actions"  : expected_actions,    # ← NEW field
            "q_values"          : q_values.tolist(),
            "q_std"             : q_std.tolist(),
            "policy_confidence" : float(policy_conf),
            "anomaly_raw"       : float(anomaly_score_raw),
            "anomaly_normalized": float(state[0]),
            "reward"            : reward,              # ← now gas_id-based
            "explanation"       : explanation,
            "critique"          : critique,
            "latency"           : float(latency),
        }

        if self.memory is not None:
            self.memory.add(result)

        return result