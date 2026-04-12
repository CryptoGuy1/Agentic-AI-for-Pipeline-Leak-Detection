import os
import time
from collections import deque
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.agent.safety import safety_override
from src.agent.reward_system import (
    compute_reward,
    is_correct_action,
    get_expected_action,
)


class MultimodalAgent:
    """
    Live inference agent aligned to the trained 22-feature DQN model.

    Sensor branch:
      1. Keep a rolling 20-step window of 7 gas sensors
      2. Compute anomaly from current 7-sensor reading
      3. Build 22-feature state [anomaly | current | delta | std]
      4. DQN chooses action
      5. Safety override can force safer action

    Vision branch (Stage 1; no DQN retraining):
      6. YOLO runs on attached image
      7. YOLO output is mapped to canonical gas_id by VisionTool
      8. High-confidence dangerous visual evidence can escalate final action

    IMPORTANT:
    - YOLO does NOT change the DQN input state here.
    - YOLO is only used as a post-DQN safety/verification branch.
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
        vision_tool=None,
    ):
        self.anomaly = anomaly_tool
        self.decision = decision_tool
        self.explainer = explanation_tool
        self.memory = memory
        self.goal_manager = goal_manager
        self.critic = critic
        self.vision = vision_tool

        self.window_size = window_size
        self.sensor_buffer = deque(maxlen=window_size)

        self.actions = {
            0: "monitor",
            1: "increase_sampling",
            2: "request_verification",
            3: "raise_alert",
            4: "emergency_shutdown",
        }

        self.gas_names = {
            0: "NoGas",
            1: "Smoke",
            2: "Mixture",
            3: "Perfume",
        }

        self.expected_actions = {
            0: [0],     # NoGas -> Monitor
            1: [3],     # Smoke -> Raise Alarm
            2: [4],     # Mixture -> Emergency Shutdown
            3: [1, 2],  # Perfume -> Increase Sampling / Request Verification
        }

    # =========================================================
    # BUFFER / WINDOW
    # =========================================================
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

    def get_expected_actions_for_gas(self, gas_id: Optional[int]):
        if gas_id is None:
            return []
        return self.expected_actions.get(int(gas_id), [])

    # =========================================================
    # PRECOMPUTED INTERPRETATIONS
    # =========================================================
    def classify_anomaly_level(self, anomaly: float) -> str:
        if anomaly < 0.20:
            return "very low anomaly"
        elif anomaly < 0.50:
            return "mild to moderate anomaly"
        elif anomaly < 0.75:
            return "elevated anomaly"
        return "high anomaly"

    def classify_confidence_level(self, conf: float) -> str:
        if conf < 0.20:
            return "very weak confidence"
        elif conf < 0.40:
            return "weak confidence"
        elif conf < 0.70:
            return "moderate confidence"
        return "strong confidence"

    def classify_q_gap(self, q_gap: float) -> str:
        if q_gap < 0.25:
            return "fragile decision"
        elif q_gap < 0.75:
            return "moderate separation"
        return "strong separation"

    def classify_vision_agreement(
        self,
        raw_action: int,
        final_action: int,
        vision_action_support: Optional[bool],
        vision_result: Optional[dict],
    ) -> str:
        if vision_result is None:
            return "vision unavailable"
        if vision_action_support is True:
            return "vision agrees with the raw policy choice"
        if vision_action_support is False and final_action != raw_action:
            return "vision disagreed and contributed to a changed final action"
        if vision_action_support is False:
            return "vision disagrees with the raw policy choice"
        return "vision agreement unclear"

    def classify_decision_robustness(self, policy_conf: float, q_gap: float) -> str:
        if policy_conf < 0.20 or q_gap < 0.25:
            return "fragile"
        if policy_conf < 0.70 or q_gap < 0.75:
            return "moderately robust"
        return "robust"

    def classify_decision_support(self, policy_conf: float, q_gap: float) -> str:
        if policy_conf < 0.20 and q_gap < 0.25:
            return "weakly supported"
        if policy_conf < 0.40 or q_gap < 0.75:
            return "moderately supported"
        return "strongly supported"

    def classify_counterintuitive_pattern(self, anomaly: float, final_action: int) -> str:
        if anomaly < 0.20 and final_action in [3, 4]:
            return "low anomaly but severe final action"
        if anomaly >= 0.75 and final_action in [0, 1, 2]:
            return "high anomaly but mild final action"
        return "no major anomaly-action inconsistency"

    def classify_risk_level(
        self,
        final_action: int,
        confidence_level: str,
        vision_danger_flag: bool,
        correctness: str,
    ) -> str:
        if correctness == "WRONG":
            return "elevated risk"
        if final_action == 4 or vision_danger_flag:
            return "high risk"
        if final_action == 3:
            return "moderate risk"
        if confidence_level in ["very weak confidence", "weak confidence"]:
            return "uncertain risk"
        return "low risk"

    def classify_potential_hazard(
        self,
        gas_id: Optional[int],
        final_action: int,
        vision_result: Optional[dict],
    ) -> str:
        gas_name = self.gas_names.get(gas_id, "Unknown") if gas_id is not None else "Unknown"
        if final_action == 4:
            return f"critical hazard associated with {gas_name}"
        if final_action == 3:
            return f"active operational hazard associated with {gas_name}"
        if final_action == 2:
            return f"possible but unconfirmed issue associated with {gas_name}"
        if vision_result is not None and vision_result.get("gas_name") == "NoGas":
            return "no strong visual hazard indication"
        return "no immediate hazard escalation"

    # =========================================================
    # VISION VERIFICATION
    # =========================================================
    def apply_vision_verification(
        self,
        action_after_safety: int,
        action_raw: int,
        vision_result: Optional[dict],
    ):
        """
        Vision verification policy:
        - If no vision result -> do nothing
        - If confidence < 0.80 -> log only, no escalation
        - If high-confidence Smoke -> escalate to at least Raise Alarm
        - If high-confidence Mixture -> escalate to Emergency Shutdown
        - NoGas/Perfume do not downgrade actions in Stage 1
        """
        final_action = int(action_after_safety)
        vision_reason = "No visual verification applied."
        vision_action_support = None
        vision_danger_flag = False
        yolo_semantic_gas_id = None

        if vision_result is None:
            return (
                final_action,
                yolo_semantic_gas_id,
                vision_action_support,
                vision_danger_flag,
                vision_reason,
            )

        yolo_semantic_gas_id = vision_result.get("gas_id", None)
        yolo_conf = float(vision_result.get("confidence", 0.0))

        if yolo_semantic_gas_id is None:
            vision_reason = "Vision result did not contain canonical gas_id."
            return (
                final_action,
                yolo_semantic_gas_id,
                vision_action_support,
                vision_danger_flag,
                vision_reason,
            )

        supported_actions = self.get_expected_actions_for_gas(yolo_semantic_gas_id)
        vision_action_support = int(action_raw) in supported_actions

        if yolo_conf < 0.80:
            vision_reason = (
                f"YOLO confidence low ({yolo_conf:.4f}); "
                f"vision branch logged only, no escalation applied."
            )
            return (
                final_action,
                yolo_semantic_gas_id,
                vision_action_support,
                vision_danger_flag,
                vision_reason,
            )

        if yolo_semantic_gas_id == 1:
            vision_danger_flag = True
            if final_action < 3:
                final_action = 3
                vision_reason = (
                    f"High-confidence visual evidence indicates Smoke "
                    f"(conf={yolo_conf:.4f}); escalated final action to Raise Alarm."
                )
            else:
                vision_reason = (
                    f"High-confidence visual evidence indicates Smoke "
                    f"(conf={yolo_conf:.4f}); current action already sufficiently severe."
                )

        elif yolo_semantic_gas_id == 2:
            vision_danger_flag = True
            if final_action != 4:
                final_action = 4
                vision_reason = (
                    f"High-confidence visual evidence indicates Mixture "
                    f"(conf={yolo_conf:.4f}); escalated final action to Emergency Shutdown."
                )
            else:
                vision_reason = (
                    f"High-confidence visual evidence indicates Mixture "
                    f"(conf={yolo_conf:.4f}); current action already at maximum severity."
                )

        else:
            vision_reason = (
                f"Visual branch predicted {vision_result.get('gas_name', 'Unknown')} "
                f"(conf={yolo_conf:.4f}); no downgrade policy applied in Stage 1."
            )

        return (
            final_action,
            yolo_semantic_gas_id,
            vision_action_support,
            vision_danger_flag,
            vision_reason,
        )

    # =========================================================
    # CONTRADICTION VALIDATORS
    # =========================================================
    def _contains_action_contradiction(self, text: str, final_action: int) -> bool:
        text_lower = text.lower()
        for action_id, action_name in self.actions.items():
            action_text = action_name.replace("_", " ").lower()
            if action_id != int(final_action) and action_text in text_lower:
                return True
        return False

    def _contains_numeric_label_contradiction(
        self,
        text: str,
        confidence_level: str,
        q_gap_level: str,
        vision_confidence_level: Optional[str],
    ) -> bool:
        text_lower = text.lower()

        contradiction_map = {
            "strong confidence": ["weak confidence", "low confidence", "very weak confidence"],
            "moderate confidence": ["very weak confidence"],
            "weak confidence": ["strong confidence"],
            "very weak confidence": ["strong confidence", "moderate confidence"],
            "strong separation": ["fragile decision", "weakly separated", "small gap", "almost tied"],
            "moderate separation": ["strong separation", "fragile decision"],
            "fragile decision": ["strong separation", "well separated"],
        }

        for bad_phrase in contradiction_map.get(confidence_level, []):
            if bad_phrase in text_lower:
                return True

        for bad_phrase in contradiction_map.get(q_gap_level, []):
            if bad_phrase in text_lower:
                return True

        if vision_confidence_level is not None:
            for bad_phrase in contradiction_map.get(vision_confidence_level, []):
                if bad_phrase in text_lower:
                    return True

        return False

    def _contains_semantic_contradiction(
        self,
        text: str,
        confidence_level: str,
        q_gap_level: str,
        decision_robustness: str,
        decision_support: str,
        correctness: str,
        vision_agreement: str,
        counterintuitive_pattern: str,
        vision_confidence_level: Optional[str] = None,
    ) -> bool:
        text_lower = text.lower()

        contradiction_map = {
            "strong confidence": ["weak confidence", "low confidence", "very weak confidence", "uncertain confidence"],
            "moderate confidence": ["very weak confidence", "strong confidence"],
            "weak confidence": ["strong confidence", "high confidence"],
            "very weak confidence": ["strong confidence", "high confidence", "moderate confidence"],

            "strong separation": ["fragile decision", "weak separation", "small gap", "almost tied"],
            "moderate separation": ["strong separation", "fragile decision"],
            "fragile decision": ["strong separation", "well separated", "clear margin"],

            "robust": ["fragile", "weak decision", "unstable decision"],
            "moderately robust": ["highly fragile", "fully robust"],
            "fragile": ["robust", "strongly robust"],

            "strongly supported": ["weakly supported", "poorly supported"],
            "moderately supported": ["strongly supported", "weakly supported"],
            "weakly supported": ["strongly supported", "well supported"],

            "CORRECT": ["wrong decision", "incorrect decision", "bad action", "unsafe final action"],
            "WRONG": ["correct decision", "appropriate action", "safe action"],

            "vision agrees with the raw policy choice": ["vision disagrees", "vision conflict", "vision contradicts"],
            "vision disagrees with the raw policy choice": ["vision agrees", "vision supports"],
            "vision unavailable": ["vision agrees", "vision supports", "vision confirms"],

            "no major anomaly-action inconsistency": ["counterintuitive", "anomaly-action mismatch", "inconsistent pattern"],
            "low anomaly but severe final action": ["no inconsistency"],
            "high anomaly but mild final action": ["no inconsistency"],
        }

        keys_to_check = [
            confidence_level,
            q_gap_level,
            decision_robustness,
            decision_support,
            correctness,
            vision_agreement,
            counterintuitive_pattern,
        ]

        if vision_confidence_level is not None:
            keys_to_check.append(vision_confidence_level)

        for key in keys_to_check:
            bad_phrases = contradiction_map.get(key, [])
            for phrase in bad_phrases:
                if phrase.lower() in text_lower:
                    return True

        return False

    def validate_explanation_text(
        self,
        explanation: str,
        final_action: int,
        confidence_level: str,
        q_gap_level: str,
        vision_confidence_level: Optional[str] = None,
    ) -> str:
        if not explanation:
            return explanation

        if self._contains_action_contradiction(explanation, final_action) or self._contains_numeric_label_contradiction(
            explanation,
            confidence_level=confidence_level,
            q_gap_level=q_gap_level,
            vision_confidence_level=vision_confidence_level,
        ):
            allowed_action_name = self.actions[int(final_action)].replace("_", " ")
            return (
                "Structured explanation fallback\n\n"
                "1. Sensor state interpretation\n"
                f"- The only valid final action in the structured output is '{allowed_action_name}'.\n"
                f"- Policy confidence is classified as '{confidence_level}'.\n"
                f"- Q-gap is classified as '{q_gap_level}'.\n\n"
                "2. Policy interpretation\n"
                "- The original LLM explanation was not fully grounded and has been replaced.\n\n"
                "3. Vision branch interpretation\n"
                "- Use the logged vision fields as the source of truth.\n\n"
                "4. Branch agreement analysis\n"
                "- Raw action, safety-adjusted action, and final action in the result dictionary are authoritative.\n\n"
                "5. Final action analysis\n"
                f"- The correct final action for this step is '{allowed_action_name}'.\n\n"
                "6. Operational honesty check\n"
                "- Structured outputs are more trustworthy than the rejected free-form response.\n\n"
                "7. Bottom line\n"
                "- Review the structured fields directly."
            )

        return explanation

    def validate_critique_text(
        self,
        critique: str,
        final_action: int,
        confidence_level: str,
        q_gap_level: str,
        decision_robustness: str,
        decision_support: str,
        correctness: str,
        vision_agreement: str,
        counterintuitive_pattern: str,
        vision_confidence_level: Optional[str] = None,
    ) -> str:
        if not critique:
            return critique

        action_contradiction = self._contains_action_contradiction(
            critique,
            final_action=final_action,
        )

        semantic_contradiction = self._contains_semantic_contradiction(
            critique,
            confidence_level=confidence_level,
            q_gap_level=q_gap_level,
            decision_robustness=decision_robustness,
            decision_support=decision_support,
            correctness=correctness,
            vision_agreement=vision_agreement,
            counterintuitive_pattern=counterintuitive_pattern,
            vision_confidence_level=vision_confidence_level,
        )

        if action_contradiction or semantic_contradiction:
            allowed_action_name = self.actions[int(final_action)].replace("_", " ")
            return (
                "Structured critique fallback\n\n"
                "1. Confidence audit\n"
                f"- Valid confidence classification: '{confidence_level}'.\n"
                f"- Valid q-gap classification: '{q_gap_level}'.\n\n"
                "2. Q-value audit\n"
                f"- Decision support classification: '{decision_support}'.\n"
                f"- Decision robustness classification: '{decision_robustness}'.\n\n"
                "3. Anomaly-vs-action audit\n"
                f"- Counterintuitive pattern assessment: '{counterintuitive_pattern}'.\n"
                f"- The only valid final action is '{allowed_action_name}'.\n\n"
                "4. Vision audit\n"
                f"- Vision agreement assessment: '{vision_agreement}'.\n"
                f"- Vision confidence classification: '{vision_confidence_level if vision_confidence_level is not None else 'unavailable'}'.\n\n"
                "5. Decision robustness audit\n"
                f"- Correctness label: '{correctness}'.\n"
                "- The free-form critique was rejected because it contradicted the structured evidence.\n\n"
                "6. Safety risk audit\n"
                "- Use q-values, anomaly, confidence, and vision fields directly from the structured result.\n\n"
                "7. Critic verdict\n"
                f"Verdict: Trust the structured outputs over the rejected critique text. Final action = '{allowed_action_name}'."
            )

        return critique

    # =========================================================
    # PROMPT BUILDERS
    # =========================================================
    def build_grounded_explanation_prompt(
        self,
        state,
        action_raw,
        action_after_safety,
        final_action,
        policy_conf,
        q_values,
        gas_id,
        is_correct,
        vision_result,
        vision_error,
        vision_action_support,
        vision_danger_flag,
        vision_reason,
    ) -> tuple[str, dict]:
        correct_str = (
            "CORRECT" if is_correct
            else ("WRONG" if is_correct is False else "UNKNOWN")
        )

        gas_name = self.gas_names.get(gas_id, "Unknown") if gas_id is not None else "Unknown"
        q_values_rounded = np.round(q_values, 4).tolist()

        q_arr = np.array(q_values, dtype=float)
        ranked_idx = np.argsort(q_arr)[::-1]
        top_idx = int(ranked_idx[0])
        second_idx = int(ranked_idx[1]) if len(ranked_idx) > 1 else top_idx
        q_gap = float(q_arr[top_idx] - q_arr[second_idx]) if len(ranked_idx) > 1 else 0.0

        anomaly_level = self.classify_anomaly_level(float(state[0]))
        confidence_level = self.classify_confidence_level(float(policy_conf))
        q_gap_level = self.classify_q_gap(float(q_gap))
        decision_robustness = self.classify_decision_robustness(float(policy_conf), float(q_gap))
        decision_support = self.classify_decision_support(float(policy_conf), float(q_gap))
        counterintuitive_pattern = self.classify_counterintuitive_pattern(float(state[0]), int(final_action))

        if vision_result is not None:
            vision_conf = float(vision_result.get("confidence", 0.0))
            vision_conf_level = self.classify_confidence_level(vision_conf)
            vision_agreement = self.classify_vision_agreement(
                raw_action=int(action_raw),
                final_action=int(final_action),
                vision_action_support=vision_action_support,
                vision_result=vision_result,
            )
            vision_text = (
                f"Vision branch:\n"
                f"- gas_name={vision_result.get('gas_name')}\n"
                f"- gas_id={vision_result.get('gas_id')}\n"
                f"- raw_yolo_idx={vision_result.get('yolo_class_idx')}\n"
                f"- raw_yolo_label={vision_result.get('yolo_class_label')}\n"
                f"- confidence={vision_conf:.6f}\n"
                f"- confidence_level={vision_conf_level}\n"
                f"- agreement={vision_agreement}\n"
                f"- supports_raw_dqn_action={vision_action_support}\n"
                f"- danger_flag={vision_danger_flag}\n"
                f"- vision_note={vision_reason}\n"
            )
        elif vision_error is not None:
            vision_conf_level = None
            vision_agreement = "vision unavailable"
            vision_text = f"Vision branch error:\n- {vision_error}\n"
        else:
            vision_conf_level = None
            vision_agreement = "vision unavailable"
            vision_text = "Vision branch:\n- unavailable\n"

        metadata = {
            "confidence_level": confidence_level,
            "q_gap_level": q_gap_level,
            "vision_confidence_level": vision_conf_level,
        }

        prompt = f"""
You are an industrial gas safety analyst writing an explanation for a logged AI decision.

Your job is to explain EXACTLY what happened in this step.
You must be honest, specific, and grounded in the provided values.
Do NOT invent facts.
Do NOT mention any final action other than: {self.actions[int(final_action)]}.

IMPORTANT:
- Treat the structured fields below as the only source of truth.
- The final action is already chosen. You are not choosing an action.
- Use the precomputed interpretation labels exactly as provided.
- Do not reinterpret the numeric values differently from those labels.
- Output bullets only.
- One bullet per claim.
- No paragraphs.

WRITE THE RESPONSE USING THESE EXACT SECTION HEADINGS:

1. Sensor state interpretation
2. Policy interpretation
3. Vision branch interpretation
4. Branch agreement analysis
5. Final action analysis
6. Operational honesty check
7. Bottom line

Each section must contain 2-4 bullet points.

STRUCTURED FIELDS

Ground truth:
- gas_id={gas_id}
- gas_name={gas_name}
- correctness={correct_str}

Sensor/DQN branch:
- normalized_anomaly={float(state[0]):.6f}
- anomaly_level={anomaly_level}
- raw_dqn_action={int(action_raw)} ({self.actions[int(action_raw)]})
- action_after_safety={int(action_after_safety)} ({self.actions[int(action_after_safety)]})
- final_action={int(final_action)} ({self.actions[int(final_action)]})
- policy_confidence={float(policy_conf):.6f}
- confidence_level={confidence_level}
- q_values={q_values_rounded}
- top_q_action={top_idx} ({self.actions[int(top_idx)]})
- second_q_action={second_idx} ({self.actions[int(second_idx)]})
- q_gap={q_gap:.6f}
- q_gap_level={q_gap_level}
- decision_robustness={decision_robustness}
- decision_support={decision_support}
- counterintuitive_pattern={counterintuitive_pattern}
- safety_changed_action={int(action_after_safety) != int(action_raw)}
- vision_changed_action={int(final_action) != int(action_after_safety)}

{vision_text}

STRICT REQUIREMENTS
- Mention the actual final action explicitly.
- Mention whether safety override changed the raw action.
- Mention whether vision changed the action.
- Use the exact labels: confidence_level, q_gap_level, decision_robustness, decision_support.
- If confidence_level is "strong confidence", do not describe confidence as weak or low.
- If q_gap_level is "strong separation", do not describe the decision as fragile or weakly separated.
- If vision confidence is low, say visual evidence was not trusted.
- If counterintuitive_pattern is not "no major anomaly-action inconsistency", mention it explicitly.
- Be concise and factual.

Now write the explanation.
""".strip()

        return prompt, metadata

    def build_grounded_critique_prompt(
        self,
        state,
        action_raw,
        action_after_safety,
        final_action,
        policy_conf,
        q_values,
        gas_id,
        is_correct,
        vision_result,
        vision_error,
        vision_action_support,
        vision_danger_flag,
        vision_reason,
    ) -> tuple[str, dict]:
        correct_str = (
            "CORRECT" if is_correct
            else ("WRONG" if is_correct is False else "UNKNOWN")
        )

        gas_name = self.gas_names.get(gas_id, "Unknown") if gas_id is not None else "Unknown"
        q_values_rounded = np.round(q_values, 4).tolist()

        q_arr = np.array(q_values, dtype=float)
        ranked_idx = np.argsort(q_arr)[::-1]
        top_idx = int(ranked_idx[0])
        second_idx = int(ranked_idx[1]) if len(ranked_idx) > 1 else top_idx
        q_gap = float(q_arr[top_idx] - q_arr[second_idx]) if len(ranked_idx) > 1 else 0.0

        anomaly_level = self.classify_anomaly_level(float(state[0]))
        confidence_level = self.classify_confidence_level(float(policy_conf))
        q_gap_level = self.classify_q_gap(float(q_gap))
        decision_robustness = self.classify_decision_robustness(float(policy_conf), float(q_gap))
        decision_support = self.classify_decision_support(float(policy_conf), float(q_gap))
        counterintuitive_pattern = self.classify_counterintuitive_pattern(float(state[0]), int(final_action))

        if vision_result is not None:
            vision_conf = float(vision_result.get("confidence", 0.0))
            vision_conf_level = self.classify_confidence_level(vision_conf)
            vision_agreement = self.classify_vision_agreement(
                raw_action=int(action_raw),
                final_action=int(final_action),
                vision_action_support=vision_action_support,
                vision_result=vision_result,
            )
            vision_text = (
                f"Vision branch:\n"
                f"- gas_name={vision_result.get('gas_name')}\n"
                f"- gas_id={vision_result.get('gas_id')}\n"
                f"- raw_yolo_idx={vision_result.get('yolo_class_idx')}\n"
                f"- raw_yolo_label={vision_result.get('yolo_class_label')}\n"
                f"- confidence={vision_conf:.6f}\n"
                f"- confidence_level={vision_conf_level}\n"
                f"- agreement={vision_agreement}\n"
                f"- supports_raw_dqn_action={vision_action_support}\n"
                f"- danger_flag={vision_danger_flag}\n"
                f"- vision_note={vision_reason}\n"
            )
        elif vision_error is not None:
            vision_conf_level = None
            vision_agreement = "vision unavailable"
            vision_text = f"Vision branch error:\n- {vision_error}\n"
        else:
            vision_conf_level = None
            vision_agreement = "vision unavailable"
            vision_text = "Vision branch:\n- unavailable\n"

        risk_level = self.classify_risk_level(
            final_action=int(final_action),
            confidence_level=confidence_level,
            vision_danger_flag=vision_danger_flag,
            correctness=correct_str,
        )
        potential_hazard = self.classify_potential_hazard(
            gas_id=gas_id,
            final_action=int(final_action),
            vision_result=vision_result,
        )

        metadata = {
            "confidence_level": confidence_level,
            "q_gap_level": q_gap_level,
            "vision_confidence_level": vision_conf_level,
            "decision_robustness": decision_robustness,
            "decision_support": decision_support,
            "correctness": correct_str,
            "vision_agreement": vision_agreement,
            "counterintuitive_pattern": counterintuitive_pattern,
        }

        prompt = f"""
You are an industrial gas safety critic reviewing a logged AI decision.

Your job is to critique the quality of the final decision honestly.
Be direct.
Be technical.
Be faithful to the numbers.
Do NOT invent facts.
Do NOT mention any final action other than: {self.actions[int(final_action)]}.

IMPORTANT:
- Use the precomputed interpretation labels exactly as provided.
- Do not reinterpret the numeric values differently from those labels.
- Output bullets only.
- One bullet per claim.
- No paragraphs.

WRITE THE RESPONSE USING THESE EXACT SECTION HEADINGS:

1. Confidence audit
2. Q-value audit
3. Anomaly-vs-action audit
4. Vision audit
5. Decision robustness audit
6. Safety risk audit
7. Critic verdict

Each section must contain 2-4 bullet points.

STRUCTURED FIELDS

Ground truth:
- gas_id={gas_id}
- gas_name={gas_name}
- correctness={correct_str}

Sensor/DQN branch:
- normalized_anomaly={float(state[0]):.6f}
- anomaly_level={anomaly_level}
- raw_dqn_action={int(action_raw)} ({self.actions[int(action_raw)]})
- action_after_safety={int(action_after_safety)} ({self.actions[int(action_after_safety)]})
- final_action={int(final_action)} ({self.actions[int(final_action)]})
- policy_confidence={float(policy_conf):.6f}
- confidence_level={confidence_level}
- q_values={q_values_rounded}
- top_q_action={top_idx} ({self.actions[int(top_idx)]})
- second_q_action={second_idx} ({self.actions[int(second_idx)]})
- q_gap={q_gap:.6f}
- q_gap_level={q_gap_level}
- decision_robustness={decision_robustness}
- decision_support={decision_support}
- counterintuitive_pattern={counterintuitive_pattern}
- safety_changed_action={int(action_after_safety) != int(action_raw)}
- vision_changed_action={int(final_action) != int(action_after_safety)}

{vision_text}

Derived audit fields:
- risk_level={risk_level}
- potential_hazard={potential_hazard}

STRICT REQUIREMENTS
- Use the exact labels: confidence_level, q_gap_level, decision_robustness, decision_support.
- If confidence_level is "strong confidence", do not call it weak.
- If q_gap_level is "strong separation", do not call it fragile.
- If decision_robustness is "robust", do not call the decision fragile or unstable.
- If decision_support is "strongly supported", do not call the decision weakly supported.
- If vision confidence is low, say visual evidence should not be heavily trusted.
- If vision confidence is high and dangerous, say it strengthens the safety case.
- If correctness is TRUE but confidence is weak, say the action was right but fragile.
- If correctness is FALSE, say the operational consequence clearly.
- The last bullet under section 7 must begin exactly with: "Verdict:"
- Be concise and factual.

Now write the critique.
""".strip()

        return prompt, metadata

    # =========================================================
    # MAIN STEP
    # =========================================================
    def run_once(
        self,
        sensor_row,
        step: int,
        gas_id: Optional[int] = None,
        image_path: Optional[str] = None,
        use_mc_dropout: bool = False,
        enable_explanations: bool = False,
        enable_critique: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.time()

        self.update_sensor_buffer(sensor_row)

        if not self.is_ready():
            return {
                "step": step,
                "ready": False,
                "message": f"Filling window: {len(self.sensor_buffer)}/{self.window_size}",
                "image_path": image_path,
            }

        sensor_window = self.get_sensor_window()
        current_row = sensor_window[-1]

        anomaly_score_raw = float(self.anomaly.compute(current_row))

        state = self.decision.build_state_from_window(
            sensor_window=sensor_window,
            anomaly_score_raw=anomaly_score_raw,
        )

        action_raw, q_values, q_std, policy_conf = self.decision.decide(
            state=state,
            use_mc_dropout=use_mc_dropout,
        )

        action_after_safety = int(safety_override(state.tolist(), int(action_raw)))

        vision_result = None
        vision_error = None

        if image_path is not None and self.vision is not None:
            try:
                if os.path.exists(image_path):
                    vision_result = self.vision.predict(image_path)
                else:
                    vision_error = f"Image path does not exist: {image_path}"
            except Exception as e:
                vision_error = f"YOLO inference failed: {str(e)}"

        (
            final_action,
            yolo_semantic_gas_id,
            vision_action_support,
            vision_danger_flag,
            vision_reason,
        ) = self.apply_vision_verification(
            action_after_safety=action_after_safety,
            action_raw=int(action_raw),
            vision_result=vision_result,
        )

        reward = compute_reward(
            state=state,
            action=int(final_action),
            gas_id=gas_id,
            anomaly=float(state[0]),
        )

        is_correct = is_correct_action(gas_id, final_action) if gas_id is not None else None
        expected_actions = get_expected_action(gas_id) if gas_id is not None else []

        explanation = "disabled"
        if enable_explanations and self.explainer is not None:
            explanation_prompt, explanation_meta = self.build_grounded_explanation_prompt(
                state=state,
                action_raw=action_raw,
                action_after_safety=action_after_safety,
                final_action=final_action,
                policy_conf=policy_conf,
                q_values=q_values,
                gas_id=gas_id,
                is_correct=is_correct,
                vision_result=vision_result,
                vision_error=vision_error,
                vision_action_support=vision_action_support,
                vision_danger_flag=vision_danger_flag,
                vision_reason=vision_reason,
            )
            explanation = self.explainer.explain(explanation_prompt)
            explanation = self.validate_explanation_text(
                explanation=explanation,
                final_action=int(final_action),
                confidence_level=explanation_meta["confidence_level"],
                q_gap_level=explanation_meta["q_gap_level"],
                vision_confidence_level=explanation_meta["vision_confidence_level"],
            )

        critique = "disabled"
        if enable_critique:
            critique_prompt, critique_meta = self.build_grounded_critique_prompt(
                state=state,
                action_raw=action_raw,
                action_after_safety=action_after_safety,
                final_action=final_action,
                policy_conf=policy_conf,
                q_values=q_values,
                gas_id=gas_id,
                is_correct=is_correct,
                vision_result=vision_result,
                vision_error=vision_error,
                vision_action_support=vision_action_support,
                vision_danger_flag=vision_danger_flag,
                vision_reason=vision_reason,
            )

            if self.critic is not None:
                critique = self.critic.critique(critique_prompt)
            elif self.explainer is not None:
                critique = self.explainer.explain(critique_prompt)

            critique = self.validate_critique_text(
                critique=critique,
                final_action=int(final_action),
                confidence_level=critique_meta["confidence_level"],
                q_gap_level=critique_meta["q_gap_level"],
                decision_robustness=critique_meta["decision_robustness"],
                decision_support=critique_meta["decision_support"],
                correctness=critique_meta["correctness"],
                vision_agreement=critique_meta["vision_agreement"],
                counterintuitive_pattern=critique_meta["counterintuitive_pattern"],
                vision_confidence_level=critique_meta["vision_confidence_level"],
            )

        latency = time.time() - t0

        result = {
            "step": step,
            "ready": True,
            "state": state.tolist(),

            "action_raw": int(action_raw),
            "action_raw_name": self.actions[int(action_raw)],

            "action_after_safety": int(action_after_safety),
            "action_after_safety_name": self.actions[int(action_after_safety)],

            "action": int(final_action),
            "action_name": self.actions[int(final_action)],

            "gas_id": gas_id,
            "is_correct": is_correct,
            "expected_actions": expected_actions,

            "q_values": q_values.tolist(),
            "q_std": q_std.tolist(),
            "policy_confidence": float(policy_conf),

            "anomaly_raw": float(anomaly_score_raw),
            "anomaly_normalized": float(state[0]),

            "reward": reward,

            "image_path": image_path,
            "vision_error": vision_error,
            "yolo_class_id": vision_result.get("yolo_class_idx") if vision_result is not None else None,
            "yolo_class_label": vision_result.get("yolo_class_label") if vision_result is not None else None,
            "yolo_confidence": vision_result.get("confidence") if vision_result is not None else None,
            "yolo_semantic_gas_id": yolo_semantic_gas_id,
            "yolo_gas_name": vision_result.get("gas_name") if vision_result is not None else None,
            "vision_action_support": vision_action_support,
            "vision_danger_flag": vision_danger_flag,
            "vision_reason": vision_reason,

            "safety_changed_action": int(action_after_safety) != int(action_raw),
            "vision_escalated_action": int(final_action) > int(action_after_safety),

            "explanation": explanation,
            "critique": critique,
            "latency": float(latency),
        }

        if self.memory is not None:
            self.memory.add(result)

        return result
