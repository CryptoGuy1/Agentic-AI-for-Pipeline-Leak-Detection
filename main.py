import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.anomaly_tool import AnomalyTool
from src.tools.decision_tool import DecisionTool
from src.tools.explanation_tool import ExplanationTool
from src.tools.vision_tool import VisionTool
from src.agent.agent_core import MultimodalAgent
from src.agent.memory import ShortTermMemory
from src.agent.goal_manager import GoalManager


DQN_MODEL_PATH = "models/DeepQnet.pth"
AE_MODEL_PATH = "models/lstm_autoencoder_weights.pth"
YOLO_MODEL_PATH = "models/yolov8_gas_classifier.pt"

RUN_MODE = "folder_live_test"

RAW_CSV_PATH = r"C:\Users\HP\Downloads\archive (7)\Multimodal Dataset for Gas Detection and Classification\Gas Sensors Measurements\Gas_Sensors_Measurements.csv"
IMAGE_BASE_PATH = r"C:\Users\HP\Downloads\archive (7)\Multimodal Dataset for Gas Detection and Classification\Thermal Camera Images"

WINDOW_SIZE = 20
USE_MC_DROPOUT = True
ENABLE_EXPLANATIONS = True
ENABLE_CRITIQUE = True

RUN_ANOMALY_DIAGNOSTICS = True
RUN_ANOMALY_SENSITIVITY_TEST = True

SAVE_LOG_PATH = "folder_live_test_log.csv"
SAVE_SUMMARY_PATH = "folder_live_test_summary.csv"

ACTIONS = {
    0: "Monitor",
    1: "Increase Sampling",
    2: "Request Verification",
    3: "Raise Alarm",
    4: "Emergency Shutdown",
}

GAS_MAP = {
    "NoGas": 0,
    "Smoke": 1,
    "Mixture": 2,
    "Perfume": 3,
}

RAW_SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]

IMAGE_NAME_COL_CANDIDATES = [
    "Corresponding Image Name",
    "corresponding_image_name",
    "image_name",
    "Image Name",
]

LABEL_COL_CANDIDATES = [
    "Gas",
    "label",
    "Label",
    "class",
    "Class",
]


print("Initializing tools...")

decision = DecisionTool(
    DQN_MODEL_PATH,
    device="cpu",
    mc_dropout_samples=5,
    window_size=WINDOW_SIZE,
)

anomaly = AnomalyTool(AE_MODEL_PATH)
vision = VisionTool(YOLO_MODEL_PATH)
explainer = ExplanationTool("gemma3:1b") if ENABLE_EXPLANATIONS else None

critic = None

memory = ShortTermMemory(max_size=200)
goal_manager = GoalManager()

agent = MultimodalAgent(
    anomaly_tool=anomaly,
    decision_tool=decision,
    explanation_tool=explainer,
    memory=memory,
    goal_manager=goal_manager,
    critic=critic,
    window_size=WINDOW_SIZE,
    vision_tool=vision,
)

print("Tools initialized successfully.")


def infer_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def row_to_sensor_array(row):
    return [float(row[c]) for c in RAW_SENSOR_COLS]


def get_true_gas_id(label: str) -> int:
    if label not in GAS_MAP:
        raise ValueError(
            f"Unknown label '{label}'. Must be one of: {list(GAS_MAP.keys())}"
        )
    return GAS_MAP[label]


def load_raw_dataframe():
    df = pd.read_csv(RAW_CSV_PATH)

    missing = [c for c in RAW_SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing required sensor columns: {missing}")

    label_col = infer_column(df, LABEL_COL_CANDIDATES)
    image_col = infer_column(df, IMAGE_NAME_COL_CANDIDATES)

    if label_col is None:
        raise ValueError(f"Could not find label column. Tried: {LABEL_COL_CANDIDATES}")
    if image_col is None:
        raise ValueError(f"Could not find image-name column. Tried: {IMAGE_NAME_COL_CANDIDATES}")

    return df, label_col, image_col


def pick_one_image_per_folder(df, label_col, image_col, image_base_path, window_size):
    image_base = Path(image_base_path)
    if not image_base.exists():
        raise FileNotFoundError(f"Image base path does not exist: {image_base_path}")

    selected = {}

    for label in ["NoGas", "Smoke", "Mixture", "Perfume"]:
        label_dir = image_base / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing folder: {label_dir}")

        images = sorted(
            [
                p for p in label_dir.iterdir()
                if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )

        if not images:
            raise FileNotFoundError(f"No images found in folder: {label_dir}")

        chosen = None

        for image_path in images:
            image_name = image_path.stem

            matches = df[
                (df[label_col].astype(str) == str(label)) &
                (df[image_col].astype(str) == str(image_name))
            ]

            if matches.empty:
                continue

            valid_matches = [idx for idx in matches.index.tolist() if idx >= window_size - 1]

            if valid_matches:
                chosen = image_path
                break

        if chosen is None:
            raise ValueError(
                f"Could not find any usable image in folder '{label}' with at least "
                f"{window_size} rows of prior sensor history in the CSV."
            )

        selected[label] = chosen

    return selected


def find_matching_target_row(df, label_col, image_col, label, image_path, window_size):
    image_name = image_path.stem

    matches = df[
        (df[label_col].astype(str) == str(label)) &
        (df[image_col].astype(str) == str(image_name))
    ].copy()

    if matches.empty:
        raise ValueError(
            f"No CSV row found for label='{label}', image_name='{image_name}'."
        )

    valid_indices = [idx for idx in matches.index.tolist() if idx >= window_size - 1]

    if not valid_indices:
        raise ValueError(
            f"CSV rows were found for label='{label}', image_name='{image_name}', "
            f"but none have enough earlier history for a {window_size}-step window. "
            f"Matching indices: {matches.index.tolist()}"
        )

    target_idx = valid_indices[0]
    return target_idx, image_name


def build_window_rows(df, target_idx, window_size=20):
    start_idx = target_idx - window_size + 1
    if start_idx < 0:
        raise ValueError(
            f"Not enough earlier rows to build a {window_size}-step window for target_idx={target_idx}."
        )

    window_df = df.iloc[start_idx:target_idx + 1].copy()

    if len(window_df) != window_size:
        raise ValueError(
            f"Expected window size {window_size}, got {len(window_df)}."
        )

    return window_df


def inspect_anomaly_context(result_row):
    print("\nANOMALY DIAGNOSTIC")
    print("-" * 40)
    print(f"Label: {result_row['label']}")
    print(f"Image: {result_row['image_name']}")
    print(f"Anomaly raw: {result_row['anomaly_raw']}")
    print(f"Anomaly normalized: {result_row['anomaly_normalized']}")
    print(f"Final action: {result_row['action_name']}")
    print(f"Policy confidence: {result_row['policy_confidence']:.4f}")
    if result_row["q_values"] is not None:
        print(f"Q-values: {np.round(np.array(result_row['q_values']), 4)}")
    print("-" * 40)


# =========================================================
# FIX A — CORRECT ANOMALY SENSITIVITY TEST
# =========================================================
def test_anomaly_sensitivity(agent, sensor_window, normalized_anomaly_values):
    """
    Test DQN sensitivity by directly overriding the normalized anomaly slot
    in the already-built state, instead of passing fake raw AE values into
    build_state_from_window(...).
    """
    print("\nANOMALY SENSITIVITY TEST")
    print("-" * 50)

    base_raw_anomaly = float(agent.anomaly.compute(sensor_window[-1]))
    base_state = agent.decision.build_state_from_window(
        sensor_window=sensor_window,
        anomaly_score_raw=base_raw_anomaly,
    ).astype(np.float32)

    for test_anom in normalized_anomaly_values:
        state = base_state.copy()
        state[0] = float(test_anom)

        action, q_values, q_std, policy_conf = agent.decision.decide(
            state=state,
            use_mc_dropout=False,
        )

        print(
            f"anomaly={state[0]:.4f} | "
            f"action={action} ({agent.actions[int(action)]}) | "
            f"conf={float(policy_conf):.4f} | "
            f"q={np.round(q_values, 4)}"
        )


def run_single_folder_case(df, label_col, image_col, label, image_path):
    target_idx, image_name = find_matching_target_row(
        df=df,
        label_col=label_col,
        image_col=image_col,
        label=label,
        image_path=image_path,
        window_size=WINDOW_SIZE,
    )

    window_df = build_window_rows(df, target_idx, window_size=WINDOW_SIZE)
    true_gas_id = get_true_gas_id(label)

    print("\n" + "#" * 90)
    print(f"TESTING LABEL: {label}")
    print(f"IMAGE: {image_path.name}")
    print(f"CSV TARGET INDEX: {target_idx}")
    print(f"WINDOW ROWS: {window_df.index.min()} -> {window_df.index.max()}")

    agent.reset_window()
    final_result = None

    for i, (_, row) in enumerate(window_df.iterrows()):
        sensor_array = row_to_sensor_array(row)
        final_image_path = str(image_path) if i == (WINDOW_SIZE - 1) else None

        result = agent.run_once(
            sensor_row=sensor_array,
            step=i,
            gas_id=true_gas_id,
            image_path=final_image_path,
            use_mc_dropout=USE_MC_DROPOUT,
            enable_explanations=ENABLE_EXPLANATIONS,
            enable_critique=ENABLE_CRITIQUE,
        )
        final_result = result

    if final_result is None or not final_result.get("ready", False):
        raise RuntimeError(f"Agent did not produce a ready final result for {label}.")

    if RUN_ANOMALY_SENSITIVITY_TEST:
        sensor_window_array = agent.get_sensor_window()
        test_anomaly_values = [0.0, 0.25, 0.50, 0.75, 1.0]
        test_anomaly_sensitivity(agent, sensor_window_array, test_anomaly_values)

    return {
        "label": label,
        "image_name": image_name,
        "image_path": str(image_path),
        "target_idx": int(target_idx),

        "action_raw": final_result.get("action_raw"),
        "action_raw_name": final_result.get("action_raw_name"),

        "action_after_safety": final_result.get("action_after_safety"),
        "action_after_safety_name": final_result.get("action_after_safety_name"),

        "action": final_result.get("action"),
        "action_name": final_result.get("action_name"),

        "gas_id": final_result.get("gas_id"),
        "is_correct": final_result.get("is_correct"),
        "expected_actions": final_result.get("expected_actions"),

        "anomaly_raw": final_result.get("anomaly_raw"),
        "anomaly_normalized": final_result.get("anomaly_normalized"),
        "policy_confidence": final_result.get("policy_confidence"),
        "reward": final_result.get("reward"),
        "latency_ms": float(final_result.get("latency", 0.0) * 1000.0),

        "yolo_class_id": final_result.get("yolo_class_id"),
        "yolo_class_label": final_result.get("yolo_class_label"),
        "yolo_confidence": final_result.get("yolo_confidence"),
        "yolo_semantic_gas_id": final_result.get("yolo_semantic_gas_id"),
        "yolo_gas_name": final_result.get("yolo_gas_name"),
        "vision_action_support": final_result.get("vision_action_support"),
        "vision_danger_flag": final_result.get("vision_danger_flag"),
        "vision_reason": final_result.get("vision_reason"),
        "vision_error": final_result.get("vision_error"),

        "safety_changed_action": final_result.get("safety_changed_action"),
        "vision_escalated_action": final_result.get("vision_escalated_action"),

        "q_values": final_result.get("q_values"),
        "explanation": final_result.get("explanation"),
        "critique": final_result.get("critique"),
    }


def print_case_result(r):
    correct_str = "✅" if r["is_correct"] else "❌"

    print("=" * 70)
    print(f"LABEL: {r['label']}  {correct_str}")
    print(f"IMAGE: {r['image_name']}")
    print(f"ACTION RAW: {r['action_raw_name']}")
    print(f"ACTION AFTER SAFETY: {r['action_after_safety_name']}")
    print(f"FINAL ACTION: {r['action_name']}")
    print(f"SAFETY CHANGED ACTION: {r['safety_changed_action']}")
    print(f"VISION ESCALATED ACTION: {r['vision_escalated_action']}")
    print(f"ANOMALY NORM: {r['anomaly_normalized']:.6f}")
    print(f"POLICY CONF: {r['policy_confidence']:.4f}")

    if r["q_values"] is not None:
        print(f"Q-VALUES: {np.round(np.array(r['q_values']), 4)}")

    if r["yolo_class_id"] is not None:
        print(
            f"YOLO -> raw_idx={r['yolo_class_id']} | "
            f"raw_label={r['yolo_class_label']} | "
            f"conf={r['yolo_confidence']:.4f} | "
            f"mapped_gas_id={r['yolo_semantic_gas_id']} ({r['yolo_gas_name']})"
        )
        print(f"VISION NOTE: {r['vision_reason']}")
    elif r["vision_error"]:
        print(f"VISION ERROR: {r['vision_error']}")

    print(f"REWARD: {r['reward']:.4f}")
    print(f"LATENCY: {r['latency_ms']:.3f} ms")

    if ENABLE_EXPLANATIONS and r["explanation"]:
        print(f"EXPLANATION:\n{r['explanation']}")
    if ENABLE_CRITIQUE and r["critique"]:
        print(f"CRITIQUE:\n{r['critique']}")

    if RUN_ANOMALY_DIAGNOSTICS:
        inspect_anomaly_context(r)


def run_folder_live_test():
    print("\nRunning folder-driven live test...")

    df, label_col, image_col = load_raw_dataframe()

    print(f"Using label column: {label_col}")
    print(f"Using image-name column: {image_col}")

    chosen_images = pick_one_image_per_folder(
        df=df,
        label_col=label_col,
        image_col=image_col,
        image_base_path=IMAGE_BASE_PATH,
        window_size=WINDOW_SIZE,
    )

    print("\nSelected images for testing:")
    for lbl, img in chosen_images.items():
        print(f"  {lbl}: {img.name}")

    all_results = []

    for label, image_path in chosen_images.items():
        result = run_single_folder_case(
            df=df,
            label_col=label_col,
            image_col=image_col,
            label=label,
            image_path=image_path,
        )
        all_results.append(result)
        print_case_result(result)

    log_df = pd.DataFrame(all_results)
    log_df.to_csv(SAVE_LOG_PATH, index=False)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "label": r["label"],
            "image_name": r["image_name"],
            "target_idx": r["target_idx"],
            "final_action": r["action"],
            "final_action_name": r["action_name"],
            "is_correct": r["is_correct"],
            "policy_confidence": r["policy_confidence"],
            "reward": r["reward"],
            "yolo_class_id": r["yolo_class_id"],
            "yolo_class_label": r["yolo_class_label"],
            "yolo_confidence": r["yolo_confidence"],
            "yolo_semantic_gas_id": r["yolo_semantic_gas_id"],
            "yolo_gas_name": r["yolo_gas_name"],
            "vision_danger_flag": r["vision_danger_flag"],
            "safety_changed_action": r["safety_changed_action"],
            "vision_escalated_action": r["vision_escalated_action"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SAVE_SUMMARY_PATH, index=False)

    print("\n" + "=" * 90)
    print("FINAL FOLDER TEST SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print(f"\nDetailed log  → {SAVE_LOG_PATH}")
    print(f"Summary table → {SAVE_SUMMARY_PATH}")


if __name__ == "__main__":
    if RUN_MODE == "folder_live_test":
        run_folder_live_test()
    else:
        raise ValueError("RUN_MODE must be 'folder_live_test'")
