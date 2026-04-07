import time
import numpy as np
import pandas as pd

from src.tools.anomaly_tool import AnomalyTool
from src.tools.decision_tool import DecisionTool
from src.tools.explanation_tool import ExplanationTool
from src.agent.agent_core import MultimodalAgent
from src.agent.memory import ShortTermMemory
from src.agent.goal_manager import GoalManager
from src.agent.reward_system import compute_reward, is_correct_action, get_expected_action


# =========================================================
# SETTINGS
# =========================================================
DQN_MODEL_PATH = "models/DeepQnet.pth"
AE_MODEL_PATH  = "models/lstm_autoencoder_weights.pth"

RUN_MODE = "live"   # "verify" or "live"

PROCESSED_TEST_DF_PATH = "test_df_processed.csv"
RAW_CSV_PATH = r"C:\Users\HP\Downloads\archive (7)\Multimodal Dataset for Gas Detection and Classification\Gas Sensors Measurements\Gas_Sensors_Measurements.csv"

WINDOW_SIZE           = 20
MAX_STEPS_PER_EPISODE = 40

USE_MC_DROPOUT      = True
ENABLE_EXPLANATIONS = True
ENABLE_CRITIQUE     = True

SAVE_LOG_PATH     = "evaluation_log.csv"
SAVE_SUMMARY_PATH = "episode_summary.csv"

ACTIONS = {
    0: "Monitor",
    1: "Increase Sampling",
    2: "Request Verification",
    3: "Raise Alarm",
    4: "Emergency Shutdown",
}

# =========================================================
# THE CANONICAL GAS_MAP
# =========================================================
# This is the ONLY source of truth for label → gas_id.
# It must match exactly what the notebook used during training.
#
# YOLO's internal class indices are DIFFERENT from this map.
# YOLO class 0 = Mixture, YOLO class 1 = NoGas, etc.
# (proved by the verify output: label=Mixture had gas_id=0 from YOLO)
#
# We NEVER use YOLO's class index as the ground-truth gas_id.
# We ALWAYS derive gas_id from the text label via this map.
#
# gas_id → correct action(s):
#   0 (NoGas)   → [0]    Monitor
#   1 (Smoke)   → [3]    Raise Alarm
#   2 (Mixture) → [4]    Emergency Shutdown
#   3 (Perfume) → [1, 2] Increase Sampling or Request Verification
GAS_MAP = {
    "NoGas":   0,
    "Smoke":   1,
    "Mixture": 2,
    "Perfume": 3,
}

CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}

RAW_SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
DELTA_COLS      = ["dMQ2", "dMQ3", "dMQ5", "dMQ6", "dMQ7", "dMQ8", "dMQ135"]
STD_COLS        = ["sMQ2", "sMQ3", "sMQ5", "sMQ6", "sMQ7", "sMQ8", "sMQ135"]


# =========================================================
# INIT TOOLS
# =========================================================
print("Initializing tools...")

decision = DecisionTool(
    DQN_MODEL_PATH,
    device="cpu",
    mc_dropout_samples=5,
    window_size=WINDOW_SIZE,
)

anomaly      = None
explainer    = None
memory       = None
goal_manager = None
agent        = None

if RUN_MODE == "live":
    anomaly      = AnomalyTool(AE_MODEL_PATH)
    explainer    = ExplanationTool("gemma3:1b") if ENABLE_EXPLANATIONS else None
    memory       = ShortTermMemory(max_size=200)
    goal_manager = GoalManager()

    agent = MultimodalAgent(
        anomaly_tool=anomaly,
        decision_tool=decision,
        explanation_tool=explainer,
        memory=memory,
        goal_manager=goal_manager,
        critic=None,
        window_size=WINDOW_SIZE,
    )

print("Tools initialized successfully.")


# =========================================================
# HELPERS
# =========================================================
def row_to_sensor_array(row):
    return [float(row[c]) for c in RAW_SENSOR_COLS]


def get_true_gas_id(label: str) -> int:
    """
    Convert a gas label string to the canonical gas_id via GAS_MAP.

    This is the ONLY correct way to get gas_id in this codebase.
    Do NOT use the gas_id column from test_df — that column was
    populated from YOLO predictions whose class ordering differs
    from GAS_MAP. YOLO class 0 = Mixture, but GAS_MAP['Mixture'] = 2.
    """
    if label not in GAS_MAP:
        raise ValueError(
            f"Unknown label '{label}'. Must be one of: {list(GAS_MAP.keys())}"
        )
    return GAS_MAP[label]


def infer_label_column(df):
    for c in ["label", "Label", "class", "Class", "gas_label", "Gas"]:
        if c in df.columns:
            return c
    return None


def infer_anomaly_column(df):
    for c in ["anomaly_norm", "anomaly", "anomaly_score", "anomaly_scaled",
              "anomaly_normalized", "anom", "anomaly_normed"]:
        if c in df.columns:
            return c
    return None


def build_state_columns_from_df(df):
    anomaly_col = infer_anomaly_column(df)
    if anomaly_col is None:
        raise ValueError("Cannot find anomaly column in processed CSV.")

    missing = (
        [c for c in RAW_SENSOR_COLS if c not in df.columns]
        + [c for c in DELTA_COLS if c not in df.columns]
        + [c for c in STD_COLS if c not in df.columns]
    )
    if missing:
        raise ValueError(f"Processed test df is missing columns: {missing}")

    return anomaly_col, [anomaly_col] + RAW_SENSOR_COLS + DELTA_COLS + STD_COLS


def summarize_verify_results(results, true_label=None):
    if not results:
        return {
            "true_label": true_label, "num_steps": 0,
            "accuracy": 0.0, "correct_count": 0,
            "expected_actions": "N/A", "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0, "mean_policy_conf": 0.0,
            "mean_reward": 0.0, "final_action": None,
            "final_action_name": None, "dominant_action": None,
            "dominant_action_name": None,
        }

    latencies_ms  = [r["latency_ms"]         for r in results]
    policy_confs  = [r["policy_confidence"]   for r in results]
    actions       = [r["action"]              for r in results]
    rewards       = [r.get("reward", 0.0)     for r in results]
    is_corrects   = [r.get("is_correct")      for r in results]

    valid_correct = [c for c in is_corrects if c is not None]
    accuracy      = float(np.mean(valid_correct)) if valid_correct else None

    dominant_action = int(pd.Series(actions).value_counts().index[0])

    return {
        "true_label"          : true_label,
        "num_steps"           : len(results),
        "accuracy"            : round(accuracy, 4) if accuracy is not None else "N/A",
        "correct_count"       : sum(1 for c in is_corrects if c),
        "expected_actions"    : results[0].get("expected_actions", "N/A"),
        "avg_latency_ms"      : float(np.mean(latencies_ms)),
        "p95_latency_ms"      : float(np.percentile(latencies_ms, 95)),
        "mean_policy_conf"    : float(np.mean(policy_confs)),
        "mean_reward"         : float(np.mean(rewards)),
        "final_action"        : int(actions[-1]),
        "final_action_name"   : ACTIONS[int(actions[-1])],
        "dominant_action"     : dominant_action,
        "dominant_action_name": ACTIONS[dominant_action],
    }


def summarize_live_results(results, true_label=None):
    ready = [r for r in results if r.get("ready", False)]
    if not ready:
        return {
            "true_label": true_label, "num_ready_steps": 0,
            "episode_reward": 0.0, "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0, "mean_policy_conf": 0.0,
            "alerts": 0, "emergency_shutdowns": 0,
            "final_action": None, "final_action_name": None,
            "dominant_action": None, "dominant_action_name": None,
        }

    latencies_ms    = [r["latency"] * 1000  for r in ready]
    rewards         = [r["reward"]           for r in ready]
    policy_confs    = [r["policy_confidence"] for r in ready]
    actions         = [r["action"]            for r in ready]
    dominant_action = int(pd.Series(actions).value_counts().index[0])

    return {
        "true_label"          : true_label,
        "num_ready_steps"     : len(ready),
        "episode_reward"      : float(np.sum(rewards)),
        "avg_latency_ms"      : float(np.mean(latencies_ms)),
        "p95_latency_ms"      : float(np.percentile(latencies_ms, 95)),
        "mean_policy_conf"    : float(np.mean(policy_confs)),
        "alerts"              : int(sum(1 for a in actions if a >= 3)),
        "emergency_shutdowns" : int(sum(1 for a in actions if a == 4)),
        "final_action"        : int(actions[-1]),
        "final_action_name"   : ACTIONS[int(actions[-1])],
        "dominant_action"     : dominant_action,
        "dominant_action_name": ACTIONS[dominant_action],
    }


# =========================================================
# MODE 1: NOTEBOOK-FAITHFUL VERIFICATION
# =========================================================
def run_verify_mode():
    print("\nRunning NOTEBOOK-FAITHFUL verification mode...")

    df = pd.read_csv(PROCESSED_TEST_DF_PATH)
    print(f"Processed test df loaded: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    anomaly_col, state_cols = build_state_columns_from_df(df)
    print(f"\nAnomaly column : {anomaly_col}")
    print(f"State columns  : {state_cols}")

    label_col = infer_label_column(df)
    print(f"Label column   : {label_col or 'NOT FOUND'}")

    # ── IMPORTANT: we do NOT use the gas_id column from test_df ──────
    # That column = YOLO prediction indices (Mixture=0, NoGas=1, etc.)
    # which differ from GAS_MAP (NoGas=0, Smoke=1, Mixture=2, Perfume=3).
    # We ALWAYS derive gas_id from the label via GAS_MAP.
    if "gas_id" in df.columns:
        print("\n⚠️  NOTE: test_df has a 'gas_id' column but it is YOLO's")
        print("   class index, NOT the semantic gas_id from GAS_MAP.")
        print("   It will be IGNORED for correctness checking.")
        print("   True gas_id is derived from label via GAS_MAP.")

    all_logs          = []
    episode_summaries = []

    if label_col is None:
        raise ValueError(
            "No label column found. Cannot determine true gas_id. "
            "Make sure test_df_processed.csv has a 'label' column."
        )

    available_labels = sorted(df[label_col].dropna().unique().tolist())
    print(f"\nAvailable labels: {available_labels}")

    # ── Validate all labels are in GAS_MAP ───────────────────────────
    unknown = [l for l in available_labels if l not in GAS_MAP]
    if unknown:
        raise ValueError(
            f"Labels not in GAS_MAP: {unknown}. "
            f"GAS_MAP keys: {list(GAS_MAP.keys())}"
        )

    for label in available_labels:
        df_label = df[df[label_col] == label].reset_index(drop=True)
        steps_to_run = min(len(df_label), MAX_STEPS_PER_EPISODE)

        # ── FIX: derive true gas_id from label via GAS_MAP ───────────
        true_gas_id = get_true_gas_id(label)
        expected    = get_expected_action(true_gas_id)
        expected_str = " or ".join(f"{a}={ACTIONS[a]}" for a in expected)

        print("\n" + "#" * 80)
        print(f"VERIFY LABEL: {label}")
        print(f"Rows: {len(df_label)}   Steps: {steps_to_run}")
        print(f"True gas_id (from GAS_MAP): {true_gas_id}  →  Expected: {expected_str}")

        results = []

        for step in range(steps_to_run):
            row   = df_label.iloc[step]
            state = row[state_cols].values.astype(np.float32)

            t0             = time.time()
            action, q_values, q_std, policy_conf = decision.decide(
                state=state, use_mc_dropout=USE_MC_DROPOUT
            )
            latency_ms = (time.time() - t0) * 1000

            # ── Use true_gas_id (from GAS_MAP), NOT row["gas_id"] ────
            is_correct = is_correct_action(true_gas_id, action)
            reward     = compute_reward(
                state=state,
                action=int(action),
                gas_id=true_gas_id,
                anomaly=float(state[0]),
            )

            result = {
                "true_label"       : label,
                "true_gas_id"      : true_gas_id,
                "step"             : step,
                "action"           : int(action),
                "action_name"      : ACTIONS[int(action)],
                "is_correct"       : is_correct,
                "expected_actions" : expected_str,
                "reward"           : reward,
                "policy_confidence": float(policy_conf),
                "latency_ms"       : float(latency_ms),
                "q0": float(q_values[0]), "q1": float(q_values[1]),
                "q2": float(q_values[2]), "q3": float(q_values[3]),
                "q4": float(q_values[4]),
                "anomaly_used"     : float(state[0]),
            }

            results.append(result)
            all_logs.append(result)

        for r in results[:3]:
            correct_str = (
                "✅" if r.get("is_correct")
                else ("❌" if r.get("is_correct") is False else "?")
            )
            print("=" * 60)
            print(f"STEP: {r['step']}")
            print(f"ACTION: {r['action_name']}  {correct_str}")
            print(f"ANOMALY NORM: {r['anomaly_normalized']:.6f}")
            print(f"CONF: {r['policy_confidence']:.4f}")
            print(f"Q-VALUES: {np.round(r['q_values'], 4)}")

            if "q_std" in r and r["q_std"] is not None:
                print(f"Q-STD (MC Dropout): {np.round(r['q_std'], 6)}")

            if ENABLE_EXPLANATIONS and r.get("explanation"):
                print(f"EXPLANATION: {r['explanation']}")

            if ENABLE_CRITIQUE and r.get("critique"):
                print(f"CRITIQUE: {r['critique']}")

            print(f"REWARD: {r['reward']:.4f}")
            print(f"LATENCY: {r['latency']*1000:.3f} ms")

        summary = summarize_verify_results(results, true_label=label)
        episode_summaries.append(summary)

        print(f"\n{'='*60}")
        print(f"LABEL SUMMARY: {label}  (gas_id={true_gas_id})")
        print(f"  Accuracy          : {summary['accuracy']}")
        print(f"  Correct / Total   : {summary['correct_count']} / {summary['num_steps']}")
        print(f"  Expected actions  : {summary['expected_actions']}")
        print(f"  Dominant action   : {summary['dominant_action_name']}")
        print(f"  Mean reward       : {summary['mean_reward']:.4f}")
        print(f"  Mean confidence   : {summary['mean_policy_conf']:.4f}")
        print(f"  Avg latency (ms)  : {summary['avg_latency_ms']:.3f}")

    # ── Final overall summary ─────────────────────────────────────────
    all_correct = sum(r["is_correct"] for r in all_logs)
    all_total   = len(all_logs)
    overall_acc = all_correct / all_total if all_total else 0.0

    # Danger miss rate: Smoke (gas_id=1) and Mixture (gas_id=2) missed with action 0
    danger_rows   = [r for r in all_logs if r["true_gas_id"] in {1, 2}]
    danger_missed = sum(1 for r in danger_rows if r["action"] == 0)
    danger_miss_rate = danger_missed / len(danger_rows) if danger_rows else 0.0

    # False alarm rate: NoGas (gas_id=0) triggered with action >= 3
    nogas_rows   = [r for r in all_logs if r["true_gas_id"] == 0]
    false_alarms = sum(1 for r in nogas_rows if r["action"] >= 3)
    false_alarm_rate = false_alarms / len(nogas_rows) if nogas_rows else 0.0

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"  Overall accuracy   : {overall_acc:.4f}  ({all_correct}/{all_total})")
    print(f"  ⚠️  Danger miss rate  : {danger_miss_rate:.4f}  ({danger_missed}/{len(danger_rows)})")
    print(f"  🔔  False alarm rate  : {false_alarm_rate:.4f}  ({false_alarms}/{len(nogas_rows)})")
    print()

    # Correct mapping check
    print("  Label → gas_id mapping used:")
    for lbl, gid in GAS_MAP.items():
        correct_acts = get_expected_action(gid)
        act_names = " or ".join(ACTIONS[a] for a in correct_acts)
        print(f"    {lbl:10s} → gas_id={gid} → {act_names}")

    print()
    print(f"  {'Label':10s}  {'gas_id':>6}  {'Accuracy':>10}  {'Correct/Total':>15}  {'Dominant Action':>20}")
    print(f"  {'-'*70}")
    for s in episode_summaries:
        lbl = str(s["true_label"])
        gid = GAS_MAP.get(lbl, "?")
        acc = str(s["accuracy"])
        ct  = f"{s['correct_count']}/{s['num_steps']}"
        dom = s["dominant_action_name"]
        print(f"  {lbl:10s}  {gid:>6}  {acc:>10}  {ct:>15}  {dom:>20}")

    df_log     = pd.DataFrame(all_logs)
    df_summary = pd.DataFrame(episode_summaries)
    df_log.to_csv(SAVE_LOG_PATH,     index=False)
    df_summary.to_csv(SAVE_SUMMARY_PATH, index=False)
    print(f"\nDetailed log  → {SAVE_LOG_PATH}")
    print(f"Summary table → {SAVE_SUMMARY_PATH}")


# =========================================================
# MODE 2: LIVE STREAMING SIMULATION
# =========================================================
def run_live_mode():
    print("\nRunning LIVE streaming mode...")

    df = pd.read_csv(RAW_CSV_PATH)
    print(f"Raw dataset loaded: {df.shape}")

    missing = [c for c in RAW_SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing sensor columns: {missing}")

    label_col = infer_label_column(df)
    if label_col:
        print(f"Label column: {label_col}")
    else:
        print("No label column — running blind (no correctness check).")

    all_logs          = []
    episode_summaries = []
    labels_to_run     = (
        sorted(df[label_col].dropna().unique().tolist())
        if label_col else [None]
    )

    for label in labels_to_run:
        df_label = (
            df[df[label_col] == label].reset_index(drop=True)
            if label is not None else df
        )
        if len(df_label) < WINDOW_SIZE:
            print(f"Skipping {label}: only {len(df_label)} rows")
            continue

        # ── FIX: derive gas_id from label via GAS_MAP ─────────────────
        true_gas_id = get_true_gas_id(label) if label in GAS_MAP else None

        print("\n" + "#" * 80)
        print(f"LIVE LABEL: {label}  |  gas_id={true_gas_id}  |  Rows: {len(df_label)}")

        agent.reset_window()
        results      = []
        steps_to_run = min(len(df_label), MAX_STEPS_PER_EPISODE)

        for step in range(steps_to_run):
            row          = df_label.iloc[step]
            sensor_array = row_to_sensor_array(row)

            result = agent.run_once(
                sensor_row=sensor_array,
                step=step,
                gas_id=true_gas_id,   # true gas_id from GAS_MAP
                use_mc_dropout=USE_MC_DROPOUT,
                enable_explanations=ENABLE_EXPLANATIONS,
                enable_critique=ENABLE_CRITIQUE,
            )
            results.append(result)

        ready = [r for r in results if r.get("ready", False)]

        for r in ready[:3]:
            correct_str = (
                "✅" if r.get("is_correct")
                else ("❌" if r.get("is_correct") is False else "?")
            )
            print("=" * 60)
            print(f"STEP: {r['step']}")
            print(f"ACTION: {r['action_name']}  {correct_str}")
            print(f"ANOMALY NORM: {r['anomaly_normalized']:.6f}")
            print(f"CONF: {r['policy_confidence']:.4f}")
            print(f"Q-VALUES: {np.round(r['q_values'], 4)}")
            if "q_std" in r and r["q_std"] is not None:
                print(f"Q-STD (MC Dropout): {np.round(r['q_std'], 6)}")

            if ENABLE_EXPLANATIONS and r.get("explanation"):
                print(f"EXPLANATION: {r['explanation']}")

            if ENABLE_CRITIQUE and r.get("critique"):
                print(f"CRITIQUE: {r['critique']}")
            print(f"REWARD: {r['reward']:.4f}")
            print(f"LATENCY: {r['latency']*1000:.3f} ms")

        summary = summarize_live_results(results, true_label=label)
        episode_summaries.append(summary)
        print("\nLive Summary:", summary)

        for r in ready:
            all_logs.append({
                "true_label"       : label,
                "true_gas_id"      : true_gas_id,
                "step"             : r["step"],
                "action_raw"       : r["action_raw"],
                "action_raw_name"  : r["action_raw_name"],
                "action"           : r["action"],
                "action_name"      : r["action_name"],
                "is_correct"       : r.get("is_correct"),
                "anomaly_raw"      : r["anomaly_raw"],
                "anomaly_normalized": r["anomaly_normalized"],
                "policy_confidence": r["policy_confidence"],
                "reward"           : r["reward"],
                "latency_ms"       : r["latency"] * 1000,
                "q0": r["q_values"][0], "q1": r["q_values"][1],
                "q2": r["q_values"][2], "q3": r["q_values"][3],
                "q4": r["q_values"][4],
                "q_std": r.get("q_std"),
                "explanation": r.get("explanation"),
                "critique": r.get("critique"),
            })

    pd.DataFrame(all_logs).to_csv(SAVE_LOG_PATH,     index=False)
    pd.DataFrame(episode_summaries).to_csv(SAVE_SUMMARY_PATH, index=False)
    print(f"\nDetailed log  → {SAVE_LOG_PATH}")
    print(f"Summary table → {SAVE_SUMMARY_PATH}")


if __name__ == "__main__":
    if RUN_MODE == "verify":
        run_verify_mode()
    elif RUN_MODE == "live":
        run_live_mode()
    else:
        raise ValueError("RUN_MODE must be 'verify' or 'live'")