import pandas as pd
import numpy as np

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("evaluation_log.csv")

# ================================
# MAP ACTIONS → ALERT LEVELS
# ================================
action_map = {
    0: "None",
    1: "Low",
    2: "Low",
    3: "Medium",
    4: "High"
}

df["pred_label"] = df["pred_action"].map(action_map)

# ================================
# CONFUSION MATRIX
# ================================
conf_matrix = pd.crosstab(df["true_label"], df["pred_label"])

print("\nCONFUSION MATRIX:\n")
print(conf_matrix)

# ================================
# ALERT MATCH RATE
# ================================
# Define correct mapping (you can refine this later)
correct_map = {
    "NoGas": "None",
    "Smoke": "Medium",
    "Perfume": "Low",
    "Mixture": "High"
}

df["expected"] = df["true_label"].map(correct_map)

match = (df["pred_label"] == df["expected"]).sum()
total = len(df)

match_rate = (match / total) * 100

# ================================
# FALSE ALARM RATE
# ================================
false_alarms = df[
    (df["true_label"] == "NoGas") &
    (df["pred_label"] != "None")
]

total_nogas = len(df[df["true_label"] == "NoGas"])

false_alarm_rate = (
    len(false_alarms) / total_nogas * 100
    if total_nogas > 0 else 0
)

# ================================
# MISSED DETECTION RATE
# ================================
missed = df[
    (df["true_label"] != "NoGas") &
    (df["pred_label"] == "None")
]

total_hazard = len(df[df["true_label"] != "NoGas"])

missed_rate = (
    len(missed) / total_hazard * 100
    if total_hazard > 0 else 0
)

# ================================
# ACTION DISTRIBUTION
# ================================
dist = df["pred_label"].value_counts(normalize=True) * 100

# ================================
# PRINT SUMMARY
# ================================
print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

print(f"Alert Match Rate: {match_rate:.2f}%")
print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
print(f"Missed Detection Rate: {missed_rate:.2f}%")

print("\nAction Distribution (%):")
print(dist)

# ================================
# SAVE EVERYTHING
# ================================
conf_matrix.to_csv("confusion_matrix.csv")

summary = pd.DataFrame({
    "Metric": [
        "Alert Match Rate",
        "False Alarm Rate",
        "Missed Detection Rate"
    ],
    "Value": [
        match_rate,
        false_alarm_rate,
        missed_rate
    ]
})

summary.to_csv("metrics_summary.csv", index=False)

print("\n✅ Saved confusion_matrix.csv and metrics_summary.csv")