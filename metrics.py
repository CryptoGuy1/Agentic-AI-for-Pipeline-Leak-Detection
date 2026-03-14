import json, numpy as np, pandas as pd
from pathlib import Path

SAFE_ACTIONS_AT_HIGH = {3, 4}   # alert or shutdown (edit to match your design)
ESCALATION_ACTIONS = {3, 4}

KEYWORDS = ["anomaly", "gas", "conf", "confidence", "action", "q"]

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def p95(x):
    return float(np.percentile(np.asarray(x, dtype=float), 95))

def grounded_rate(texts):
    ok = 0
    for t in texts.fillna("").astype(str):
        t2 = t.lower()
        if sum(k in t2 for k in KEYWORDS) >= 2:
            ok += 1
    return ok / max(1, len(texts))

def main(log_path):
    df = load_jsonl(log_path)

    out = {}
    out["cycles"] = len(df)
    out["latency_mean_s"] = float(df["latency_s"].mean())
    out["latency_p95_s"] = p95(df["latency_s"])

    # Tool success (if present)
    if "tool_ok" in df.columns:
        tool = pd.json_normalize(df["tool_ok"])
        for c in tool.columns:
            out[f"{c}_rate"] = float(tool[c].mean())

    out["safety_override_rate"] = float(df["safety_override_applied"].mean())

    # Safety violations: high/critical but not safe action
    high = df["severity"].isin(["HIGH", "CRITICAL"])
    out["safety_violations"] = int(((high) & (~df["action_final"].isin(SAFE_ACTIONS_AT_HIGH))).sum())

    # False escalation: low severity but escalated
    low = df["severity"].eq("LOW")
    out["false_escalation_rate"] = float(((low) & (df["action_final"].isin(ESCALATION_ACTIONS))).mean())

    # Time-to-escalate: first high/critical -> first escalation after it
    tte = []
    for _, g in df.groupby("run_id"):
        g = g.sort_values("cycle_id")
        idx = g.index[g["severity"].isin(["HIGH", "CRITICAL"])]
        if len(idx) == 0:
            continue
        start_i = idx[0]
        g2 = g.loc[start_i:]
        esc = g2.index[g2["action_final"].isin(ESCALATION_ACTIONS)]
        if len(esc) == 0:
            continue
        tte.append(int(g.loc[esc[0], "cycle_id"] - g.loc[start_i, "cycle_id"]))
    out["time_to_escalate_steps_mean"] = float(np.mean(tte)) if tte else None

    # Grounded explanation rate
    if "explanation_text" in df.columns:
        out["grounded_explanation_rate"] = float(grounded_rate(df["explanation_text"]))

    # Action distribution (stability proxy)
    counts = df["action_final"].value_counts(normalize=True).sort_index()
    out["action_dist"] = counts.to_dict()

    print(pd.Series(out).to_string())

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metrics.py runs/<run_id>.jsonl")
        raise SystemExit(1)
    main(sys.argv[1])