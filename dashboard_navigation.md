# GasSafe AI — Dashboard Navigation Guide

> **Version 2.0** | Run with `streamlit run app.py`  
> GasSafe AI Control Dashboard — Dueling Double DQN Autonomous Safety Agent

---

## Overview

The GasSafe AI dashboard is an industrial SCADA-style interface for evaluating and monitoring the Dueling Double DQN gas detection agent. It runs the trained model against the processed test dataset, displays per-step decision results, generates detailed explanations and safety critiques, and provides comprehensive analytics across all four gas classes.

The dashboard covers the complete evaluation pipeline — from loading the model to per-step decision explanations to downloadable CSV exports.

---

## Quick Start

```bash
# From the project root with virtual environment active:
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Dashboard Layout

```
GasSafe AI Control Dashboard
│
├── Sidebar (Configuration + Controls)
│
└── Main Panel
     ├── Header (system status + version)
     ├── Dataset Sample Counts (auto-shown when CSV is found)
     ├── Run Button
     └── Results (4 tabs, shown after Run)
          ├── Tab 1 — OVERVIEW
          ├── Tab 2 — PER-CLASS ANALYSIS
          ├── Tab 3 — STEP DETAIL + EXPLANATIONS
          └── Tab 4 — CHARTS & ANALYTICS
```

---

## Sidebar — Configuration and Controls

The sidebar contains all settings. Changes take effect immediately for filter and display options. Model/CSV path changes require re-clicking Run.

### Configuration Section

| Field | Default | Description |
|-------|---------|-------------|
| DQN Model Path | `models/DeepQmodel.pth` | Path to the trained Dueling DQN weights file |
| Test CSV Path | `test_df_processed.csv` | Path to the normalized test set exported from the training notebook |

> **Important:** `test_df_processed.csv` must be exported from the training notebook **after** the normalization cell runs. The file must contain the `label` column (for `GAS_MAP` lookup) and all 22 state feature columns.

### Run Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Max steps per class | 400 | Maximum number of test steps to evaluate per gas class. Set to 400 to evaluate the full test set (320–316 per class). |
| MC Dropout | Off | Enables Monte Carlo Dropout uncertainty estimation. Runs N forward passes per step, returns mean Q-values and standard deviation. Slower but provides confidence intervals. |
| Show Explanations | On | Generates a domain-expert explanation for every decision step. Rule-based, no LLM required. |
| Show Safety Critique | On | Generates an independent safety assessment for every decision step. Covers confidence thresholds, risk implications, and deployment notes. |

### Filter Section (Step Detail Tab only)

| Setting | Default | Description |
|---------|---------|-------------|
| Gas class | All | Filter the Step Detail tab to show only steps from a specific gas class (All / NoGas / Smoke / Mixture / Perfume) |
| Wrong predictions only | Off | When enabled, the Step Detail tab shows only incorrect decisions |
| Steps shown per class | 8 | How many decision cards to display per gas class in the Step Detail tab (1–50) |

### Reference Panel

The bottom of the sidebar shows the canonical gas mapping and split parameters:

```
GAS_MAP (canonical)
NoGas   → gas_id=0 → Monitor
Smoke   → gas_id=1 → Raise Alarm
Mixture → gas_id=2 → Emergency Shutdown
Perfume → gas_id=3 → Inc.Sampling / Req.Verif

SPLIT: block-wise holdout
GAP=20  TRIM=20  overlap=0
Train: 4,944  Test: 1,276

STATE: 22 features
anomaly(1) + current(7) + delta(7) + std(7)
```

---

## Running a Verification

### Step 1 — Verify file paths

If the model or CSV is not found, an error appears and the Run button shows "FILES NOT FOUND". Correct the paths in the sidebar.

When the CSV is found, the dashboard automatically shows a sample count row:

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Mixture   │    NoGas    │   Perfume   │    Smoke    │
│     315     │     316     │     320     │     321     │
│ test samples│ test samples│ test samples│ test samples│
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### Step 2 — Click Run

Click **"RUN FULL VERIFICATION — ALL 4 GAS CLASSES"**.

The progress bar shows which class and step is currently being evaluated:

```
Processing Mixture — step 47/315 (1/4 classes)  ████████░░░░  14%
```

All 4 gas classes are always processed in full — Mixture, NoGas, Perfume, and Smoke.

### Step 3 — View results

After processing completes, a success message confirms the total steps evaluated:

```
✓ Verification complete — 1,272 steps across 4 gas classes
```

Results appear across 4 tabs. Results are cached in session state. Clicking Run again forces a fresh evaluation.

---

## Tab 1 — OVERVIEW

**Purpose:** System-level performance at a glance.

### System Performance Metrics (5 KPI cards)

```
┌────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐
│ OVERALL        │ DANGER MISS    │ FALSE ALARM    │ AVG LATENCY    │ AVG CONFIDENCE │
│ ACCURACY       │ RATE           │ RATE           │                │                │
│                │                │                │                │                │
│ 92.55%         │ 0.00%          │ 0.00%          │ 0.46ms         │ 0.3287         │
│ 1181/1276      │ 0/636 missed   │ 0/652 NoGas    │ P99=0.73ms     │ mean all steps │
└────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘
```

**Color coding:**
- Green: metric meets target (accuracy ≥92%, miss rate = 0%, false alarm = 0%)
- Amber: approaching threshold
- Red: below threshold (requires attention)

**The two safety-critical metrics are Danger Miss Rate and False Alarm Rate.** Both must be 0.0% or as close as possible for a gas station deployment. The overall accuracy reflects the remaining Perfume/Smoke boundary classification difficulty.

### Per-Class Accuracy Gauges (4 circular gauges)

One gauge per gas class showing percentage accuracy:
- Green needle/fill: ≥95% accuracy
- Amber: ≥85%
- Red: <85%

### Policy Mapping Cards (4 cards)

Shows the trained policy mapping with live accuracy for this evaluation run:

```
Mixture                    NoGas
gas_id=2  DANGER           gas_id=0  SAFE
→ Emergency Shutdown       → Monitor
Maximum severity.          Normal surveillance.

100.0%                     90.0%
(315/315)                  (288/316)
```

### Evaluation Notes Panel

Explains the split method, training parameters, and key system properties including the Mixture anomaly note.

---

## Tab 2 — PER-CLASS ANALYSIS

**Purpose:** Deep dive into each gas class individually.

One collapsible section per gas class (Mixture expanded by default).

### Each Class Section Contains

**5 summary metrics:**

| Accuracy | Avg Anomaly | Avg Confidence | Mean Reward | Avg Latency |
|----------|-------------|----------------|-------------|-------------|

**Q-value bar chart** — shows Q-values from the first correct sample in that class, with the chosen action highlighted at full opacity and all others at 28% opacity. This shows how strongly the model prefers the correct action.

**Anomaly distribution histogram** — shows the spread of anomaly scores across all test steps for this class. Mixture will show a distribution clustered around 0.05–0.15 (expected low values). NoGas and Perfume will show higher values (0.4–0.7).

**Wrong predictions table** (only shown if wrong predictions exist) — lists the misclassified steps with their anomaly score, confidence, and Q-values. Useful for diagnosing class boundary failures.

---

## Tab 3 — STEP DETAIL + EXPLANATIONS

**Purpose:** Full per-step decision cards with expert explanations and safety critiques for all gas classes.

> **This tab always shows all 4 gas classes.** Use the sidebar filter to focus on one class or wrong predictions only.

### Summary Table

A compact table showing the first 400 filtered records:

| Label | Step | Action | OK | Expected | Anomaly | Conf | Reward |
|-------|------|--------|----|----------|---------|------|--------|

### Decision Cards

One card per step, organized by gas class with a colored header bar.

**Card structure:**

```
┌── [OK / WRONG] — Step 0 — Mixture ── [gas_id=2] ── [Emergency Shutdown] ── CORRECT ──────────────────────────────┐
│   anomaly=0.1025  conf=0.3174  reward=1.034  2.77ms                                                               │
│                                                                                                                    │
│   [A0:92.33] [A1:92.22] [A2:92.19] [A3:92.18] [A4:93.92]  ← A4 bold = chosen action                           │
│                                                                                                                    │
│   ▸ DECISION EXPLANATION                                                                                           │
│   CORRECT — Emergency Shutdown (action 4) confirmed for Mixture gas (gas_id=2). Mixture is the highest-severity   │
│   class — a volatile chemical combination creating immediate explosion risk at a gas station near any ignition     │
│   source. Anomaly: low (0.1025). This low anomaly is an LSTM AE artifact: the AE learned to reconstruct Mixture   │
│   with low MSE, producing a paradoxically low anomaly score for the most dangerous gas. This is why the 22-       │
│   feature state includes 7 delta features and 7 std features — these temporal signals carry the primary           │
│   discriminative information. Q(Emergency)=93.92. Gap to second-best: 1.59. Q-values: moderately spread           │
│   (range=2.74). Confidence: 31.7% (acceptable). Protocol: (1) Trigger audible+visual emergency alarm across       │
│   all zones. (2) Initiate controlled personnel evacuation. (3) Execute automatic fuel line isolation. (4) Notify   │
│   emergency services. (5) No re-entry until atmosphere verified below LEL.                                         │
│                                                                                                                    │
│   ▸ SAFETY CRITIQUE                                                                                                │
│   Confidence 0.3174 — acceptable for autonomous operation (above 0.25 threshold). Q-gap=1.59 shows clear action   │
│   preference. Safety: OPTIMAL. Emergency Shutdown for Mixture is the maximum-severity correct response. Model      │
│   demonstrates correct severity calibration — does not under-respond to the most dangerous gas class. Low          │
│   anomaly (0.1025) correctly handled via temporal sensor features, not anomaly threshold logic.                    │
│   Architecture note: Low anomaly (0.1025) for Mixture is an expected LSTM AE artifact. The 22-feature state       │
│   provides temporal sensor change signals that correctly override the misleading anomaly score. Validated: 0.0%   │
│   danger miss rate on 636 hazardous test samples using block-wise holdout with zero raw-row overlap.              │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Card color coding:**
- Left border green: correct decision
- Left border red: incorrect decision

**Q-chip row:** All 5 Q-values shown as chips. The chosen action's chip is bold; others are faded. Higher Q-value = stronger preference.

**Wrong decision cards** show expanded detail including the specific safety concern and retraining recommendation.

### Controlling What Cards Are Shown

The sidebar **"Steps shown per class"** slider (1–50, default 8) controls how many decision cards appear per gas class section. Increasing this shows more decisions — useful for spotting patterns in wrong predictions.

The **"Gas class"** filter limits the entire tab to one class. Combined with **"Wrong predictions only"**, this is the fastest way to investigate specific failures.

---

## Tab 4 — CHARTS & ANALYTICS

**Purpose:** Visual analytics across the full test set.

### Action Distribution Bar Chart

Shows how many times each action was selected across all 1,276 test steps. In a correctly trained model:
- Action 0 (Monitor): ~315 times — matching NoGas test samples
- Action 3 (Raise Alarm): ~294 times — matching Smoke test samples
- Action 4 (Emergency): ~358 times — matching Mixture test samples
- Action 1 + 2 combined: ~214 + 81 = ~295 — matching Perfume test samples

An uneven distribution (e.g. Action 4 appearing for all classes) indicates a policy problem.

### Confidence Distribution Histogram

Correct predictions shown in green, wrong in red. The distributions should be well-separated — high confidence on correct decisions, lower confidence on wrong ones. Significant overlap indicates the model has uncertainty in both correct and incorrect regions.

### Anomaly Score vs Chosen Action Scatter

All 1,276 test steps plotted as coloured points (colour = true gas class). Y-axis = chosen action. Shows that:
- Mixture points (red) cluster at Action 4 regardless of anomaly value (~0.10)
- NoGas points (green) cluster at Action 0 with anomaly ~0.5
- Smoke points (orange) cluster at Action 3 with anomaly ~0.48
- Perfume points (blue) spread between Action 1 and 2 based on anomaly level

### Mean Q-Values per Gas Class

Grouped bar chart showing the average Q-value for each action, grouped by true gas class. This reveals the trained policy structure — for Mixture, Q(Emergency) should be the tallest bar by a significant margin.

### Inference Latency Distribution

Histogram of per-step latency in milliseconds. Mean and P99 shown in the chart title. On CPU, the distribution should be centered around 0.4–0.8ms with a long right tail from occasional first-inference JIT compilation spikes.

---

## Download / Export

At the bottom of the main panel (below the tabs), two download buttons are available after any completed run:

| Button | File | Contents |
|--------|------|----------|
| Download Detailed Log CSV | `evaluation_log.csv` | One row per step: label, gas_id, step, action, action_name, is_correct, expected_actions, reward, policy_confidence, latency_ms, q0–q4, anomaly_used, explanation, critique |
| Download Summary CSV | `episode_summary.csv` | One row per gas class: label, gas_id, accuracy, correct, total, mean_reward, mean_conf, avg_latency_ms |

---

## Understanding the Metrics

### Decision Accuracy
The fraction of test steps where the agent's action is in `CORRECT_ACTIONS[gas_id]`. For example, choosing action 1 (Increase Sampling) for Perfume is correct. Choosing action 2 (Request Verification) for Perfume is also correct. Anything else is wrong.

### Danger Miss Rate
Of all Smoke (gas_id=1) and Mixture (gas_id=2) test steps combined, the fraction where the agent chose action 0 (Monitor) — a complete failure to respond to a dangerous gas. This is the most safety-critical metric. It must be 0.0%.

### False Alarm Rate
Of all NoGas (gas_id=0) test steps, the fraction where the agent chose action 3 or 4 — triggering an alarm or emergency shutdown on clean air. This must be very low to prevent alarm fatigue.

### Policy Confidence
The Q-value gap between the best and second-best action, normalised. Values above 0.25 indicate the agent has a clear preference and can act autonomously. Values below 0.15 should be flagged for human review in deployment.

### Policy Entropy
Measures how evenly the actions are distributed across the test set. Higher entropy = more diverse policy. For 5 actions, the maximum entropy is 1.609 (uniform distribution). The trained model achieves ~1.52, indicating a well-distributed, non-degenerate policy.

---

## Troubleshooting

### "FILES NOT FOUND" button state

Verify the model and CSV paths in the sidebar match the actual file locations. Paths are relative to the directory where you ran `streamlit run app.py`.

### Results only show Mixture

This was a bug in earlier versions where the first N records overall (all from Mixture, which appears first alphabetically) were shown instead of N records per class. The current version shows records per class. If this appears, ensure you are using the latest `app.py`.

### Progress bar reaches 100% before all classes finish

This can happen if the actual number of test samples per class differs from `max_steps`. The bar is normalized to `total_possible = sum(min(len(class), max_steps) for all classes)`. It is cosmetic only and does not affect results.

### Explanations show "disabled"

Enable "Show Explanations" in the sidebar. The explanation engine is rule-based and requires no external services.

### Import error on startup

Ensure you are running `streamlit run app.py` from the project root directory with the virtual environment activated. The error message shows the specific import that failed.

### Very slow first step (2000+ ms)

The first inference call initializes PyTorch JIT compilation. All subsequent steps run at normal speed (~0.5ms). This is expected behavior.

---

## Gas_id Mapping — Critical Reference

The system uses `GAS_MAP` to convert text labels to `gas_id`. This must match the training notebook exactly.

```python
GAS_MAP = {
    "NoGas":   0,   # correct action: Monitor (0)
    "Smoke":   1,   # correct action: Raise Alarm (3)
    "Mixture": 2,   # correct action: Emergency Shutdown (4)
    "Perfume": 3,   # correct action: Increase Sampling (1) or Request Verification (2)
}
```

The `gas_id` column in `test_df_processed.csv` is NOT used for correctness checking. That column contains YOLO's internal class indices which use a different ordering. The dashboard always derives `gas_id` from the `label` column via `GAS_MAP`.

---

## Actions Reference

| ID | Name | Gas class | When triggered |
|----|------|-----------|----------------|
| 0 | Monitor | NoGas | Clean air, all sensors baseline |
| 1 | Increase Sampling | Perfume (low anomaly) | Non-hazardous VOC, log + increase monitoring rate |
| 2 | Request Verification | Perfume (high anomaly) | Non-hazardous VOC, high concentration — ventilate + human confirm |
| 3 | Raise Alarm | Smoke | Combustion byproducts detected — evacuate + fire check |
| 4 | Emergency Shutdown | Mixture | Chemical mixture — maximum emergency response |

---

## What the Explanations Cover

### Decision Explanation (cyan border)

For **correct decisions**, covers:
- Confirmation of the correct gas class identification
- What the anomaly score means for this class (including the Mixture paradox for anomaly ≈ 0.10)
- Q-value analysis: dominant Q, gap to second-best, spread description
- Confidence assessment (very high / good / acceptable / moderate / LOW)
- The exact response protocol for this gas class

For **wrong decisions**, covers:
- The specific policy violation (severity: SAFETY-CRITICAL or non-critical)
- What the correct action should have been
- Q-gap analysis explaining why the model was uncertain
- Whether the sensor readings were near a class boundary
- Retraining recommendation

### Safety Critique (amber border)

Always covers:
1. **Confidence assessment** — Is the policy confidence above the 0.25 operational threshold? Below 0.15 requires mandatory human review.
2. **Safety assessment** — Is the decision policy-correct? If wrong, what are the real-world consequences?
3. **Architecture note** (for Mixture) — Explains why low anomaly is expected and how the model correctly handles it.
