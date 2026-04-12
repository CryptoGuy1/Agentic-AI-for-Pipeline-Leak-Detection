# GasSafe AI — Dashboard Navigation Guide

> **Version 3.0** | Run with `streamlit run app.py`  
> **GasSafe AI Control Dashboard — Central Multimodal Agent · Folder-Driven Live Test · Sensor + Vision Verification**

---

## Overview

The GasSafe AI dashboard is an industrial SCADA-style interface for running the **current central multimodal agent pipeline** through a **folder-driven live test workflow**.

It is no longer a simple processed-test CSV DQN evaluator.

The dashboard now matches the current `main.py` and `agent_core.py` flow:

- loads the **Decision Agent** (`DecisionTool`)
- loads the **Anomaly Agent** (`AnomalyTool`)
- loads the **Vision Agent** (`VisionTool`)
- optionally loads the **Explanation Agent** (`ExplanationTool`)
- creates the central **`MultimodalAgent`**
- reads the **raw gas sensor CSV**
- selects **one usable image per class folder**
- builds the **20-step sliding window** from the raw sequential CSV
- computes anomaly
- runs the DQN policy
- applies safety override
- runs YOLO visual verification
- optionally generates explanation and critique
- displays structured outputs, charts, and downloadable results

This means the dashboard is now aligned with the **actual multimodal agent runtime path**, not the older processed-test verification path.

---

## Quick Start

```bash
streamlit run app.py
```

The dashboard typically opens at:

```text
http://localhost:8501
```

Run it from the **project root directory** with your virtual environment activated.

---

## What the Dashboard Is Evaluating

The dashboard runs the **current live multimodal decision pipeline**.

For each class:

- **NoGas**
- **Smoke**
- **Mixture**
- **Perfume**

the dashboard:

1. finds one image from that class folder
2. matches the image to the corresponding row in the raw CSV
3. constructs the prior **20-step sensor history**
4. computes anomaly from the sensor stream
5. builds the **22-dimensional RL state**
6. runs the DQN policy
7. applies safety override
8. runs the vision verification branch
9. generates explanation and critique if enabled
10. logs the final result

So one dashboard run is a **full multimodal live verification pass**, not just a static CSV replay.

---

## Dashboard Layout

```text
GasSafe AI Control Dashboard
│
├── Sidebar
│   ├── Configuration
│   ├── Run Settings
│   ├── Filter
│   └── Reference Mapping
│
└── Main Panel
    ├── Header
    ├── Raw Dataset Sample Counts
    ├── Run Button
    └── Results (3 tabs)
        ├── Tab 1 — Overview
        ├── Tab 2 — Case Details
        └── Tab 3 — Structured Table
```

---

## Sidebar — Configuration and Controls

The sidebar controls the paths, runtime options, and display filters.

### Configuration Section

| Field | Default | Description |
|-------|---------|-------------|
| DQN Model Path | `models/DeepQnet.pth` | Path to the DQN policy weights |
| AE Model Path | `models/lstm_autoencoder_weights.pth` | Path to the LSTM autoencoder weights |
| YOLO Model Path | `models/yolov8_gas_classifier.pt` | Path to the YOLO classifier weights |
| Raw CSV Path | raw gas measurement CSV path | Path to the original sensor CSV used for window reconstruction |
| Image Folder Path | thermal image folder path | Folder containing `NoGas`, `Smoke`, `Mixture`, and `Perfume` image subfolders |

### Important Note

The dashboard now expects the **raw dataset structure**, not only a processed test CSV.

That means it needs:

- a raw CSV with sensor values
- label column
- image-name column
- image folders grouped by class

The dashboard automatically searches for a matching image per class and only uses images that have enough prior CSV history to build the required window.

---

## Run Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Window Size | `20` | Number of time steps used for the rolling sensor window |
| MC Dropout (slower) | `True` | Enables multiple stochastic forward passes for uncertainty estimation |
| Show Explanations | `True` | Generates explanation output from the current explanation pipeline |
| Show Critique | `True` | Generates critique output from the current critique pipeline |

### Performance Warning

The dashboard may feel slow because the current defaults are computationally expensive:

- MC Dropout is enabled
- explanations are enabled
- critiques are enabled
- all 4 class folders are processed in one run
- each class replays a full 20-step window
- YOLO runs once per class

This is expected for **analysis mode**, but not ideal for fast interactive checks.

For a faster run, disable:

- **MC Dropout**
- **Show Explanations**
- **Show Critique**

That makes the dashboard much more responsive.

---

## Filter Section

| Setting | Default | Description |
|---------|---------|-------------|
| Filter by label | `All` | Limits the displayed results to a specific class |
| Show only wrong predictions | `False` | Shows only incorrect cases in the current results |

These filters affect what is shown after a completed run. They do **not** change what gets executed.

---

## Reference Mapping Panel

The bottom of the sidebar shows the canonical gas/action mapping:

```text
GAS_MAP
├ NoGas   → 0 → Monitor
├ Smoke   → 1 → Raise Alarm
├ Mixture → 2 → Emergency Shutdown
└ Perfume → 3 → Increase Sampling / Request Verification
```

This is the same semantic mapping used by the current agent pipeline.

---

## System Header

At the top of the dashboard, the header shows:

- system title
- system status
- architecture mode
- version text

Current header subtitle:

```text
CENTRAL MULTIMODAL AGENT · FOLDER-DRIVEN LIVE TEST · SENSOR + VISION VERIFICATION
```

That is the correct description of the current app behavior.

---

## Automatic Raw Dataset Overview

If the raw CSV path is valid, the dashboard automatically shows **raw row counts** per gas class before any run starts.

This gives a quick check that:

- the raw CSV loaded successfully
- the label column was found
- the class distribution looks reasonable

These counts are not final evaluation counts. They are simply raw dataset counts from the CSV.

---

## Running a Folder-Driven Live Test

### Step 1 — Verify File Paths

If any of these are missing:

- DQN model
- AE model
- YOLO model
- raw CSV
- image folder

the dashboard displays errors and disables the run button.

### Step 2 — Click Run

Click:

```text
RUN FOLDER LIVE TEST
```

The dashboard then:

- loads the central multimodal agent
- loads the raw CSV
- selects one usable image per class folder
- runs one case for each class
- stores results in session state

### Step 3 — Wait for Processing

The progress bar updates as each class is completed:

```text
Processed NoGas
Processed Smoke
Processed Mixture
Processed Perfume
```

### Step 4 — Review Results

After completion, results are displayed across 3 tabs.

---

## What Happens During Each Case

For each chosen class image, the dashboard runs the following pipeline:

### 1. Match Image to Raw CSV Row

The dashboard finds the image stem in the image-name column and locates the matching row in the raw CSV.

### 2. Rebuild the Sensor Window

It constructs the **20-step historical window** ending at that target row.

### 3. Reset the Central Agent Buffer

The central `MultimodalAgent` resets its internal sensor buffer so the case starts clean.

### 4. Replay the Sensor Sequence

Each of the 20 sensor rows is passed into `agent.run_once(...)`.

Only on the **last step** does the dashboard attach the image path, so vision is executed once at the final decision point.

### 5. Capture the Final Structured Result

The dashboard stores the final result fields such as:

- raw action
- action after safety
- final action
- anomaly values
- reward
- confidence
- YOLO outputs
- explanation
- critique
- latency

This is the exact current runtime path the app follows.

---

## The Three Action Stages Shown in the Dashboard

The dashboard shows **three action levels**:

### Raw Action

The first action proposed by the DQN policy before any safety or vision checks.

### Action After Safety

The action after the deterministic safety override logic is applied.

### Final Action

The final action after vision verification / escalation has been considered.

This distinction is very important in the current app because the dashboard is no longer showing only one action output. It shows how the agent arrived at the final decision.

---

## Vision Fields Shown in the Dashboard

The app now displays several vision-specific fields:

| Field | Description |
|-------|-------------|
| `yolo_class_id` | Raw YOLO internal class index |
| `yolo_class_label` | YOLO class label string |
| `yolo_confidence` | YOLO prediction confidence score |
| `yolo_semantic_gas_id` | Canonical gas ID derived from YOLO label |
| `yolo_gas_name` | Canonical gas name after semantic mapping |
| `vision_action_support` | Whether vision supports the sensor-based decision |
| `vision_danger_flag` | Whether vision detected a dangerous condition |
| `vision_reason` | Human-readable explanation from the vision branch |
| `vision_error` | Error message if vision inference failed |

These fields come from the current visual verification branch and are central to the current dashboard behavior.

---

## Tab 1 — Overview

**Purpose:** quick run summary and aggregated charts.

### Top KPI Cards

The dashboard shows 4 summary metrics:

| Metric | Description |
|--------|-------------|
| **Cases** | Number of currently displayed results after filtering |
| **Correct** | How many cases have final actions matching expected canonical actions |
| **Avg Policy Confidence** | Mean policy confidence across displayed cases |
| **Avg Latency** | Average case latency in milliseconds |

### Charts Shown

#### Final Action Distribution

A bar chart showing how often each final action was selected.

#### Policy Confidence Distribution

A histogram split into:

- correct predictions
- wrong predictions

#### Anomaly vs Final Action

A scatter plot showing:

- x-axis = normalized anomaly
- y-axis = final action
- color = gas label

### Selected Images Section

This section gives a compact card for each processed class showing:

- label
- image name
- raw action
- action after safety
- final action
- correctness
- whether vision escalated the action
- whether safety changed the action

This is the quickest place to see the full decision chain per class.

---

## Tab 2 — Case Details

**Purpose:** deep inspection of each class case.

Each case gets its own expandable detail panel.

### Each Case Panel Shows

#### Structured Summary

A structured block containing:

- label
- image
- CSV target index
- raw action
- action after safety
- final action
- correctness
- expected actions
- normalized anomaly
- raw anomaly
- policy confidence
- reward
- latency
- YOLO raw and mapped fields
- vision support
- danger flag
- safety changed action
- vision escalated action

#### Vision Reason Block

If vision ran successfully, the dashboard shows the full `vision_reason`.

If vision failed, the dashboard shows the `vision_error`.

#### Q-Value Bar Chart

Each case shows a Q-value chart for the five actions, with the chosen final action highlighted.

#### Explanation Block

If explanations are enabled, the app displays the current explanation output from the agent.

#### Critique Block

If critique is enabled, the app displays the current critique output from the agent.

This is the most detailed tab in the current dashboard.

---

## Tab 3 — Structured Table

**Purpose:** compact analysis and export-ready results.

The table contains the following columns:

| Column | Description |
|--------|-------------|
| `label` | Gas class label |
| `image_name` | Matched image filename |
| `target_idx` | CSV row index of the matched sample |
| `action_raw_name` | Action name from DQN policy before overrides |
| `action_after_safety_name` | Action name after safety guardrail applied |
| `action_name` | Final action name after vision escalation |
| `is_correct` | Whether final action matches canonical expected action |
| `policy_confidence` | DQN policy confidence score |
| `anomaly_normalized` | Normalized anomaly score from LSTM autoencoder |
| `reward` | Reward signal from the reward function |
| `yolo_class_label` | Raw YOLO class label |
| `yolo_confidence` | YOLO prediction confidence |
| `yolo_gas_name` | Canonical gas name after YOLO semantic mapping |
| `vision_action_support` | Whether vision supported the sensor decision |
| `vision_danger_flag` | Whether vision raised a danger flag |
| `safety_changed_action` | Whether safety override changed the DQN action |
| `vision_escalated_action` | Whether vision escalated the final action |
| `latency_ms` | Case processing latency in milliseconds |

This table can be downloaded as:

```text
dashboard_folder_live_test_results.csv
```

---

## What the Dashboard Is No Longer Doing

The current app is **not** the old DQN-only processed test dashboard.

It no longer primarily does:

- direct evaluation from `test_df_processed.csv`
- dashboard-side rule-based explanation generation
- dashboard-side critique generation
- pure `DecisionTool`-only evaluation

Instead, it now uses the **same central runtime path** as the multimodal agent system.

---

## Current Runtime Bottlenecks

The dashboard can feel slow because the app currently does all of the following in one run:

- loads the agent
- processes all 4 class folders
- replays a 20-step window per class
- runs DQN on each step
- optionally runs MC Dropout
- runs YOLO once per class
- generates explanation once per class
- generates critique once per class
- renders charts and tables

### Why This Is Expensive

One button click is roughly equivalent to:

- **4 classes**
- × **20 agent steps**
- × **4 YOLO inferences**
- × **4 explanation generations**
- × **4 critique generations**
- × optional **MC Dropout passes**

So this is a **heavy analysis dashboard**, not a lightweight real-time widget.

---

## Fast vs Analysis Mode

### Fast Mode

Use this for quick operational checks:

| Setting | Value |
|---------|-------|
| MC Dropout | **Off** |
| Show Explanations | **Off** |
| Show Critique | **Off** |

### Analysis Mode

Use this when you want detailed interpretation:

| Setting | Value |
|---------|-------|
| MC Dropout | **On** |
| Show Explanations | **On** |
| Show Critique | **On** |

The current app defaults are closer to **analysis mode**, which is why it feels slow.

---

## Current Agent Components Used by the Dashboard

The dashboard directly uses these current system components:

| Component | Role |
|-----------|------|
| `AnomalyTool` | LSTM autoencoder-based anomaly estimation |
| `DecisionTool` | Dueling DQN policy action selection |
| `ExplanationTool` | Operator-facing explanation generation |
| `VisionTool` | YOLOv8-based visual verification and escalation |
| `MultimodalAgent` | Central orchestrator coordinating all sub-agents |
| `ShortTermMemory` | Maintains recent sensor history and context |
| `GoalManager` | Manages agent goals and operational context |

That is what makes the app consistent with the current core system.

---

## Canonical Safety Mapping

The dashboard uses the current canonical gas mapping:

```python
GAS_MAP = {
    "NoGas": 0,
    "Smoke": 1,
    "Mixture": 2,
    "Perfume": 3,
}
```

And the current correct-action mapping:

```python
CORRECT_ACTIONS = {
    0: [0],       # NoGas   → Monitor
    1: [3],       # Smoke   → Raise Alarm
    2: [4],       # Mixture → Emergency Shutdown
    3: [1, 2],    # Perfume → Increase Sampling or Request Verification
}
```

So:

- **NoGas** should end in → **Monitor**
- **Smoke** should end in → **Raise Alarm**
- **Mixture** should end in → **Emergency Shutdown**
- **Perfume** should end in → **Increase Sampling** or **Request Verification**

---

## Interpreting the Main Metrics

### Cases

The number of currently displayed results after filtering.

### Correct

How many of those cases have final actions that match the current canonical expected actions.

### Avg Policy Confidence

The mean policy confidence across displayed cases.

### Avg Latency

Average case latency in milliseconds for the displayed results.

Because the dashboard works on one selected image per folder, this is currently a **small-case diagnostic dashboard**, not a bulk benchmark dashboard.

---

## Troubleshooting

### The App Is Very Slow

This is usually because MC Dropout, explanation, and critique are all enabled simultaneously while processing all 4 classes.

Turn these off for a faster run:

- MC Dropout → **Off**
- Show Explanations → **Off**
- Show Critique → **Off**

### Run Button Is Disabled

One or more required paths are invalid. Check that all of the following exist and are accessible:

- DQN model path
- AE model path
- YOLO model path
- raw CSV path
- image folder path

### Vision Fields Are Empty

Possible causes:

- image path missing
- YOLO model path invalid
- image match not found in CSV
- YOLO inference error

Check the `vision_error` field in the Case Details tab for the specific failure reason.

### Wrong Image / Class Matching

The dashboard expects:

- a usable label column in the CSV
- a usable image-name column in the CSV
- image filenames that match the CSV image names exactly

### Explanations or Critique Are Blank

This usually means:

- the explanation tool is disabled in run settings
- generation failed silently
- the current pipeline returned a disabled or empty string

### Results Do Not Change After Rerun

The app stores results in `st.session_state["dashboard_results"]`. Clicking Run again should refresh them, but if you changed paths or settings without rerunning, you may still be seeing old session results. Force a fresh run by clicking the Run button again after any settings change.

---

## Recommended Usage

Use the dashboard for:

- multimodal case inspection
- final-action analysis
- raw vs safety vs final action comparison
- YOLO verification debugging
- explanation and critique review
- export of structured case results

Do **not** treat it as the fastest possible runtime path when MC Dropout, explanation, and critique are all enabled. This is a **diagnostic analysis dashboard**, not a minimal-latency deployment frontend.

---

## Final Practical Summary

The current dashboard is best described as:

> **a folder-driven live analysis dashboard for the central multimodal agent**

It reflects the current architecture because it:

- rebuilds the sensor window from raw data
- computes anomaly dynamically
- runs the DQN decision path
- applies safety override
- runs the vision verification branch
- shows explanation and critique outputs from the current agent pipeline
- logs the structured final result

That is the correct description of what the current `app.py` is doing.
