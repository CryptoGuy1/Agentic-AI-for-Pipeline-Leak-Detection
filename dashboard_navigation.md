# GasSafe AI — Dashboard Navigation Guide

> **Version 4.0** | Run with `streamlit run app.py`
> **GasSafe AI Control Dashboard — Central Multimodal Agent · Folder-Driven Live Test · Sensor + Vision Verification**

---

## Overview

The GasSafe AI dashboard is an industrial SCADA-style interface for running the **current central multimodal agent pipeline** through a **folder-driven live test workflow**.

It is **not** the older processed-test CSV evaluator anymore.

The current dashboard now mirrors the live multimodal runtime path used by the core system:

- loads the **Decision Agent** (`DecisionTool`)
- loads the **Anomaly Agent** (`AnomalyTool`)
- loads the **Vision Agent** (`VisionTool`)
- conditionally loads the **Explanation Agent** (`ExplanationTool`)
- creates the central **`MultimodalAgent`**
- reads the **raw gas sensor CSV**
- selects **one usable image per class folder**
- builds the **20-step sliding sensor window**
- computes anomaly dynamically
- runs the DQN policy
- applies safety override
- runs YOLO visual verification
- optionally generates explanation and critique
- displays structured outputs, charts, and exportable results

That means the dashboard is now aligned with the **actual multimodal agent runtime path**, not the old processed-test verification flow.

---

## Quick Start

Run the dashboard from the **project root directory** with your virtual environment activated:

```bash
streamlit run app.py
```

The dashboard typically opens at:

```text
http://localhost:8501
```

---

## What the Dashboard Is Evaluating

The dashboard runs the **current live multimodal decision pipeline**.

For each of the four gas classes:

- **NoGas**
- **Smoke**
- **Mixture**
- **Perfume**

the dashboard performs one full live case:

1. selects one usable image from that class folder
2. matches the image to the correct row in the raw CSV
3. reconstructs the prior **20-step sensor history**
4. computes anomaly from the sensor stream
5. builds the **22-dimensional RL state**
6. runs the DQN policy
7. applies safety override
8. runs the vision verification branch
9. optionally generates explanation and critique
10. stores the final structured result

So one dashboard run is a **full multimodal live verification pass**, not a static replay from a preprocessed test file.

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
    ├── Master Header
    ├── Raw Dataset Sample Counts
    ├── Run Button
    └── Results (3 tabs)
        ├── Tab 1 — Overview
        ├── Tab 2 — Case Details
        └── Tab 3 — Structured Table
```

---

## Sidebar — Configuration and Controls

The sidebar controls:

- model paths
- dataset paths
- runtime options
- display filters

### Configuration Section

| Field             | Default                               | Description                                                                    |
| ----------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
| DQN Model Path    | `models/DeepQnet.pth`                 | Path to the trained DQN policy weights                                         |
| AE Model Path     | `models/lstm_autoencoder_weights.pth` | Path to the LSTM autoencoder weights                                           |
| YOLO Model Path   | `models/yolov8_gas_classifier.pt`     | Path to the YOLO classifier weights                                            |
| Raw CSV Path      | dataset-specific raw CSV path         | Path to the original sensor CSV used to rebuild the sliding window             |
| Image Folder Path | dataset-specific image folder path    | Folder containing `NoGas`, `Smoke`, `Mixture`, and `Perfume` image subfolders |

### Important Note

The dashboard expects the **raw dataset structure**, not only a processed test CSV.

That means it needs:

- a raw CSV with sensor values
- a usable label column
- a usable image-name column
- class image folders

The dashboard automatically looks for one image per class that has enough prior CSV history to build the required sensor window.

---

## Run Settings

| Setting             | Default | Description                                                             |
| ------------------- | ------- | ----------------------------------------------------------------------- |
| Window Size         | `20`    | Number of time steps in the rolling sensor window                       |
| MC Dropout (slower) | `True`  | Enables stochastic multi-pass inference for uncertainty estimation      |
| Show Explanations   | `True`  | Enables explanation generation through the current explanation pipeline |
| Show Critique       | `True`  | Enables critique generation through the current critique pipeline       |

### Important Behavior

The dashboard currently defaults to a **heavy analysis mode**, not a minimal-speed mode.

That means the default run is expensive because it can do all of the following:

- process all 4 class folders
- replay a full 20-step window per class
- run the DQN repeatedly
- optionally run MC Dropout
- run YOLO once per class
- generate one explanation per class
- generate one critique per class

That is why the dashboard can feel slow.

### Fast Run Recommendation

For quicker runs, disable:

- **MC Dropout**
- **Show Explanations**
- **Show Critique**

That gives you a much more responsive dashboard.

---

## Filter Section

| Setting                     | Default | Description                                          |
| --------------------------- | ------- | ---------------------------------------------------- |
| Filter by label             | `All`   | Filters displayed results by gas class               |
| Show only wrong predictions | `False` | Shows only incorrect cases in the current result set |

These filters affect **display only**. They do **not** reduce the amount of work done during execution.

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

This matches the current semantic mapping used by the agent pipeline.

---

## Master Header

At the top of the dashboard, the master header shows:

- dashboard title
- system status
- architecture mode
- version identity

Current subtitle:

```text
CENTRAL MULTIMODAL AGENT · FOLDER-DRIVEN LIVE TEST · SENSOR + VISION VERIFICATION
```

That is the correct current description of the app.

---

## Automatic Raw Dataset Overview

If the raw CSV path is valid, the dashboard automatically shows class counts from the raw CSV before any run starts.

This confirms:

- the CSV loaded successfully
- the label column was found
- the class distribution is visible

These are **raw CSV row counts**, not final evaluation totals.

---

## Running a Folder-Driven Live Test

### Step 1 — Verify File Paths

If any of these are missing:

- DQN model
- AE model
- YOLO model
- raw CSV
- image folder

the dashboard displays an error and disables the run button.

### Step 2 — Click Run

Click the run button:

```text
RUN FOLDER LIVE TEST
```

The dashboard then:

- loads the central multimodal agent
- loads the raw CSV
- selects one valid image per class
- runs one full case for each class
- stores results in session state

### Step 3 — Wait for Processing

The progress bar updates as each class finishes, for example:

```text
Processed NoGas
Processed Smoke
Processed Mixture
Processed Perfume
```

### Step 4 — Review Results

After completion, the dashboard displays results across 3 tabs.

---

## What Happens During Each Case

For each selected class image, the app runs the following actual pipeline.

### 1. Match Image to Raw CSV Row

The dashboard finds the image filename stem in the image-name column and locates the matching row in the raw CSV.

### 2. Rebuild the Sensor Window

The dashboard constructs the **20-step historical sensor window** ending at the matched target row.

### 3. Reset the Central Agent Buffer

The central `MultimodalAgent` resets its internal sensor buffer before each case.

### 4. Replay the Sensor Sequence

Each of the 20 sensor rows is passed to:

```python
agent.run_once(...)
```

Only on the **last step** is `image_path` attached, so vision runs once at the final decision point.

### 5. Capture the Final Result

The dashboard stores the final structured output, including:

- raw action
- action after safety
- final action
- anomaly values
- reward
- policy confidence
- YOLO fields
- explanation
- critique
- latency

This is the exact current runtime path used by the app.

---

## The Three Action Stages Shown in the Dashboard

The dashboard shows **three action stages** for every case.

### Raw Action

The first action proposed by the DQN before safety or vision changes.

### Action After Safety

The action after deterministic safety override logic runs.

### Final Action

The action after the vision verification / escalation branch has been applied.

This is a major part of the current dashboard because the UI is designed to show how the final safety decision was formed, not just the initial policy action.

---

## Vision Fields Shown in the Dashboard

The dashboard displays several vision-specific fields:

| Field                   | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `yolo_class_id`         | Raw YOLO internal class index                         |
| `yolo_class_label`      | Raw YOLO class label                                  |
| `yolo_confidence`       | YOLO prediction confidence                            |
| `yolo_semantic_gas_id`  | Canonical gas ID after semantic mapping               |
| `yolo_gas_name`         | Canonical gas name after semantic mapping             |
| `vision_action_support` | Whether the vision result supports the raw DQN action |
| `vision_danger_flag`    | Whether vision detected a dangerous case              |
| `vision_reason`         | Human-readable explanation from the vision branch     |
| `vision_error`          | Error message if vision inference fails               |

These are core parts of the current dashboard behavior.

---

## Explainability Layer in the Dashboard

The current dashboard uses the project's explainability layer through:

- `ExplanationTool`
- the central `MultimodalAgent`
- the current explanation / critique pipeline

### Current Model

The current explanation tool is configured to use:

```text
gemma3:1b
```

when explanations or critiques are enabled. This is the current local LLM-backed explainability setup in the app.

### What Explanation Does

The explanation block helps interpret:

- the sensor-driven decision
- the anomaly signal
- the policy choice
- the safety-adjusted action
- the final action
- the role of the vision branch

### What Critique Does

The critique block audits:

- confidence quality
- Q-gap strength
- robustness
- anomaly/action consistency
- vision agreement
- operational trustworthiness

### Important UI Behavior

The dashboard now intentionally **always renders** explanation and critique blocks, even when they are:

- disabled
- empty
- fallback-generated
- tool-failed

So instead of silently disappearing, the UI now shows explicit state messages. That is part of the current app behavior.

---

## Vision, Explanation, and Critique Rendering Behavior

The current app has dedicated render functions for:

- explanation
- critique
- vision

That means the UI deliberately shows:

- normal outputs
- fallback outputs
- disabled outputs
- error outputs

### Explanation Block Behavior

The explanation block can show:

- normal explanation text
- explanation disabled message
- explanation returned no output
- explanation returned an empty string

### Critique Block Behavior

The critique block can show:

- normal critique text
- critique disabled message
- critique returned no output
- critique returned an empty string

### Vision Block Behavior

The vision block can show:

- a normal vision report
- a vision error report

So the dashboard is designed to **make system state visible**, not hide missing outputs.

---

## Tab 1 — Overview

**Purpose:** quick run summary and aggregate charts.

### Top KPI Cards

The dashboard shows 4 top metrics:

| Metric                    | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| **Cases**                 | Number of displayed results after filtering                                |
| **Correct**               | Number of results whose final action matches the canonical expected action |
| **Avg Policy Confidence** | Mean policy confidence across displayed cases                              |
| **Avg Latency**           | Mean latency in milliseconds across displayed cases                        |

### Charts Shown

#### Final Action Distribution

Bar chart showing how often each final action was selected.

#### Policy Confidence Distribution

Histogram split into:

- correct cases
- wrong cases

#### Anomaly vs Final Action

Scatter plot showing:

- x-axis = normalized anomaly
- y-axis = final action
- color = gas label

### Selected Images Section

This section gives a quick compact summary for each processed class:

- label
- image name
- raw action
- action after safety
- final action
- correctness
- whether vision escalated the action
- whether safety changed the action

This is the fastest place to inspect the action chain per class.

---

## Tab 2 — Case Details

**Purpose:** full per-case inspection.

Each case appears inside its own expandable detail panel.

### Each Case Panel Shows

#### Structured Summary

A compact detail block containing:

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

#### Vision Block

If vision ran successfully, the full `vision_reason` is shown.

If vision failed, the `vision_error` is shown instead.

#### Q-Value Bar Chart

Each case shows a Q-value chart for the 5 actions, with the final chosen action highlighted.

#### Explanation Block

If explanation is enabled, the current explanation output is shown.

If explanation is disabled or empty, the dashboard shows that state explicitly.

#### Critique Block

If critique is enabled, the current critique output is shown.

If critique is disabled or empty, the dashboard shows that state explicitly.

This is the most detailed tab in the current app.

---

## Tab 3 — Structured Table

**Purpose:** compact tabular analysis and export.

The current table includes these columns:

| Column                     | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `label`                    | Gas class label                                |
| `image_name`               | Matched image filename                         |
| `target_idx`               | CSV row index used for the case                |
| `action_raw_name`          | DQN action before overrides                    |
| `action_after_safety_name` | Action after safety override                   |
| `action_name`              | Final action after vision                      |
| `is_correct`               | Whether final action matches expected action   |
| `policy_confidence`        | Policy confidence score                        |
| `anomaly_normalized`       | Normalized anomaly score                       |
| `reward`                   | Reward signal                                  |
| `yolo_class_label`         | Raw YOLO label                                 |
| `yolo_confidence`          | YOLO confidence                                |
| `yolo_gas_name`            | Canonical gas name after semantic mapping      |
| `vision_action_support`    | Whether vision supported the raw policy action |
| `vision_danger_flag`       | Whether vision raised a danger flag            |
| `safety_changed_action`    | Whether safety changed the raw action          |
| `vision_escalated_action`  | Whether vision escalated the final action      |
| `latency_ms`               | Case latency in milliseconds                   |

The table can be exported as:

```text
dashboard_folder_live_test_results.csv
```

---

## Session State and Rerun Behavior

The dashboard stores results in:

```python
st.session_state["dashboard_results"]
```

That means completed results can persist in the app until a fresh run is triggered.

### Practical Implication

If you change:

- model paths
- CSV path
- image folder
- window size
- MC Dropout
- explanation toggle
- critique toggle

you should click **Run** again to refresh the results.

So the dashboard is not purely stateless; it preserves results until rerun. That is part of the current app behavior.

---

## What the Dashboard Is No Longer Doing

The current dashboard is **not** the old processed-test DQN-only viewer.

It no longer primarily does:

- direct evaluation from `test_df_processed.csv`
- dashboard-side rule-based explanation generation
- dashboard-side rule-based critique generation
- pure `DecisionTool`-only evaluation

Instead, it uses the **same central runtime path** as the multimodal agent system.

---

## Current Runtime Bottlenecks

The dashboard can feel slow because one run currently does all of the following:

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

One full dashboard run is roughly equivalent to:

- **4 classes**
- × **20 agent steps**
- \+ **4 YOLO inferences**
- \+ **4 explanation generations**
- \+ **4 critique generations**
- \+ optional **multiple MC Dropout forward passes**

So this is a **diagnostic analysis dashboard**, not a minimal-latency real-time frontend.

---

## Fast Mode vs Analysis Mode

### Fast Mode

Use this for quick operational checks:

| Setting           | Value   |
| ----------------- | ------- |
| MC Dropout        | **Off** |
| Show Explanations | **Off** |
| Show Critique     | **Off** |

### Analysis Mode

Use this when you want full interpretability:

| Setting           | Value  |
| ----------------- | ------ |
| MC Dropout        | **On** |
| Show Explanations | **On** |
| Show Critique     | **On** |

The current defaults are closer to **analysis mode**, which is why the dashboard feels heavier.

---

## Current Agent Components Used by the Dashboard

The dashboard directly uses these components:

| Component         | Role                                                           |
| ----------------- | -------------------------------------------------------------- |
| `AnomalyTool`     | LSTM autoencoder-based anomaly estimation                      |
| `DecisionTool`    | Dueling DQN policy action selection                            |
| `VisionTool`      | YOLO-based visual verification and escalation                  |
| `ExplanationTool` | Local explainability interface for explanation and critique    |
| `MultimodalAgent` | Central orchestrator coordinating the full multimodal workflow |
| `ShortTermMemory` | Stores recent structured outputs and agent memory entries      |
| `GoalManager`     | Provides goal/context management for the central agent         |

### Important Clarification

The **rolling 20-step sensor window** is maintained by the central agent buffer logic, not by `ShortTermMemory`.

`ShortTermMemory` is attached to the agent for structured memory logging, but it is **not** the sliding window mechanism itself.

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
    0: [0],
    1: [3],
    2: [4],
    3: [1, 2],
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

The number of currently displayed cases after filtering.

### Correct

How many displayed cases have final actions matching the canonical expected action.

### Avg Policy Confidence

The mean policy confidence across displayed cases.

### Avg Latency

Average latency in milliseconds for the displayed cases.

Because the dashboard currently works on one selected image per folder, it is a **small-case multimodal diagnostic dashboard**, not a bulk benchmark dashboard.

---

## Troubleshooting

### The App Is Very Slow

Usually caused by:

- MC Dropout enabled
- explanations enabled
- critiques enabled
- all 4 classes processed in one run

For a faster run, disable:

- MC Dropout
- Show Explanations
- Show Critique

### Run Button Is Disabled

One or more required paths are invalid:

- DQN model
- AE model
- YOLO model
- raw CSV
- image folder

### Vision Fields Are Empty

Possible reasons:

- image path missing
- YOLO model path invalid
- image match not found
- YOLO inference error

Check the `vision_error` field in Case Details.

### Wrong Image/Class Matching

The app expects:

- a valid label column
- a valid image-name column
- filenames that match correctly between CSV and image folders

### Explanations or Critique Look Empty

The dashboard now shows explicit explanation/critique state messages instead of silently hiding them.

Possible reasons include:

- disabled in settings
- tool returned no output
- empty output
- fallback output generated after contradiction checking

### Results Look Stale

The dashboard stores results in session state. If you changed configuration but did not rerun, you may still be seeing older results.

Click **Run** again after changing important settings.

---

## Recommended Usage

Use the dashboard for:

- multimodal case inspection
- raw vs safety vs final action comparison
- YOLO verification debugging
- explanation and critique review
- compact result export
- final-case analysis per gas class

Do **not** treat it as the fastest possible runtime path when MC Dropout, explanation, and critique are all enabled.

This is a **diagnostic analysis dashboard**, not a minimal-latency deployment interface.

---

## Final Practical Summary

The current dashboard is best described as:

> **a folder-driven live analysis dashboard for the central multimodal agent**

It reflects the actual current architecture because it:

- rebuilds the sensor window from raw data
- computes anomaly dynamically
- runs the DQN decision path
- applies safety override
- runs the vision verification branch
- uses the current explanation / critique pipeline
- renders explicit explanation / critique / vision states
- logs the structured final result

That is the correct description of what the current `app.py` is doing.
