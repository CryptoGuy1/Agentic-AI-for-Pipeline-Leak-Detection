# GasSafe AI — Multimodal Agentic Gas Safety System

> **Agentic multimodal safety system for gas monitoring and action selection.**  
> Dueling Deep Q-Network · LSTM Autoencoder Anomaly Detection · YOLOv8 Verification · Safety Guardrails · Explainability Layer · Streamlit Dashboard

---

## Overview

GasSafe AI is a **multimodal, agentic industrial gas safety system** that combines:

- **time-series gas sensor analysis**
- **anomaly detection**
- **reinforcement learning**
- **thermal-image verification**
- **operator-facing explanation and critique**

This project is designed for operational decision support in gas monitoring scenarios where the system must do more than classify gas.

It must:

1. read recent sensor behavior
2. estimate abnormality
3. build a structured temporal state
4. choose a safety action using reinforcement learning
5. apply hard safety rules
6. verify the decision with visual evidence
7. generate explanation and critique outputs
8. log the result for operator review

The current system detects four gas conditions and selects one of five calibrated safety actions.

---

## What This Project Is

This project is best described as:

> **a central agentic multimodal safety system with specialized functional sub-agents**

It is **agentic** because it:

- perceives the environment from multiple inputs
- builds an internal state representation
- chooses actions autonomously
- applies safety guardrails
- verifies actions with visual evidence
- generates explanations and critiques
- logs decisions for traceability


The system is organized around one main orchestrator (`MultimodalAgent`) and several specialized sub-agents:

- **Anomaly Agent** → implemented as `AnomalyTool` for LSTM autoencoder-based anomaly estimation
- **Decision Agent** → implemented as `DecisionTool` for Dueling Deep Q-Network action selection
- **Vision Agent** → implemented as `VisionTool` for YOLO-based visual gas classification and verification
- **Explanation Agent** → implemented as `ExplanationTool` for operator-facing explanation generation
- **Critique Agent** → implemented through the explanation/critique pipeline for decision quality auditing
- **Memory Agent / Logging Layer** → maintains structured output history and diagnostics

So conceptually, these are described as **sub-agents**, while in code they are **implemented as tools/modules coordinated by one central agent**.

---

## Core Capability

The system answers this question:

> **Given recent gas sensor behavior and available thermal-image evidence, what safety action should be taken right now?**

It does this by combining:

- recent sensor dynamics from a sliding window
- anomaly estimation from an LSTM autoencoder
- a Dueling DQN policy over a 22-dimensional state
- YOLO visual confirmation / escalation logic
- explanation and critique outputs for operators

---

## System Architecture

```text
          Thermal Image Input                          Sensor Time-Series Input
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                  ┌───────────────────────────┐
   │        Vision Agent       │                  │       Sliding Window      │
   │      (YOLOv8 Classifier)  │                  │      20-step history      │
   │   - visual gas class      │                  │   - current / delta / std │
   │   - visual confidence     │                  └──────────────┬────────────┘
   └──────────────┬────────────┘                                 │
                  │                                              ▼
                  │                           ┌───────────────────────────────┐
                  │                           │        Anomaly Agent          │
                  │                           │    (LSTM Autoencoder)         │
                  │                           │  - raw anomaly score          │
                  │                           │  - normalized anomaly         │
                  │                           └──────────────┬────────────────┘
                  │                                          │
                  └───────────────────────┬──────────────────┘
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │              State Builder                   │
                    │      22-feature RL state construction        │
                    │  [ anomaly | current | delta | std ]         │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │              Decision Agent                  │
                    │       Dueling Deep Q-Network Policy          │
                    │                                              │
                    │  Actions:                                    │
                    │  0 → Monitor                                 │
                    │  1 → Increase Sampling                       │
                    │  2 → Request Verification                    │
                    │  3 → Raise Alert                             │
                    │  4 → Emergency Shutdown                      │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │         Safety Override / Guardrails         │
                    │  - deterministic safety escalation rules     │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Vision Verification / Escalation        │
                    │  - confirm or challenge sensor decision      │
                    │  - escalate on strong dangerous evidence     │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │    Explanation Agent + Critique Agent        │
                    │  - action interpretation                     │
                    │  - confidence / robustness audit             │
                    │  - operator-facing reasoning                 │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │         Final Logged Safety Output           │
                    │  - final action                              │
                    │  - Q-values / confidence                     │
                    │  - anomaly / vision result                   │
                    │  - explanation / critique                    │
                    └──────────────────────────────────────────────┘
```

---

## Current Action Space

The Deep Q-Network chooses one of **five safety actions**:

| Action ID | Action |
|-----------|--------|
| 0 | Monitor |
| 1 | Increase Sampling |
| 2 | Request Verification |
| 3 | Raise Alert |
| 4 | Emergency Shutdown |

Semantic gas mapping used throughout the project:

- **NoGas → gas_id 0 → Monitor**
- **Smoke → gas_id 1 → Raise Alert**
- **Mixture → gas_id 2 → Emergency Shutdown**
- **Perfume → gas_id 3 → Increase Sampling / Request Verification**

---

## Canonical Gas Mapping

This is the canonical ground truth for the system. Reward logic, correctness checks, evaluation, and explanation all use this semantic mapping.

| Gas Class | `gas_id` | Hazard Level | Correct Action | Action ID |
|-----------|----------|--------------|----------------|-----------|
| NoGas | 0 | Safe | Monitor | `0` |
| Smoke | 1 | **Danger** | Raise Alert | `3` |
| Mixture | 2 | **Danger** | Emergency Shutdown | `4` |
| Perfume | 3 | Low-risk VOC | Increase Sampling or Request Verification | `1` or `2` |

### Critical Implementation Note

Canonical gas meaning is always derived from the text label using:

```python
GAS_MAP = {"NoGas": 0, "Smoke": 1, "Mixture": 2, "Perfume": 3}
```

YOLO's internal class index order is **not** the same as this semantic gas mapping.

YOLO internal order in the current project is:

- `Mixture = 0`
- `NoGas = 1`
- `Perfume = 2`
- `Smoke = 3`

That means:

> YOLO class index must never be treated as the semantic `gas_id` directly.

The Vision Agent must convert:

- raw YOLO class index
- to text label
- then to canonical `gas_id` through `GAS_MAP`

---

## Core AI Pipeline

### Sensor Branch

The sensor branch is the main decision-making path.

It takes sequential gas sensor values and turns them into a structured temporal RL state.

### Vision Branch

The vision branch is currently a **verification / escalation branch**, not the primary policy driver.

That means:

- the Deep Q-Network makes the main decision from the engineered sensor/anomaly state
- the Vision Agent checks whether visual evidence supports or contradicts that decision
- dangerous high-confidence visual evidence can escalate the final action

### Explanation / Critique Branch

After the final action is chosen, the system generates:

- a detailed explanation of what happened
- a critique of how strong or fragile the decision is

---

## Model Architecture — Dueling Deep Q-Network

The current Decision Agent is a Dueling DQN operating on a **22-feature state vector**.

Conceptually:

```text
Input: 22-feature state vector
  │
  ├── hidden layers
  │
  ├── Value stream        -> V(s)
  └── Advantage stream    -> A(s, a)
            │
            ▼
Q(s, a) = V(s) + [A(s, a) − mean_a A(s, a')]
```

### Current Training Components

- Dueling DQN policy structure
- target network
- prioritized experience replay
- n-step returns
- epsilon-greedy exploration
- soft target network updates
- safety-oriented reward shaping
- MC dropout support during inference through the decision pipeline

### What the DQN Is Learning

The DQN is learning:

> **Given the current 22-dimensional state, which of the 5 safety actions gives the best expected safety outcome?**

---

## The 22-Feature State Vector

The current reinforcement learning state is:

```python
state = [
    anomaly,                                   # normalized anomaly score
    MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135,      # current sensor readings
    dMQ2, dMQ3, dMQ5, dMQ6, dMQ7, dMQ8, dMQ135,  # delta over window
    sMQ2, sMQ3, sMQ5, sMQ6, sMQ7, sMQ8, sMQ135,  # std over window
]
```

Total:

- 1 anomaly feature
- 7 current sensor features
- 7 delta features
- 7 standard deviation features

**State dimension = 22**

### Why This Matters

The DQN is **not** making decisions from:

- image features directly
- anomaly alone

It is making decisions from the **full temporal engineered sensor state**.

This is why some cases can still be correctly classified even when anomaly alone looks counterintuitive.

---

## Sliding Window

The system uses a **20-step sliding window** over the sensor stream.

### What the Window Captures

A single sensor row only tells the model what is happening now.

The sliding window tells the model:

- what the sensors look like now
- how they changed over time
- how stable or unstable they were across the recent interval

### Window-Derived Features

From each 20-step window, the system computes:

- **current** = the last row of the window
- **delta** = last row minus first row
- **std** = standard deviation of each sensor over the 20 rows

This temporal feature engineering is critical because it allows the DQN to detect patterns such as:

- rising concentration
- falling concentration
- stable background
- unstable or oscillating conditions

---

## Anomaly Agent — LSTM Autoencoder

The Anomaly Agent uses an LSTM autoencoder to estimate how unusual the current sensor pattern is.

### What It Does

- reconstructs the sensor reading
- measures reconstruction error
- converts that error into an anomaly score

### Outputs

- **raw anomaly**
- **normalized anomaly**

The normalized anomaly becomes the first feature in the 22-dimensional DQN state.

### Important Operational Detail

High anomaly does **not** automatically mean dangerous gas.

The anomaly score is just one input to the DQN.

The final action depends on the full state:

- anomaly
- current sensor values
- deltas
- standard deviations

That is why the model can still choose a mild action in some high-anomaly cases if the rest of the state indicates a non-hazardous pattern.

---

## Vision Agent — YOLOv8 Verification Branch

The Vision Agent uses YOLO classification on the thermal image.

### Current Role

The vision branch is currently used for:

- visual class prediction
- visual confidence estimation
- post-policy verification
- post-policy escalation

### What It Does Not Currently Do

The vision output is **not** currently part of the 22-dimensional DQN state.

### Current Escalation Logic

- low-confidence visual results are logged but not trusted strongly
- strong Smoke evidence can escalate to at least **Raise Alert**
- strong Mixture evidence can escalate to **Emergency Shutdown**
- NoGas / Perfume do not currently downgrade strong safety actions in Stage 1

---

## Explainability Layer — `gemma3:1b`

The current explainability layer is powered by a **local Ollama-served language model**, currently configured as:

```text
gemma3:1b
```

This model is used by the **Explanation Agent** through `ExplanationTool`.

### What the Explainability Layer Does

The explainability layer runs **after** the final action has already been selected by the main multimodal decision pipeline.

It does **not** control the safety action.

Instead, it helps interpret the structured result by generating:

- an explanation of why the final action was chosen
- a critique of how strong, fragile, or trustworthy the decision appears

### What It Explains

The explanation/critique layer interprets:

- the sensor state
- the anomaly score
- the DQN preference
- the raw action
- the safety-adjusted action
- the final action
- the vision branch output
- branch agreement or disagreement
- confidence and robustness cues

### Current Safety Design

The explainability layer is constrained by:

- precomputed interpretation labels
- prompt constraints
- contradiction validators
- structured fallback outputs when the free-form text is not sufficiently grounded

That means the LLM is being used as an **operator-facing reasoning layer**, not as the main decision-maker.

---

## MC Dropout

The current decision pipeline supports **MC Dropout**.

### What MC Dropout Means

Instead of doing one deterministic forward pass, the decision model can perform multiple stochastic forward passes with dropout active at inference time.

This gives:

- slightly different Q-values on each pass
- mean Q-values
- Q-value spread / standard deviation
- a policy confidence estimate

### Why It Matters

MC Dropout gives a practical uncertainty estimate:

- **low spread** = stable decision
- **high spread** = fragile decision

This helps the critique layer judge whether a decision is:

- robust
- moderately robust
- fragile

### Important Note

MC Dropout improves interpretability and uncertainty awareness, but it also increases runtime.

---

## Data Split and Training Logic

The reinforcement learning policy was trained offline from processed sliding-window samples.

### Stage 1 — Build Windows from Raw Sequential CSV

The raw gas sensor CSV is segmented into overlapping **20-step windows**.

Each window becomes one training sample.

### Stage 2 — Build Temporal Features

For each window:

- current sensor row is extracted
- delta across the window is computed
- standard deviation across the window is computed
- anomaly score is computed

### Stage 3 — Normalize Anomaly

Anomaly is normalized using train-only statistics so that deployment uses the same scale seen during training.

### Stage 4 — Construct the RL State

The final training state is:

```text
[anomaly | current | delta | std]
```

State dimension = **22**

### Stage 5 — Define the Reward Policy

The reward function encodes the operational safety policy:

- NoGas should favor **Monitor**
- Smoke should favor **Raise Alert**
- Mixture should favor **Emergency Shutdown**
- Perfume should favor **Increase Sampling** or **Request Verification**

Dangerous under-reactions are penalized more strongly than mild mistakes.

### Stage 6 — Train DQN Offline

The environment iterates through processed training samples.

At each step:

- current state is given to the DQN
- an action is selected
- reward is computed from the ground-truth gas class
- next state is returned

This trains the DQN to map structured temporal states to calibrated safety actions.

---

## Current Design Choice: DQN Is Sensor-State Driven

A very important implementation detail:

> The DQN is trained on the 22-dimensional engineered state, not directly on image features.

The image branch is currently a **post-decision verification branch**, not part of the DQN input state.

That means:

- the sensor/anomaly branch is the main policy driver
- the vision branch confirms or escalates the decision afterward

This was chosen so YOLO could be added safely without retraining the full RL model.

---

## Folder-Driven Live Testing Workflow

The current testing workflow is folder-driven and uses one image from each class folder:

- NoGas
- Smoke
- Mixture
- Perfume

For each chosen image, the system:

1. finds the matching CSV row
2. builds the 20-step sensor window ending at that row
3. computes anomaly
4. constructs the 22-dimensional state
5. runs the DQN
6. applies safety override
7. runs YOLO on the image
8. applies visual verification / escalation
9. generates explanation and critique
10. logs the result

This makes the live verification path consistent with the trained state representation.

---

## Dashboard Behavior

The current `app.py` dashboard matches the central multimodal pipeline.

It is no longer just a processed-test CSV DQN viewer.

It now:

- loads the central multimodal agent
- loads the raw CSV
- selects one usable image per class folder
- reconstructs the 20-step sensor window
- runs anomaly + DQN + safety override + YOLO
- optionally generates explanation and critique
- displays raw action, action after safety, final action, anomaly, confidence, YOLO outputs, explanation, critique, and latency

### Performance Note

The dashboard may be slow if these are enabled together:

- MC Dropout
- explanations
- critiques

That is because each run is doing a **heavy analysis workflow**, not only a lightweight forward pass.

---

## Explanation Agent and Critique Agent

The explanation and critique layers are operator-facing reasoning layers.

### Explanation Agent

The Explanation Agent explains:

- what the sensor state suggested
- what the DQN preferred
- whether safety override changed the action
- what the vision branch suggested
- whether the branches agreed
- why the final action was selected

### Critique Agent

The Critique Agent audits:

- policy confidence
- Q-gap strength
- decision robustness
- anomaly/action consistency
- vision agreement strength
- operational trustworthiness

### Current Implementation Detail

The explanation and critique layers use:

- precomputed interpretation labels from Python
- prompt constraints
- contradiction validators
- structured fallback text if the free-form output contradicts the structured evidence

This makes the language layer safer, even though it is still not perfect.

---

## Full Local Setup and Execution Guide

This section explains how to set up and run the full GasSafe AI system locally on **macOS**, including:

- Python virtual environment setup in VS Code
- Python package installation
- Ollama installation
- pulling and testing the `gemma3:1b` model
- the role of the explainability layer
- running `main.py`
- running the Streamlit dashboard `app.py`

---

## Local Requirements

At minimum, you need:

- **Python 3.10+ or 3.11+**
- **VS Code**
- **Ollama**
- your trained model files:
  - `models/DeepQnet.pth`
  - `models/lstm_autoencoder_weights.pth`
  - `models/yolov8_gas_classifier.pt`

---

## Step 1 — Open the Project in VS Code

Open VS Code and use **File → Open Folder**, then select the project root folder containing:

```text
main.py
app.py
src/
models/
```

---

## Step 2 — Create a Virtual Environment

Open a terminal in VS Code via **Terminal → New Terminal**.

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### Confirm activation

Your terminal prompt should change to show the environment name:

```text
(.venv) yourname@your-mac YourProject %
```

To deactivate at any time:

```bash
deactivate
```

---

## Step 3 — Upgrade pip

```bash
pip install --upgrade pip
```

---

## Step 4 — Install Python Libraries

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you need to install the core libraries manually:

```bash
pip install torch torchvision torchaudio pandas numpy scikit-learn ultralytics streamlit plotly pillow
```

Optional notebook tools:

```bash
pip install jupyter ipykernel
```

---

## Step 5 — Test Python Imports

```bash
python -c "import torch, pandas, numpy, sklearn, streamlit, plotly; print('Core imports OK')"
```

Then test YOLO:

```bash
python -c "from ultralytics import YOLO; print('YOLO import OK')"
```

---

## Step 6 — Install Ollama

### Option A — macOS Installer

Download the macOS installer from the official Ollama website at [ollama.com](https://ollama.com) and follow the installation instructions.

### Option B — Homebrew

If you have Homebrew installed:

```bash
brew install ollama
```

After installation, open a fresh terminal and verify it works:

```bash
ollama
```

If the CLI opens, Ollama is installed correctly.

---

## Step 7 — Pull `gemma3:1b`

Pull the explainability model:

```bash
ollama pull gemma3:1b
```

This downloads the local model used by the Explanation Agent.

---

## Step 8 — Test `gemma3:1b`

Run the model interactively:

```bash
ollama run gemma3:1b
```

Then type a simple prompt like:

```text
Explain why a gas safety system should not rely on anomaly score alone.
```

If you get a response, the local explainability backend is working.

Exit with:

```text
/bye
```

or `Ctrl + C`.

---

## Step 9 — Test the Ollama API

If your `ExplanationTool` talks to Ollama through the local API, test it manually:

```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"gemma3:1b","prompt":"Say hello from Ollama.","stream":false}'
```

If you receive JSON output, the API is reachable.

---

## Step 10 — Verify Your Model Files

Make sure these files exist in the project root:

```text
models/DeepQnet.pth
models/lstm_autoencoder_weights.pth
models/yolov8_gas_classifier.pt
```

---

## Step 11 — Verify Your Dataset Paths

The current system uses:

- a **raw CSV**
- an **image folder** with subfolders: `NoGas`, `Smoke`, `Mixture`, `Perfume`

Expected structure:

```text
Gas Sensors Measurements/Gas_Sensors_Measurements.csv
Thermal Camera Images/NoGas/
Thermal Camera Images/Smoke/
Thermal Camera Images/Mixture/
Thermal Camera Images/Perfume/
```

---

## Step 12 — Run `main.py`

The current `main.py` is the **folder-driven live test pipeline**.

It:

1. loads the DQN
2. loads the anomaly model
3. loads YOLO
4. optionally enables explanation/critique
5. loads the raw CSV
6. selects one usable image per class folder
7. reconstructs the 20-step sensor window
8. runs the central multimodal agent
9. logs results

Run it with:

```bash
python main.py
```

Typical console flow:

```text
Initializing tools...
Tools initialized successfully.

Running folder-driven live test...
Using label column: Gas
Using image-name column: Corresponding Image Name
Selected images for testing:
  NoGas: ...
  Smoke: ...
  Mixture: ...
  Perfume: ...
```

Then it processes each class and prints the raw action, action after safety, final action, anomaly, confidence, YOLO outputs, explanation, and critique.

---

## Step 13 — Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard typically opens at:

```text
http://localhost:8501
```

The dashboard lets you:

- configure model paths
- configure raw CSV path and image folder path
- toggle MC Dropout, explanations, and critique
- run the folder-driven live test
- inspect the full decision chain: raw action → safety-adjusted → final
- inspect YOLO outputs, explanation, and critique per case
- export structured results as CSV

---

## Step 14 — Recommended First Run Order

Use this order the first time you set everything up:

1. open VS Code in the project root
2. create and activate the virtual environment
3. upgrade pip
4. install Python libraries
5. test Python imports
6. install Ollama
7. pull `gemma3:1b`
8. test `ollama run gemma3:1b`
9. verify model files exist
10. verify raw CSV and image folders exist
11. run `python main.py`
12. run `streamlit run app.py`

---

## Fast Debug Mode vs Full Analysis Mode

### Fast Debug Mode

Use this when you want the system to run more quickly. In the dashboard, set:

- **MC Dropout = Off**
- **Show Explanations = Off**
- **Show Critique = Off**

### Full Analysis Mode

Use this when you want detailed reasoning and audit output. Set:

- **MC Dropout = On**
- **Show Explanations = On**
- **Show Critique = On**

This is slower, but it gives full interpretability.

---

## Current Strengths

- clear 22-feature RL state design
- anomaly-aware temporal reasoning
- post-policy visual verification
- deterministic safety guardrails
- centralized agentic orchestration
- explanation and critique outputs
- strong logging and debugging visibility
- semantic gas mapping is explicit and safer
- local LLM-powered explainability layer via Ollama + `gemma3:1b`

---

## Current Limitations

- not a fully distributed multi-agent architecture
- YOLO is not yet part of the DQN state itself
- explanation and critique quality still depend on strong prompt and validator design
- latency is still high when explanation and critique are enabled together
- some decisions remain counterintuitive because anomaly is only one part of the full state
- dashboard runtime becomes heavy when MC Dropout, explanation, and critique are all enabled

---

## Repository Structure

```text
src/
├── agent/
│   ├── agent_core.py
│   ├── safety.py
│   ├── reward_system.py
│   ├── memory.py
│   ├── goal_manager.py
│   ├── critic.py
│   └── metrics_logger.py
│
├── tools/
│   ├── anomaly_tool.py
│   ├── decision_tool.py
│   ├── vision_tool.py
│   └── explanation_tool.py
│
main.py
app.py
models/
├── DeepQnet.pth
├── lstm_autoencoder_weights.pth
└── yolov8_gas_classifier.pt
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Gas sensor anomaly estimation | LSTM Autoencoder (PyTorch) |
| RL action policy | Dueling Deep Q-Network (PyTorch) |
| Visual verification | YOLOv8 (Ultralytics) |
| Explainability layer | Ollama + `gemma3:1b` via `ExplanationTool` |
| Dashboard | Streamlit |
| Data processing | Pandas, NumPy, scikit-learn |

---

## Running the System

### CLI Verification

```bash
python main.py
```

This runs the folder-driven live verification flow and logs anomaly values, Q-values, policy confidence, vision outputs, final actions, explanation, and critique.

### Streamlit Dashboard

```bash
streamlit run app.py
```

This launches the dashboard interface for operator-style monitoring.

---

## Recommended Terminology

For documentation, presentations, and portfolio use, the most accurate wording is:

- **central multimodal agent**
- **specialized functional sub-agents**
- **implemented as tools/modules in code**

That gives the best balance between conceptual clarity and implementation truth.

---

## Practical One-Line Description

> **A multimodal agentic gas safety decision system that combines temporal sensor modeling, anomaly detection, Dueling DQN action selection, thermal-image verification, and operator-facing interpretability.**

---

## Short Repo Summary

```text
This project is an agentic multimodal industrial gas safety system.

It uses:
- gas sensor time-series windows
- an LSTM autoencoder for anomaly estimation
- a Dueling Deep Q-Network for autonomous safety action selection
- YOLO-based thermal-image verification
- an Ollama-powered explanation layer using gemma3:1b
- explanation and critique layers for operator-facing interpretability

The system is agentic because it perceives, interprets, decides, verifies, explains, and logs autonomously.
It is not a distributed multi-agent swarm, but a central multimodal agent with specialized decision-support sub-agents.
```

---

## Final Note

If you are describing this project publicly, the most accurate summary is:

- the system is **agentic**
- the main decision policy is **Dueling DQN over a 22-dimensional sensor/anomaly state**
- the Anomaly Agent estimates abnormality from sensor reconstruction error
- the Vision Agent acts as a **visual verifier / escalation branch**
- the Explainability Layer uses **Ollama + `gemma3:1b`**
- the architecture is **centralized and modular**
- the system uses **specialized sub-agents implemented as tools/modules in code**
