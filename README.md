# GasSafe AI — Multimodal Agentic Gas Safety System

> **Agentic multimodal safety system for gas monitoring and action selection.**  
> Dueling Deep Q-Network · LSTM Autoencoder Anomaly Detection · YOLOv8 Verification · Safety Guardrails · Explanation & Critique Layer

---

## What This System Does

GasSafe AI is a multimodal, agentic industrial gas safety system that continuously monitors an environment using two input modalities:

- a **7-channel MQ gas sensor array**
- a **thermal image branch**

The system does not only classify gas conditions. It performs an **agentic safety workflow**:

1. reads recent sensor behavior
2. estimates anomaly
3. builds a structured temporal state
4. chooses a safety action using reinforcement learning
5. applies hard safety rules
6. verifies the decision with visual evidence
7. generates explanation and critique outputs
8. logs the result for operator review

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

## Current Strengths

- clear 22-feature RL state design
- anomaly-aware temporal reasoning
- post-policy visual verification
- deterministic safety guardrails
- centralized agentic orchestration
- explanation and critique outputs
- strong logging and debugging visibility
- semantic gas mapping is now explicit and safer

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
| Explanation / critique | explanation tool + structured validators |
| Dashboard | Streamlit |
| Data processing | Pandas, NumPy, scikit-learn |

---

## Running the System

### CLI Verification

```bash
python main.py
```

This runs the folder-driven live verification flow and logs:

- anomaly values
- Q-values
- policy confidence
- vision outputs
- final actions
- explanation
- critique

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
- the architecture is **centralized and modular**
- the system uses **specialized sub-agents implemented as tools/modules in code**
