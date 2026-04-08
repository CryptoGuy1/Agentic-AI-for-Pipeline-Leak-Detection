# GasSafe AI — Multimodal Autonomous Gas Detection System

> **Production-grade autonomous safety agent for gas station monitoring.**  
> Dueling Double DQN · LSTM Autoencoder Anomaly Detection · YOLOv8 · Prioritized Experience Replay · Streamlit Dashboard

---

## What This System Does

GasSafe AI is a multimodal reinforcement learning safety system that continuously monitors a gas station environment using two sensor modalities — a 7-channel MQ gas sensor array and a thermal/optical camera — and autonomously selects the correct safety response in real time.

The system detects four gas conditions and responds with one of five calibrated safety actions. Validated results on a leakage-safe block-wise holdout test set:

- **92.55% overall decision accuracy** (1,181 / 1,276 test samples)
- **0.0% danger miss rate** — no Smoke or Mixture event was silently ignored
- **0.0% false alarm rate** — no unnecessary emergency response on clean air
- **Sub-millisecond inference** — ~0.46ms mean latency on CPU

---

## Core AI Pipeline

```
7 MQ Gas Sensors                      Thermal / Optical Camera
(MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135)         │
          │                                     │
          └─────────────────┬───────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │       Perception Layer        │
            │  YOLOv8 → gas class label     │
            │  LSTM AE → anomaly score      │
            └───────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Feature Engineering          │
            │  20-step rolling window:      │
            │  [anomaly | current | delta   │
            │             | std]  = 22 dim  │
            └───────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │     Decision Layer            │
            │  Dueling Double DQN + PER     │
            │  → 1 of 5 safety actions      │
            └───────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │   Explainability Layer        │
            │  Rule-based expert engine     │
            │  (no LLM, always active)      │
            └───────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │   GasSafe AI Dashboard        │
            │   Streamlit SCADA Interface   │
            └───────────────────────────────┘
```

---

## Gas Classification and Action Mapping

This is the canonical ground truth for the entire system. Every reward function, evaluation metric, and explanation uses exactly these mappings.

| Gas Class | `gas_id` | Hazard Level | Correct Action | Action ID |
|-----------|----------|--------------|----------------|-----------|
| NoGas | 0 | Safe | Monitor | `0` |
| Smoke | 1 | **DANGER** | Raise Alarm | `3` |
| Mixture | 2 | **DANGER** | Emergency Shutdown | `4` |
| Perfume | 3 | Low risk VOC | Increase Sampling or Request Verification | `1` or `2` |

> **Critical implementation note:** `gas_id` is always derived from the text label via `GAS_MAP`, never from YOLOv8's internal class index. YOLO's class ordering (`Mixture=0, NoGas=1, Perfume=2, Smoke=3`) differs from `GAS_MAP` and must never be used as ground-truth `gas_id`. This was a major bug that caused the entire policy to be semantically inverted before the fix.

```python
GAS_MAP = {"NoGas": 0, "Smoke": 1, "Mixture": 2, "Perfume": 3}  # canonical — never change
```

---

## Model Architecture — Dueling Double DQN

```
Input: 22-feature state vector
  │
  ├── Linear(22 → 256) → LayerNorm → ReLU → Dropout(0.15)
  ├── Linear(256 → 256) → LayerNorm → ReLU → Dropout(0.15)
  └── Linear(256 → 128) → LayerNorm → ReLU
                   │
       ┌───────────┴───────────┐
       │                       │
  Value Stream             Advantage Stream
  Linear(128→64)→ReLU      Linear(128→64)→ReLU
  Linear(64→1) = V(s)      Linear(64→5) = A(s,a)
       │                       │
       └───────────┬───────────┘
                   │
        Q(s,a) = V(s) + [A(s,a) − mean_a A(s,a')]
```

**Training components:**
- Double DQN: online network selects action, target network evaluates it
- Prioritized Experience Replay (SumTree, iterative, no Python recursion)
- N-step returns (N=3) for faster credit assignment
- Soft target network updates (tau=0.005)
- Orthogonal weight initialization with small output gain (0.01)
- LayerNorm instead of BatchNorm (works with batch_size=1 at inference)

---

## The 22-Feature State Vector

```python
state = [
    anomaly,                         # normalized LSTM AE reconstruction error
    MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135,   # current sensor readings
    dMQ2, dMQ3, dMQ5, dMQ6, dMQ7, dMQ8, dMQ135,  # delta: last - first over window
    sMQ2, sMQ3, sMQ5, sMQ6, sMQ7, sMQ8, sMQ135,  # std over 20-step window
]
```

The delta and std features are critical — they capture whether gas levels are **rising** (alarming) or **falling** (recovering). This is especially important for Mixture gas, which produces a low anomaly score because the LSTM AE reconstructs it well, but has distinctive temporal sensor patterns in the delta/std features.

---

## Data Split — Block-Wise Holdout with Boundary Trim

### Why not a random split?

Overlapping 20-step windows mean window `i` shares 19 raw rows with window `i+1`. A random 80/20 split puts approximately 24,000 overlapping window-pairs across train and test, artificially inflating test accuracy by ~6%.

### Why not `iloc[:80%]`?

The raw CSV is arranged in sequential gas blocks. Index 5,104 falls inside the Perfume block, making the test set 100% Perfume — a completely invalid benchmark.

### The correct approach

```
For each gas class block:
  [BOUNDARY_TRIM=20][══════ TRAIN ══════][GAP=20][═════ TEST ═════]

NoGas:   [trim=20][train=1,224][gap=20][test=316]
Smoke:   [trim=20][train=1,240][gap=20][test=320]
Mixture: [trim=20][train=1,240][gap=20][test=320]
Perfume: [trim=20][train=1,240][gap=20][test=320]
─────────────────────────────────────────────────────────
         Total train: 4,944   |   Total test: 1,276
         Raw-row overlap: 0 rows (mathematically verified)
```

**Why `BOUNDARY_TRIM=20`?** Without trimming, the start of each class block shares raw rows with the end of the previous class's test windows (57 rows total across 3 boundaries). Trimming 20 windows from each block's start eliminates all cross-class boundary contamination.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 200 |
| Training samples | 4,944 |
| Test samples | 1,276 |
| State dimensions | 22 |
| Actions | 5 |
| Optimizer | Adam lr=2e-4 |
| Gamma | 0.99 |
| N-step | 3 |
| Batch size | 128 |
| Buffer | 100,000 |
| Learn every | 4 steps |
| Tau (soft update) | 0.005 |
| Gradient clip | 5.0 |
| Epsilon decay | 1.0 → 0.05 over 150 episodes |
| Reward scale | `/3.0` clipped to `[-2, +2]` |

---

## Reward Function

```python
# src/agent/reward_system.py
CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
DANGER_GAS_IDS  = {1, 2}

def compute_reward(state, action, gas_id, anomaly=None):
    norm_anomaly = anomaly if anomaly is not None else float(state[0])
    
    reward = 0.0
    if action in CORRECT_ACTIONS[gas_id]:
        reward += 2.0
        if gas_id in DANGER_GAS_IDS:
            reward += norm_anomaly          # severity bonus (0–1)
    else:
        reward -= 2.0

    if gas_id in DANGER_GAS_IDS and action == 0:
        reward -= 10.0                      # worst case: ignored dangerous gas
    if gas_id == 0 and action >= 3:
        reward -= 4.0                       # false emergency alarm

    if gas_id == 1 and action == 3: reward += 1.0   # Smoke → Raise Alarm
    if gas_id == 2 and action == 4: reward += 1.0   # Mixture → Emergency

    if gas_id == 3:
        if   anomaly > 0.5 and action == 2: reward += 0.8   # high VOC → verify
        elif anomaly > 0.5 and action == 1: reward -= 0.5   # under-response
        elif anomaly <= 0.5 and action == 1: reward += 0.5  # low VOC → sample

    return float(np.clip(reward / 3.0, -2.0, 2.0))
```

---

## Validated Results

| Metric | Value |
|--------|-------|
| Overall decision accuracy | **92.55%** (1,181 / 1,276) |
| Danger miss rate | **0.0000%** (0 / 636 hazardous samples) |
| False alarm rate | **0.0000%** (0 / 652 NoGas alerts) |
| Policy entropy | 1.5187 (max theoretical = 1.609 for 5 actions) |
| Action 2 used | 81 times — mid-severity Perfume response active |
| Mean inference latency | ~0.46ms CPU |
| Latency P99 | ~0.73ms CPU |

**Per-class classification report:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| NoGas | 0.96 | 0.99 | 0.97 | 320 |
| Smoke | 0.93 | 0.85 | 0.88 | 321 |
| Mixture | 0.85 | 0.97 | 0.90 | 315 |
| Perfume | 0.98 | 0.90 | 0.94 | 320 |
| **Macro avg** | **0.93** | **0.93** | **0.93** | **1,276** |

> **On the Mixture anomaly paradox:** Mixture gas shows anomaly ≈ 0.10 — the lowest of all four classes. This is expected and by design. The LSTM autoencoder was trained on all gas types and learned to reconstruct Mixture sensor patterns with low mean-squared error, producing a paradoxically low anomaly score for the most dangerous gas. The DQN correctly handles this by classifying Mixture using the 14 temporal features (delta + std), not the anomaly signal.

---

## Key Bugs Fixed

| Bug | Severity | Effect | Fix Applied |
|-----|----------|--------|------------|
| `gas_id` from YOLO class index | **Critical** | Policy semantically inverted — Mixture→Monitor, NoGas→Alarm | Use `GAS_MAP[lbl]` not `int(gid)` |
| `target_q.clamp(-5, 5)` | **Critical** | All Q-values converge to ~5.0, policy random (confidence 0.0003) | Remove clamp entirely |
| `reward / 6.0` clipped `[-1, +1]` | **High** | Reward too narrow, feeds Q-value saturation | Change to `/ 3.0`, clip `[-2, +2]` |
| `reward_system.py` anomaly thresholds | **Critical** | Mixture (anomaly≈0.10) rewarded for Monitor, not Emergency | Rewrite using `gas_id + action` |
| Random split on windowed data | **High** | ~24,000 overlapping raw rows inflate test accuracy | Block-wise holdout with gap=20 |
| Cross-class boundary overlap (57 rows) | **Medium** | GAP handles within-class but not cross-class | Add `BOUNDARY_TRIM=20` |
| `MAX_STEPS_PER_EPISODE = 40` | **Medium** | Only 160 / 1,276 test samples evaluated | Change to 400 |
| `ENABLE_EXPLANATIONS = False` | **Medium** | Explanations silently disabled | Set to `True`; use rule-based engine |

---

## Explainability — Rule-Based, No LLM Required

Every decision automatically generates two outputs in both `main.py` and `app.py`:

**Decision Explanation** covers:
- Why this gas class was identified given its sensor profile
- What the anomaly score means for this class (including the Mixture paradox)
- Q-value gap analysis and confidence level description
- The exact safety protocol the response triggers

**Safety Critique** covers:
- Confidence level vs the 0.25 operational safety threshold
- Whether the decision meets gas station safety standards
- Risk implications of an incorrect decision in deployment
- Architecture notes explaining counterintuitive readings

Both are generated from the decision data using domain-expert rules — fast, deterministic, no API, no Ollama.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Gas detection (vision) | YOLOv8 (ultralytics) |
| Anomaly detection | LSTM Autoencoder (PyTorch) |
| RL decision model | Dueling Double DQN + PER (PyTorch) |
| Explainability | Rule-based domain engine (Python) |
| Dashboard | Streamlit |
| Charts | Plotly |
| Data processing | Pandas, NumPy, scikit-learn |

---

## Project File Structure

```
GasSafeAI/
├── app.py                           # Streamlit dashboard — run this
├── main.py                          # CLI verification runner
├── test_df_processed.csv            # Normalized test set (export from notebook)
│
├── models/
│   ├── DeepQmodel.pth               # Trained Dueling DQN weights
│   └── lstm_autoencoder_weights.pth # LSTM autoencoder
│
├── gas_dqn_honest_split.ipynb       # Production training notebook
├── notebook_patches.ipynb           # Critical bug fixes (apply before training)
├── gas_id_fix_patch.ipynb           # Gas-id mismatch fix and verification cell
├── zero_overlap_split.ipynb         # Leakage-safe split with boundary trim
│
├── src/
│   ├── agent/
│   │   ├── agent_core.py            # MultimodalAgent (live streaming mode)
│   │   ├── reward_system.py         # Reward function (gas_id-based)
│   │   ├── memory.py                # Short-term memory buffer
│   │   └── goal_manager.py          # Goal tracking
│   └── tools/
│       ├── decision_tool.py         # DQN model loader and inference
│       ├── anomaly_tool.py          # LSTM AE wrapper
│       └── explanation_tool.py      # Optional LLM explanation (Ollama)
│
├── evaluation_log.csv               # Per-step output (auto-generated)
├── episode_summary.csv              # Per-class summary (auto-generated)
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/yourname/gassafe-ai.git
cd gassafe-ai

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit>=1.28.0
plotly>=5.0.0
pandas>=1.5.0
numpy>=1.23.0
torch>=2.0.0
ultralytics
scikit-learn
```

---

## Running the System

### CLI Verification (evaluate on full test set)

```bash
python main.py
```

Output: per-step actions, correctness checks, full explanations, safety critiques, overall accuracy, danger miss rate, false alarm rate. Results auto-saved to `evaluation_log.csv` and `episode_summary.csv`.

### Dashboard

```bash
streamlit run app.py
```

Opens the GasSafe AI Control Dashboard at `http://localhost:8501`.

---

## Training the Model

Open `gas_dqn_honest_split.ipynb` and apply all patches from `notebook_patches.ipynb` and `gas_id_fix_patch.ipynb` before running.

**Key cells to verify before training:**

| Cell | What to check |
|------|---------------|
| Cell 4 — Dataset building | `gas_id = GAS_MAP[lbl]` not `int(gid)` |
| Cell 5 — Split | Uses `block_wise_holdout()` with `gap=20` and `boundary_trim=20` |
| Cell 7 — Reward | Uses `gas_id + action`, not anomaly thresholds |
| Cell 11 — Agent.learn | NO `.clamp(-5, 5)` on `target_q` |
| After Cell 6 — Export | Export cell saves `test_df_processed.csv` with correct normalization |

---

## Sensor Reference

| Sensor | Primary Detection | Notes |
|--------|-------------------|-------|
| MQ2 | LPG, Smoke | Propane, hydrogen, methane, alcohol |
| MQ3 | Alcohol, Ethanol | Vapors, smoke |
| MQ5 | LPG, Coal Gas | Natural gas, LPG |
| MQ6 | LPG, Butane | Iso-butane, propane |
| MQ7 | Carbon Monoxide | CO from incomplete combustion |
| MQ8 | Hydrogen | Hydrogen gas, alcohol |
| MQ135 | Air Quality | Ammonia, benzene, NOx, CO2 |

---

## License

MIT License — see `LICENSE` for full terms.
