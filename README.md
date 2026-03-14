# MethaneSAFE — Multimodal AI Control Room for Industrial Methane Monitoring

MethaneSAFE is a multimodal AI monitoring platform designed to detect methane leaks and hazardous gas conditions in industrial environments.

The system integrates:

- Computer Vision
- Sensor Anomaly Detection
- Reinforcement Learning Decision Policy
- Explainable AI Reasoning
- Real-Time Monitoring Dashboard

These components create a fully integrated **AI-powered industrial monitoring control room**.

---

# Project Overview

Industrial methane leaks can cause:

- Explosions
- Toxic exposure
- Environmental damage
- Financial losses

Traditional monitoring systems rely on static thresholds and manual inspection.

MethaneSAFE introduces an **autonomous AI monitoring agent capable of:**

- Detecting gas anomalies
- Interpreting camera feeds
- Making decisions using reinforcement learning
- Explaining decisions using a local LLM
- Presenting insights in a real-time monitoring dashboard

---

# Core Capabilities

The system implements the following AI pipeline:

```
Camera Feed + Gas Sensors
        │
        ▼
Computer Vision Gas Detection (YOLOv8)
        │
        ▼
Sensor Anomaly Detection (LSTM Autoencoder)
        │
        ▼
Reinforcement Learning Decision Policy
        │
        ▼
Safety Critic + Supervisor Agent
        │
        ▼
Explainable AI Reasoning (Gemma via Ollama)
        │
        ▼
Industrial Monitoring Dashboard (Streamlit)
```

---

# System Architecture

```
Sensors / Camera
       │
       ▼
Perception Layer
   ├── Vision Tool (YOLOv8)
   └── Sensor Anomaly Tool (LSTM Autoencoder)

       │
       ▼
Decision Layer
   └── RL Policy (Deep Random Forest)

       │
       ▼
Agent Intelligence
   ├── Memory
   ├── Goal Manager
   ├── Critic Agent
   └── Supervisor Agent

       │
       ▼
Explainability Layer
   └── Gemma LLM (Ollama)

       │
       ▼
Visualization
   └── Streamlit Control Room Dashboard
```

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| Computer Vision | YOLOv8 |
| Anomaly Detection | LSTM Autoencoder |
| Decision Model | Reinforcement Learning |
| Explainability | Gemma LLM |
| LLM Runtime | Ollama |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Language | Python |

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/yourname/methanesafe.git
cd methanesafe
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```
streamlit
plotly
pandas
numpy
torch
ultralytics
ollama
```

---

# Installing Ollama

The system uses **Ollama** to run a local large language model.

Download:

```
https://ollama.com/download
```

Verify installation:

```bash
ollama --version
```

---

# Pulling the Gemma Model

Download the model:

```bash
ollama pull gemma3:1b
```

Test the model:

```bash
ollama run gemma3:1b
```

Example prompt:

```
explain methane detection
```

---

# Running the AI System

Start the monitoring dashboard:

```bash
streamlit run app.py
```

---

# How the System Works

Each monitoring cycle follows these steps.

## Step 1 — Sensor Data Acquisition

Sensors capture environmental readings such as:

- methane concentration
- temperature
- humidity
- pressure
- gas mixture levels

Example sensor matrix:

```
50 x 7
```

Representing:

```
time window x sensor channels
```

---

## Step 2 — Computer Vision Gas Detection

YOLOv8 processes the camera feed.

Possible classifications:

```
NoGas
Smoke
Mixture
Perfume
```

Outputs:

```
gas_class
confidence
```

---

## Step 3 — Sensor Anomaly Detection

An LSTM autoencoder analyzes temporal sensor patterns.

Output:

```
anomaly_score ∈ [0,1]
```

---

## Step 4 — State Vector Construction

Example state vector:

```python
state = [
    anomaly_score,
    gas_concentration,
    vision_confidence,
    temperature,
    humidity
]
```

---

## Step 5 — RL Policy Decision

Available actions:

| Action | Meaning |
|------|------|
| 0 | Monitor |
| 1 | Increase Sampling |
| 2 | Request Verification |
| 3 | Raise Alarm |
| 4 | Emergency Shutdown |

Example Q-values:

```
[0.18, -0.06, 0.07, -0.04, 0.02]
```

The highest value determines the action.

---

# Reinforcement Learning Metrics

## Episode Reward

```
episode_reward
```

Total reward accumulated during a monitoring episode.

---

## Policy Entropy

```
policy_entropy
```

Measures exploration vs exploitation.

---

## False Alarm Rate

```
false_alarm_rate = false_alarms / total_alerts
```

---

## Average Latency

```
avg_latency
```

Measures system response time per monitoring cycle.

---

# Dashboard Components

The monitoring dashboard includes:

### KPI Panel

Displays:

- threat score
- gas concentration
- AI confidence
- monitoring cycles
- active incidents

### Sensor Telemetry

Displays:

- anomaly score timeline
- gas concentration trends

### Threat Gauge

Real-time risk score.

### Sensor Heatmap

Displays sensor intensity across time.

### RL Decision Visualization

Bar chart showing Q-values for actions.

### Incident Log

Displays anomalies with timestamps.

### AI Explanation Panel

Uses Gemma to explain decisions.

---

# File Structure

```
MethaneSAFE/
│
├── app.py
│
├── models/
│   ├── yolov8_gas_classifier.pt
│   ├── lstm_autoencoder_weights.pth
│   └── drf_gas_model.pth
│
├── src/
│   ├── agent/
│   │   ├── agent_core.py
│   │   ├── critic_agent.py
│   │   ├── supervisor_agent.py
│   │   ├── memory.py
│   │   ├── goal_manager.py
│   │   └── rl_metrics.py
│   │
│   ├── tools/
│   │   ├── vision_tool.py
│   │   ├── anomaly_tool.py
│   │   ├── decision_tool.py
│   │   └── explanation_tool.py
│
├── logs/
│   └── agent_run.json
│
├── requirements.txt
│
└── README.md
```

---

# Logging

Monitoring logs are stored in:

```
logs/agent_run.json
```

Fields include:

```
state
action
reward
latency
explanation
critic_review
timestamp
```

---

# Future Improvements

Planned upgrades:

- real IoT sensor streaming
- live camera inference
- reinforcement learning training loop
- distributed monitoring across multiple sites
- predictive incident detection
- automated emergency shutdown triggers

---

# Conclusion

MethaneSAFE demonstrates how modern AI systems combine:

- perception
- reasoning
- reinforcement learning
- explainability
- visualization

to create **autonomous industrial safety monitoring systems**.

