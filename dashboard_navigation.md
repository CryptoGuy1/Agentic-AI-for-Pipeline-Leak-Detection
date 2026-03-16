# Methane AI Control Room  
## Dashboard Navigation Guide

This document explains how to navigate and understand the **Methane AI Control Room Dashboard (v8.0)**.

The dashboard is designed as a **real-time methane pipeline monitoring system** integrating:

- Computer Vision (YOLO gas detection)
- LSTM Autoencoder anomaly detection
- Reinforcement Learning decision agent
- AI reasoning with **Gemma3:1b**
- Sensor telemetry monitoring
- Incident detection and reporting

The interface mimics a **SCADA-style industrial monitoring control room** used in pipeline safety systems.

---

# Dashboard Layout

The dashboard contains three main components:

```
Methane AI Control Room
│
├── Sidebar (Control Panel)
│
└── Main Dashboard
     ├── KPI Monitoring
     ├── Threat Status Strip
     └── Analytical Tabs
```

---

# 1. Sidebar Control Panel

The sidebar provides **operational controls and system configuration**.

## Cycle Control

Controls the execution of monitoring cycles.

| Button | Function |
|------|------|
| ▶ Run | Executes one monitoring cycle |
| ↺ Reset | Clears system history and incidents |

Each cycle performs:

1. Sensor data acquisition
2. Camera image analysis
3. Anomaly detection
4. RL decision selection
5. Risk classification
6. AI explanation generation

---

## Camera Frame Upload

Operators can upload a **pipeline inspection image**.

Supported formats:

```
PNG
JPG
JPEG
```

If no image is uploaded, the system continues operating using **sensor telemetry only**.

---

## Agent Settings

### Agent Mode

| Mode | Description |
|----|----|
| Autonomous | AI automatically decides response |
| Semi-Auto | Operator confirms AI decisions |
| Manual | Human operator decides response |

---

### Alert Threshold

Determines when an anomaly becomes an incident.

Example:

```
Threshold = 0.50
```

If

```
Anomaly Score > 0.50
```

then the system records an **incident event**.

---

## Environmental Controls

Simulated environmental parameters.

### Temperature

```
Range: 15°C – 55°C
```

### Humidity

```
Range: 20% – 95%
```

These parameters influence the anomaly detection model.

---

## Theme Control

Switch interface theme.

```
☀ Light Mode
☾ Dark Mode
```

---

## System Status

Displays real-time system health.

| Component | Description |
|---|---|
| Agent | Shows if AI agent is running |
| Gemma | Status of Ollama LLM |
| Mode | Current operation mode |
| Cycles | Number of monitoring cycles |
| Incidents | Total anomalies detected |
| Camera | Current uploaded frame |

---

# 2. Main Dashboard

The main interface provides **real-time monitoring and operational insight**.

---

# KPI Monitoring Row

Displays key operational metrics.

## Threat Score

Measures likelihood of methane leakage.

```
0 – 25   Safe
25 – 50  Elevated
50 – 75  High
75 – 100 Critical
```

---

## Gas Concentration

Measured in **parts per million (ppm)**.

| Range | Status |
|----|----|
| <20 ppm | Safe |
| 20–80 ppm | Warning |
| >80 ppm | Dangerous |

---

## AI Confidence

Indicates certainty of the AI decision.

```
Confidence = model certainty level
```

---

# Threat Status Strip

Displays current system state.

Example:

```
● HIGH RISK
Action: Raise Alarm
Threshold: 0.50
Updated: 14:21:55
```

---

# 3. Dashboard Tabs

The dashboard is divided into **nine analytical views**.

```
Overview
Telemetry
Sensors
RL Decisions
RL Metrics
Analytics
Incidents
Correlation
AI Analyst
```

---

# Tab 1 — Overview

Provides a **high-level system summary**.

Includes:

- Anomaly timeline
- Gas concentration trend
- RL reward performance
- Threat gauge
- AI explanation from Gemma

---

# Tab 2 — Telemetry

Displays **real-time monitoring telemetry**.

Visualizations include:

- anomaly score timeline
- methane concentration timeline
- confidence distribution
- reward trends

---

# Tab 3 — Sensors

Displays activity from the **7-sensor methane monitoring array**.

Components include:

### Sensor Bar Chart

Displays average sensor readings by zone.

### Sensor Heatmap

Visualizes temporal sensor activity.

### Sensor Sparklines

Mini charts showing recent sensor behavior.

---

# Tab 4 — RL Decisions

Shows how the **Reinforcement Learning agent chooses actions**.

Available actions:

```
Monitor
Increase Sampling
Verify
Raise Alarm
Shutdown
```

Displays:

- Q-values for each action
- risk distribution
- action frequency

---

# Tab 5 — RL Metrics

Evaluates reinforcement learning performance.

Metrics include:

| Metric | Description |
|---|---|
| Episode Reward | Total accumulated reward |
| Policy Entropy | Decision randomness |
| False Alarm Rate | Incorrect alarm triggers |
| Avg Latency | Inference speed |
| Episodes | Total cycles executed |

---

# Tab 6 — Analytics

Provides **historical monitoring analytics**.

Includes:

- environmental trends
- confidence distributions
- monitoring logs
- CSV export of monitoring data

---

# Tab 7 — Incidents

Displays detected methane events.

Information includes:

- cycle number
- timestamp
- risk level
- methane concentration
- event description

---

# Tab 8 — Correlation

Performs statistical analysis.

Includes:

- anomaly vs gas correlation
- rolling anomaly averages
- statistical deviation analysis

This helps detect **system behaviour trends**.

---

# Tab 9 — AI Analyst (Gemma)

The AI analyst is powered by:

```
Gemma3:1b
running locally via Ollama
```

Capabilities include:

- explaining AI decisions
- summarizing monitoring sessions
- diagnosing anomalies
- recommending operator actions

Example queries:

```
Why was an alarm raised?
Is the pipeline safe now?
Summarise the last 10 cycles
What caused the anomaly?
```

---

# Monitoring Cycle Workflow

Each monitoring cycle follows the pipeline below.

```
Sensor Data
     ↓
LSTM Autoencoder
     ↓
Anomaly Score
     ↓
Gas Estimation
     ↓
RL Decision
     ↓
Risk Classification
     ↓
Gemma AI Explanation
```

---

# Conclusion

The **Methane AI Control Room Dashboard** provides an integrated monitoring platform for:

- methane leak detection
- pipeline anomaly monitoring
- reinforcement learning decision support
- AI-assisted operational analysis

The system is designed to simulate **industrial pipeline monitoring infrastructure**, enabling operators to identify methane leaks and respond to risks in real time.
