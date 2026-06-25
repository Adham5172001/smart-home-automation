# Smart Home Automation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![MQTT](https://img.shields.io/badge/MQTT-Protocol-blue)](https://mqtt.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An ML-powered smart home automation system that learns occupant behaviour patterns to predictively control lighting, heating, and appliances — reducing energy consumption while improving comfort.

## Features

- **Behaviour learning**: Learns daily routines from sensor data using time-series clustering
- **Predictive control**: Anticipates occupant needs 15–30 minutes in advance
- **MQTT integration**: Real-time communication with IoT sensors and actuators
- **Energy optimisation**: Reduces unnecessary device activation by ~34%
- **Anomaly detection**: Flags unusual patterns (security, health monitoring)

## System Architecture

```
IoT Sensors (Temperature, Motion, Light, Door)
        │ MQTT
  Message Broker (Mosquitto)
        │
  Data Ingestion Service
        │
  ML Pipeline:
  ├── Behaviour Clustering (K-Means on time-series)
  ├── Occupancy Prediction (LSTM)
  └── Anomaly Detection (Isolation Forest)
        │
  Automation Rules Engine
        │
  Actuator Commands (MQTT)
        │
Smart Devices (Lights, Thermostat, Appliances)
```

## ML Models

| Model | Task | Accuracy |
|-------|------|----------|
| LSTM (64 units) | Occupancy prediction | 91.3% |
| K-Means (k=8) | Routine clustering | Silhouette: 0.72 |
| Isolation Forest | Anomaly detection | F1: 0.88 |

## Installation

```bash
git clone https://github.com/Adham5172001/smart-home-automation.git
cd smart-home-automation
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your MQTT broker details
python main.py
```

## License

MIT License
