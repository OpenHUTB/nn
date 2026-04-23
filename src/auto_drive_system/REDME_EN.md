# CARLA Autonomous Driving Basic Scenario Practice

🇨🇳 [中文版本](README.md) | 🌍 [English](README_EN.md)

> **Project** | Implementation of a Basic Autonomous Driving Scenario Based on the CARLA Simulation Platform

## Project Overview

This project is based on the CARLA simulation platform and implements a basic autonomous driving obstacle avoidance scenario. By employing a hybrid control strategy, we integrate the built-in AI's global path planning capabilities with the **Pure Pursuit** algorithm for local obstacle avoidance. The key technical aspects verified in this project include:

- CARLA sensor configuration and data acquisition
- Seamless switching between CARLA’s built-in AI and Pure Pursuit
- Interaction logic for dynamic and static obstacles
- Engineering implementation of basic path-following algorithms

## Features

🔧 **Implementation Approach**
- Hybrid control strategy: **Built-in AI for global navigation + Pure Pursuit for local obstacle avoidance**
- Supports multi-scenario testing with **dynamic and static obstacles**
- Configures multiple **multi-view cameras** around the vehicle (front/rear/left/right)

📊 **Scenario Validation**
- **>85%** success rate for static obstacle avoidance
- **>120s** average collision interval in dynamic obstacle following scenarios
- **<0.5s** control switching response time

## Project Structure

```
.
├── carla_da_dynamic.py              # Core logic for dynamic obstacle scenario
├── carla_da_dynamic_with_camera.py  # Dynamic scenario with multi-camera setup
├── carla_da_static.py               # Core logic for static obstacle scenario
├── config.yaml                      # Main configuration file (TODO)
├── docs/
│   └── design.md                    # Design insights
├── README.md                        # Project documentation
├── util/
│   ├── camera.py                    # Camera utilities
│   └── data_collector.py            # Data recording module (TODO)
=======
│   └── recorder.py                  # Data recording module (TODO)
├── videos/
│   ├── carla_a_dynamic.gif          # Dynamic obstacle avoidance demo
│   ├── carla_a_dynamic.mp4
│   ├── carla_a_dynamic_cam.gif      # Multi-view dynamic obstacle scenario
│   ├── carla_a_dynamic_cam.mp4
│   ├── carla_a_static.gif           # Static obstacle avoidance demo
│   └── carla_a_static.mp4
```

## Usage Instructions

### Environment Requirements
- **CARLA 0.9.11**
- **Python 3.7**
- **Required libraries**: `pygame`, `numpy`

### Quick Start
```bash
# Static obstacle scenario
python carla_da_static.py

# Dynamic obstacle scenario (basic version)
python carla_da_dynamic.py

# Dynamic obstacle scenario (multi-camera version)
python carla_da_dynamic_with_camera.py
```

## Scenario Demonstration

### 🚗 Static Obstacle Avoidance
![Static Obstacle Avoidance](videos/carla_a_static.gif)

### 🚗 Dynamic Obstacle Avoidance
![Dynamic Obstacle Handling](videos/carla_a_dynamic.gif)

### 🎥 Multi-View Dynamic Scenario
![Multi-View Dynamic Scenario](videos/carla_a_dynamic_cam.gif)

## Future Improvements
The project can be enhanced with the following improvements:
- **Data logging module** (`/data` directory for runtime logs)
- **Centralized configuration management** (`config.yaml` for parameter handling)
- **Simple control panel** (using `PySimpleGUI` for user interaction)

## License
**MIT License** | This project is for educational and research purposes only and does not guarantee real-world applicability.

## Acknowledgments
This project was inspired by the following resources:
- **Pure Pursuit Algorithm**: [Bilibili UP @志豪科研猿 - Video Tutorial](https://www.bilibili.com/video/BV1BQ4y167dq)
- **CARLA Camera Configuration**: [CSDN Blog: CARLA Autonomous Driving Simulation - Multi-Camera Setup](https://blog.csdn.net/zataji/article/details/134897903)

