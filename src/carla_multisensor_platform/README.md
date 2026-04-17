# 🏎️ CARLA Multi-Sensor Simulation Platform

An autonomous driving simulation platform built on top of CARLA Simulator. This project provides multi-sensor data collection, real-time visualization, lane detection, and dataset recording capabilities for autonomous driving research.

## 📌 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Sensors](#sensors)
- [Data Recording](#data-recording)
- [Controls](#controls)
- [License](#license)

## ✨ Features

- **Multi-Sensor Support**: RGB Camera, Depth Camera, LiDAR, Semantic Segmentation, Instance Segmentation
- **Real-time Lane Detection**: YOLOPv2-based lane and drivable area detection with GPU acceleration
- **Data Recording System**: Synchronized recording of images, control signals, and vehicle state at configurable sampling rates (5-10 Hz)
- **Eagle Eye Map**: Bird's-eye view visualization of the simulation environment
- **Traffic Simulation**: Automated spawning of vehicles and pedestrians
- **Weather Control**: Configurable weather conditions
- **Keyboard Control**: Manual vehicle control support
- **Synchronous Mode**: Deterministic simulation for consistent data collection

## 📁 Project Structure

```
├── Main.py                 # Main simulation entry point
├── requirements.txt        # Python dependencies
├── Sensors/
│   ├── SensorManager.py    # Sensor initialization and management
│   ├── SensorHandler.py    # Sensor data processing callbacks
│   ├── RGBcamera/
│   │   ├── YOLOPv2Detecor.py   # Lane detection with YOLOPv2
│   │   └── CarLaneDetector.py  # Car and lane detection
│   └── Lidar/
│       └── lidar.py        # LiDAR point cloud processing
├── utils/
│   ├── environment.py      # CARLA environment setup
│   ├── DisplayManager.py   # Pygame display management
│   ├── DataRecorder.py     # Dataset recording system
│   ├── EgoVehicleController.py # Vehicle control
│   ├── eagle_eye_map.py    # Bird's-eye view map
│   └── weather.py          # Weather control
├── dataset/                # Recorded driving data
└── docs/                   # Documentation
```

## 📋 Requirements

- CARLA Simulator 0.9.15
- Python 3.8+
- CUDA-capable GPU (recommended)

### Python Dependencies
```
numpy==1.21.6
opencv-python==4.5.5.64
pygame==2.1.2
transforms3d==0.4.1
colorama==0.4.6
carla==0.9.15
```

## 🚀 Installation

1. **Install CARLA Simulator**
   - Download CARLA 0.9.15 from the [official releases](https://github.com/carla-simulator/carla/releases)

2. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Carla-Project
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**
   - Place YOLOPv2 model (`yolopv2.pt`) in `Sensors/RGBcamera/model/pretrained/`

## 🎮 Usage

1. **Start CARLA Server**
   ```bash
   # Windows
   CarlaUE4.exe
   
   # Linux
   ./CarlaUE4.sh
   ```

2. **Run the simulation**
   ```bash
   python Main.py
   ```

## 📷 Sensors

| Sensor | Description | Grid Position |
|--------|-------------|---------------|
| RGB Camera | Front-facing camera | Configurable |
| RGB Camera BEV | Bird's-eye view camera | Configurable |
| RGB Camera Lane | Lane detection camera | Configurable |
| Depth Camera | Depth perception | Configurable |
| Semantic Segmentation | Per-pixel class labels | Configurable |
| Instance Segmentation | Per-pixel instance labels | Configurable |
| LiDAR | 3D point cloud | Configurable |
| Semantic LiDAR | Labeled point cloud | Configurable |

Configure sensors in `Main.py`:
```python
sensors_dict = {
    'RGBCamera': [[0, 0, 2.4], [0, 1], True],  # [position, grid, enabled]
    'LiDAR': [[0, 0, 2.4], [1, 0], False],
    # ...
}
```

## 💾 Data Recording

The system records synchronized autonomous driving data:

- **Images**: RGB camera frames (400×224 JPG)
- **Control Signals**: Steering, throttle, brake
- **Vehicle State**: Speed, position, rotation
- **Timestamps**: Frame IDs and timing data

### Output Format
```
dataset/
└── session_YYYYMMDD_HHMMSS/
    ├── images/
    │   └── frame_XXXXXX.jpg
    ├── metadata/
    │   └── frame_XXXXXX.json
    └── session_summary.json
```

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `R` | Toggle data recording |
| `S` | Show recording status |
| `ESC` | Exit simulation |
| `W/↑` | Accelerate |
| `S/↓` | Brake |
| `A/←` | Steer left |
| `D/→` | Steer right |
| `SPACE` | Hand brake |

## 📝 License

This project is for research and educational purposes.