# 🚗 CARLA Environment: Traffic Sign Recognition & Autonomous Vehicle Control System

🌍 [English Version](README.md) | 🇨🇳 [中文](README_CN.md)
This project implements an intelligent autonomous driving simulation in the CARLA environment, focusing on real-time traffic sign detection and vehicle dynamic control. Powered by the YOLOv8 deep learning model, the system can accurately identify key traffic indicators (stop signs, speed limit signs, traffic lights), and automatically control the vehicle to make compliant driving decisions. The solution also integrates Pygame for real-time rendering of the vehicle-mounted camera view from the driver's perspective.

***

## 📁 Project Structure
- `main.py`: Main running script
- `requirements.txt`: Dependency configuration file
- `yolov8n.pt`: YOLOv8 pre-trained model
- `Images`: Saves the demonstration images, test results and working screenshots generated during the project operation
- `README.md`: Records the complete description, function introduction and usage specifications of the project

***

## 🛠 System Dependencies
Please configure the following libraries and tools in the Python runtime environment:
```bash
pip install carla pygame numpy torch ultralytics
```

### Additional Environmental Requirements
- **CARLA Simulator** (version 0.9.13 and above): Official download address: [CARLA GitHub Repository](https://github.com/carla-simulator/carla)
- **CUDA Toolkit** (optional): GPU acceleration support, which can significantly improve the speed of YOLOv8 model inference
- **Python 3.7 and above**: Stable Python interpreter version

***

## 📦 Pre-trained Model Application
### 🧠 YOLOv8n Lightweight Model
- **Model File**: `yolov8n.pt`
- **Provider**: [Ultralytics](https://github.com/ultralytics/ultralytics) official open-source pre-trained model
- **Core Function**: Input the real-time image collected by the vehicle-mounted RGB camera, and complete the detection and classification of traffic signs and traffic lights

The model loading method in the code:
```python
model = YOLO("yolov8n.pt")
```

***

## 🎮 System Functions
- Automatic lane keeping of autonomous vehicles based on path point navigation
- Real-time image acquisition and target detection based on vehicle-mounted camera
- Identify stop signs and execute full braking control
- Identify speed limit signs and dynamically adjust the vehicle's driving speed
- Identify traffic light status and execute corresponding control strategies
- Randomly generate dynamic traffic flow and traffic sign facilities in the simulation environment
- Real-time display of driver's perspective vision through Pygame visual interface
- The simulation automatically stops running after lasting for 2 minutes

***

## 🧪 System Working Principle
1. **CARLA Environment Initialization**:
   Establish a connection with the CARLA server, spawn the main control vehicle, traffic flow and traffic signs, and load the vehicle-mounted RGB camera sensor.
2. **YOLOv8 Real-time Inference**:
   The image frame collected by the camera is transmitted to the YOLOv8 model, and the target detection results including bounding box and category are output.
3. **Intelligent Vehicle Control**:
   Combine the detection results to execute speed adjustment, stop braking, traffic light compliance and other control logics.
4. **Visualization Output**:
   Use Pygame to render the driving view in real time, and output the detection information and vehicle control status in the terminal synchronously.

***

## ▶️ Operation Steps
1. **Start the CARLA Simulator First**:
   ```bash
   ./CarlaUE4.sh
   ```
2. **Run the Main Control Script**:
   ```bash
   python main.py
   ```
3. **Alternative: Use the Start Script**:
   Double-click the `start.bat` file in the project directory to automatically check the CARLA simulator status and start the system.

The system will automatically run a complete simulation process for 2 minutes and then exit safely.

***

## 🧼 Resource Release Mechanism
At the end of the simulation, the system will uniformly destroy all generated instances (vehicles, sensors, signs, etc.) to release memory resources and avoid system memory leaks.

***

## 📌 Important Reminders
- This project uses the lightweight `yolov8n.pt` model to balance speed and performance. For higher detection accuracy, you can replace it with `yolov8s.pt`, `yolov8m.pt` and other larger models.
- Please ensure that the traffic sign assets used in the code exist in the currently loaded CARLA map.
- The target detection function completely depends on the classification and recognition capability of the YOLOv8 model.
