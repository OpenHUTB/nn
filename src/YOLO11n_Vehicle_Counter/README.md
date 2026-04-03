# YOLO-Based Real-Time Vehicle Counting and Annotation using Supervision

This repository contains a suite of Python scripts dedicated to real-time vehicle detection, tracking, and counting, leveraging the YOLO (You Only Look Once) object detection model, specifically YOLOv8. Designed to facilitate vehicle counting across multi-lane highways, this project integrates advanced tools like Supervision for seamless annotation, object tracking, and detection smoothing, creating a robust vehicle monitoring system.

### Key Components:
- **YOLOv8 Model**: Used for high-accuracy, real-time vehicle detection. YOLOv8 is particularly effective in distinguishing various vehicle types, including cars, trucks, buses, and motorbikes, which makes it highly suitable for traffic monitoring.
- **Supervision Library**: This library is instrumental in annotating frames, visualizing bounding boxes, tracking objects, and implementing various other utilities such as line drawing and overlay creation. It also supports **ByteTrack** for high-precision object tracking and **DetectionsSmoother** for enhanced tracking stability.
- **Flexible Configurations**: Each script is tailored for specific test cases, offering different boundary conditions, lane partitioning, and counting methods. This modular approach allows users to adapt the scripts to various traffic environments or project needs.
  
### Workflow Overview:
1. **Object Detection**: The YOLOv8 model is used to detect vehicles in video frames, outputting bounding boxes and classifying detected objects.
2. **Object Tracking with ByteTrack**: Supervision’s ByteTrack algorithm tracks vehicles across frames, ensuring that each vehicle is counted only once as it crosses predefined boundaries.
3. **Annotation and Visualization**: The Supervision library annotates frames with bounding boxes, labels, and trace lines, making it easy to visualize the counting process and object movement.
4. **Count and Record**: Vehicles are counted based on crossing specific boundaries, and separate counters can be maintained for different lanes or vehicle types.

Below is an overview of each script contained in this repository, detailing their unique functionalities and configurations for diverse traffic monitoring scenarios:

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Script Descriptions](#script-descriptions)
  - [car_counter_2](#car_counter_2)
  - [car_counter_3](#car_counter_3)
  - [car_counter_4](#car_counter_4)
  - [car_counter_5](#car_counter_5)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Pushtogithub23/Real-Time-YOLO-Car-Counter.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use any of the car counter scripts, ensure that the YOLOv8 weights are downloaded and correctly linked in the scripts. Run the desired script using:
```bash
python <script_name.py>
```

## Script Descriptions

### car_counter_2
This script uses the YOLOv8 model to detect vehicles in a video and count how many cross a predefined boundary line. Key features include:
- Integration of the YOLOv8 model for object detection.
- Counting vehicles crossing a line placed at a fixed height within the video frame.
- Displays vehicle counts in real-time as the video is processed.

![car_counter_2](https://github.com/user-attachments/assets/08c04d28-efb5-4352-a76c-bda0eeb81cd2)

### car_counter_3
Similar to `car_counter_2`, this version includes additional vehicle detection and counting logic improvements. It also introduces optimized annotation tools, like Elliptical annotations and labels for vehicle tracking.

![car_counter_3](https://github.com/user-attachments/assets/b69bae96-00f7-41f9-9dda-f29ce589024e)

### car_counter_4
This script builds upon the previous ones by adding a multi-lane vehicle counting system. It counts vehicles in two directions (up and down) separately:
- Different counting lines are defined for upward and downward traffic flows.
- Real-time display of counts for total vehicles and those going up and down.

![car_counter_4](https://github.com/user-attachments/assets/df474661-8748-44ef-a7be-c1b7f1010640)

### car_counter_5
This version is an advanced iteration with further optimizations for smoother detections. It introduces:
- Smoothing of detection results to avoid abrupt changes in vehicle positions.
- Counting vehicles in both directions of traffic flow
- Customized visual enhancements like semi-transparent overlays and trace annotations to clearly show vehicle movement paths.

![car_counter_yolo11x](https://github.com/user-attachments/assets/60119c3f-04b4-46d7-a43a-598cb964d7e9)

---
