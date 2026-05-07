<h1 align='center'>Autonomous Vehicle(AV) Self Driving System</h1>

<div align='center'>

![Nix Flake Check](https://github.com/DivitMittal/CARLA-Autonomous-Driving/actions/workflows/flake-check.yml/badge.svg)
![Nix Flake Lock Update](https://github.com/DivitMittal/CARLA-Autonomous-Driving/actions/workflows/flake-lock-update.yml/badge.svg)

</div>

<div align='center'>
    <img src='./assets/simulation_preview.png' alt='Preview of the autonomous-driving simulation' title='Simulation'/>
</div>

---

- **Enhance Perception Systems**: Utilize deep learning techniques for semantic segmentation to improve the vehicle's understanding of its surroundings.
- **Integrate Multi-Modal Sensors**: Combine data from RGBA cameras and LiDAR sensors to achieve comprehensive environmental awareness.
- **Facilitate Research and Development**: Provide a modular and extensible codebase that supports the development of novel autonomous driving algorithms.
- **Ensure Realism and Accuracy**: Leverage the high-fidelity simulation capabilities of CARLA to mimic real-world driving conditions and challenges.

## Features

- **Realistic Simulation**: Leverage the high-fidelity CARLA simulator to create diverse driving scenarios.
- **Autonomous Driving Algorithms**: Implemented state-of-the-art algorithms for perception, planning, and control.
- **Sensor Integration**: Support for various sensors including cameras, LiDAR, radar, and GPS.
- **Modular Architecture**: Easily extend and customize components to fit your research or development needs.
- **Data Collection**: Tools for collecting and analyzing simulation data for training and evaluation.
- **Visualization**: Real-time visualization of vehicle dynamics, sensor data, and decision-making processes.

## Deep Learning for Semantic Segmentation

This project leverages advanced deep learning methodologies to perform semantic segmentation, enhancing the vehicle's perception capabilities by accurately identifying and classifying various elements within the driving environment. The integration of RGBA sensor data and LiDAR ensures a robust and precise understanding of the surroundings.

### Core Components:

- **RGBA Sensors**: Capture rich color and texture information, aiding in the distinction of objects, lane markings, and environmental features.
- **LiDAR Integration**: Provides detailed depth and distance information, complementing RGBA data to achieve a comprehensive 3D perception of the environment.
- **Neural Network Architecture**: Utilizes CNNs tailored for real-time semantic segmentation, ensuring efficient and accurate processing of sensor data.
- **Training Pipeline**: Includes scripts and tools for training models on simulated data, allowing for extensive experimentation and optimization.
- **Evaluation Metrics**: Implements robust evaluation frameworks to assess the performance and reliability of the semantic segmentation models under various scenarios.

### Benefits:

- **Enhanced Accuracy**: Combining RGBA and LiDAR data improves the precision of object detection and classification.
- **Real-Time Processing**: Optimized neural network architectures ensure that semantic segmentation can be performed in real-time, essential for autonomous driving applications.
- **Scalability**: The modular design allows for easy integration of additional sensors or alternative deep learning models as needed.

### Technologies/Software/Non-Standard libraries used:

| Technology | Application                            |
| ---------- | -------------------------------------- |
| CARLA      | Vehicle Simulator                      |
| OpenCV     | For data visualization from RGB Camera |
| Keras      | For deep-learning CNN model            |
| Tensorflow | Optimizing the model weights           |
| Pygame     | Manual Control                         |

---

### TL;DR [Video Presentation](https://drive.google.com/drive/folders/1te0HDyyQaOI47RANxhuFZEUii8KMuGsd?usp=share_link)
