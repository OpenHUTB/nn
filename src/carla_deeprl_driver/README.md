# CARLA Deep Reinforcement Learning Driver

This repository implements a reinforcement learning-based autonomous vehicle control system using the [`CARLA Simulator (version 0.9.13)`](https://carla.org/). The project aims to train an intelligent agent that can navigate through complex urban environments using deep reinforcement learning algorithms.

## 📋 Project Overview

The goal of this project is to develop a robust autonomous driving agent that can:
- Navigate through CARLA's urban environments
- Avoid collisions with other vehicles and obstacles
- Make intelligent driving decisions based on visual input
- Learn optimal driving policies through reinforcement learning

## 🛠️ Environment Setup

### Prerequisites
1. **CARLA Simulator 0.9.13** - Download from official or mirror sources
2. **Python 3.6+** - Required for CARLA Python API compatibility
3. **conda** - Recommended for environment management

### Installation Steps

#### Step 1: Download CARLA Simulator
```bash
# Option 1: Download from SUSTech mirror (recommended for China)
wget https://mirrors.sustech.edu.cn/carla/carla/0.9.13/CARLA_0.9.13.tar.gz
tar -zxvf CARLA_0.9.13.tar.gz

# Option 2: Download from official website
# https://github.com/carla-simulator/carla/releases/tag/0.9.13
```

#### Step 2: Set Up Python Environment
```bash
# Create conda environment
conda env create -f environment.yml
conda activate carla-rl

# Install dependencies manually (if needed)
pip install -r requirements.txt
```

#### Step 3: Launch CARLA Server
```bash
# Navigate to CARLA directory
cd CARLA_0.9.13

# Start CARLA server in off-screen mode (recommended for training)
./CarlaUE4.sh -RenderOffScreen

# Or start with visualization (for testing/demonstration)
# ./CarlaUE4.sh
```

## 🚀 Quick Start

### Run A2C Algorithm
```bash
python run.py
```

### Run SAC Algorithm
```bash
python run_sac.py
```

## 📐 Design Details

### CARLA World Settings
- Uses default CARLA town environment
- Deploys and destroys vehicles on reset instead of reloading the entire world
- Retrieves RGB camera frames in synchronous mode
- Converts frames to tensors for efficient storage in replay buffer

### Agent Configuration
- Spawns agent at the first spawn point
- Equipped with `sensor.camera.rgb` and `sensor.other.collision`
- Observes visual input and collision events

### Action Space

| Action Index | Action Description | Vehicle Control |
| :----------: | :----------------: | :-------------: |
|      0       |     Go Straight    | `(1, 0, 0)`     |
|      1       |      Turn Left     | `(1, -1, 0)`    |
|      2       |     Turn Right     | `(1, 1, 0)`     |
|      3       |       Brake        | `(0, 0, 1)`     |

*Note: SAC uses continuous action space for steering control*

### Reward Function

**A2C Reward Scheme:**

| Reward | Event |
| :----: | :---- |
|  -200  | Collision detected |
|  -100  | Brake action taken |
|   +2   | Go straight action |
|   +1   | Turn left/right action |

**SAC Reward Scheme:**

| Reward | Event |
| :----: | :---- |
|  -200  | Collision detected |
|   +1   | All other actions |

### Implemented RL Algorithms
- **A2C (Advantage Actor-Critic)** - Discrete action space
- **SAC (Soft Actor-Critic)** - Continuous action space

## ✅ Progress Tracking

### Core Implementation
- [x] CARLA environment wrapper (OpenAI Gym compatible)
  - [x] `env()` - Initialize CARLA world
  - [x] `step()` - Execute action and return observation
  - [x] `reset()` - Reset environment to initial state
  - [x] Agent (actor) management

### Technical Solutions
- [x] Sensor management (stop sensors before destruction)
- [x] Efficient world reset using actor destruction instead of `reload_world()`
- [x] Collision-based episode termination

### RL Components
- [x] Trajectory sampling
- [x] Replay buffer implementation
- [x] A2C algorithm
- [x] SAC algorithm

### Future Work
- [ ] Code refactoring and optimization
- [ ] Performance improvement
- [ ] Advanced reward engineering
- [ ] Multi-agent scenarios

## 💡 Notes

### Hardware Considerations
To run on limited computational resources (e.g., 1 RTX 3060):
- Implemented online A2C (sample one episode then update)
- Directly resize and crop frames upon reception
- Store data in Tensor type to save memory
- Tested with small episode lengths

### Reward Design Insights
- Braking frequently is penalized to encourage smooth driving
- Positive reward for braking leads to undesirable behavior
- Current reward scheme balances exploration and exploitation

### SAC Specifics
SAC uses continuous action space for steering control:
- Steering range: `[-1, 1]`
- Action format: `carla.VehicleControl(1, steer, 0)`
- Policy outputs continuous steering value

## 📁 Project Structure
```
.
├── agent.py          # RL agent implementation
├── carlaenv.py       # CARLA environment wrapper
├── utility.py        # Utility functions (action mapping)
├── run.py            # A2C training script
├── run_sac.py        # SAC training script
├── requirements.txt  # Python dependencies
├── environment.yml   # Conda environment configuration
└── README.md         # Project documentation
```

## 📝 License
This project is for educational purposes as part of the reinforcement learning course.

## 🤝 Acknowledgments
- [CARLA Simulator](https://carla.org/) for providing the simulation environment
- OpenAI Gym for the environment interface standard