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
python main.py
```

### Run SAC Algorithm
```bash
python run_sac.py
```

## 📁 Project Structure

```
.
├── source/                      # Core source code
│   ├── __init__.py
│   ├── agent.py                 # ActorCar class with sensors
│   ├── carlaenv.py              # CARLA environment wrapper
│   ├── model.py                 # A2C Actor-Critic model
│   ├── sac.py                   # SAC implementation
│   ├── trainer.py               # A2C training loop
│   ├── sac_trainer.py           # SAC training loop
│   ├── replaybuffer.py          # Replay buffer implementation
│   └── utility.py               # Utility functions
├── config.yaml                  # Configuration file
├── main.py                      # A2C entry point
├── run_sac.py                   # SAC entry point
├── requirements.txt             # Python dependencies
├── test.py                      # Test scripts
├── test_env.py                  # Environment test
└── README.md                    # Project documentation
```

## ⚙️ Configuration

All training parameters can be configured in `config.yaml`:

| Parameter | Description | Default |
| :-------- | :---------- | :------ |
| `host` | CARLA server host | localhost |
| `port` | CARLA server port | 2000 |
| `car_num` | Number of NPC vehicles | 50 |
| `lr` | Learning rate | 0.001 |
| `gamma` | Discount factor | 0.99 |
| `buffer_size` | Replay buffer size | 1000 |
| `hidden_dim` | Hidden layer dimension | 1024 |
| `epoch` | Training epochs | 500 |
| `max_episode_length` | Max steps per episode | 3000 |

## 📐 Design Details

### CARLA World Settings
- Uses default CARLA town environment
- Deploys and destroys vehicles on reset instead of reloading the entire world
- Retrieves RGB camera frames in synchronous mode
- Converts frames to tensors for efficient storage in replay buffer

### Agent Configuration
- Spawns agent at a random spawn point (after NPCs)
- Equipped with `sensor.camera.rgb` and `sensor.other.collision`
- Observes visual input (640x480 RGB) and collision events
- Image preprocessing: resize to 256x256, center crop to 224x224

### Action Space

**A2C (Discrete):**

| Action Index | Action Description | Vehicle Control |
| :----------: | :----------------: | :-------------: |
|      0       |     Go Straight    | `(1, 0, 0)`     |
|      1       |      Turn Left     | `(1, -1, 0)`    |
|      2       |     Turn Right     | `(1, 1, 0)`     |
|      3       |       Brake        | `(0, 0, 1)`     |

**SAC (Continuous):**
- Steering control only: `[-1, 1]`
- Fixed throttle: `1.0`
- No brake

### Reward Function

**A2C Reward Scheme:**

| Reward | Event |
| :----: | :---- |
|  -200  | Collision detected |
|  -100  | Brake action taken |
|   +5   | Go straight action |
|   +1   | Turn left/right action |

**SAC Reward Scheme:**

| Reward | Event |
| :----: | :---- |
|  -200  | Collision detected |
|   +1   | All other actions |

### Implemented RL Algorithms
- **A2C (Advantage Actor-Critic)** - Discrete action space with ResNet50 backbone
- **SAC (Soft Actor-Critic)** - Continuous action space with automatic entropy tuning

## ✅ Progress Tracking

### Core Implementation
- [x] CARLA environment wrapper (OpenAI Gym compatible)
- [x] RGB camera and collision sensor integration
- [x] Synchronous mode simulation
- [x] Efficient world reset mechanism

### RL Components
- [x] A2C algorithm implementation
- [x] SAC algorithm implementation
- [x] Replay buffer with trajectory management
- [x] TensorBoard logging

### Testing & Debugging
- [x] Environment test scripts
- [x] Connection test
- [x] Replay buffer test

### Future Work
- [ ] Code refactoring and optimization
- [ ] Performance improvement
- [ ] Advanced reward engineering
- [ ] Multi-agent scenarios

## 🧠 Model Architecture

### A2C Actor-Critic
```
Input (224x224 RGB)
    ↓
ResNet50 (pretrained, frozen)
    ↓
Fully Connected Layers (3 layers, 1024 hidden dim)
    ↓
┌─────────────┬─────────────┐
│  Actor Head │ Critic Head │
├─────────────┼─────────────┤
│ Softmax     │ Linear(1)   │
│ (4 actions) │ (V-value)   │
└─────────────┴─────────────┘
```

### SAC Networks
- **Value Network**: ResNet50 + FC layers → scalar output
- **Soft Q Network**: ResNet50 + FC layers (with action input) → scalar Q-value
- **Policy Network**: ResNet50 + FC layers → Gaussian distribution parameters

## 💡 Training Tips

### Hardware Considerations
To run on limited computational resources (e.g., 1 RTX 3060):
- Use online A2C (sample one episode then update)
- Directly resize and crop frames upon reception
- Store data in Tensor type to save memory
- Test with smaller episode lengths initially

### Monitoring Training
```bash
# Launch TensorBoard
tensorboard --logdir=./log/
```

### Model Checkpoints
Checkpoints are saved automatically when average episode frames improve:
- A2C: `checkpoints/a2c/model{epoch}.pt`
- SAC: `checkpoints/sac/{network}{epoch}.pt`

## 📝 License
This project is for educational purposes as part of the reinforcement learning course.

## 🤝 Acknowledgments
- [CARLA Simulator](https://carla.org/) for providing the simulation environment
- OpenAI Gym for the environment interface standard