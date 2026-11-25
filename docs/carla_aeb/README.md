# carla_aeb

CARLA simulator auto drive test module.

A simple auto drive system that spawns a vehicle in CARLA simulator and enables autopilot mode for autonomous driving in the city.

## Features

- Spawn vehicle at available spawn points
- Enable autopilot for autonomous driving
- Real-time display of vehicle speed and location
- Automatic cleanup on exit

## Directory Structure

```
src/carla_aeb/
  main.py          - Main entry file
  requirements.txt - Python dependencies
docs/carla_aeb/
  README.md        - Documentation
```

## Environment

- Python 3.10+
- CARLA Simulator 0.9.16+
- hutb (CARLA Python API)

## Run

1. Start CARLA server
   ```
   D:\python-cla\hutb\CarlaUE4.exe
   ```

2. Run the module
   ```powershell
   cd D:\Python_project\nn-main
   .venv_carla\Scripts\Activate.ps1
   python src/carla_aeb/main.py
   ```

3. Press Ctrl+C to stop

## Test Scenario

- Spawns a cyan Tesla Model 3
- Enables autopilot mode
- Vehicle drives autonomously in the city
- Real-time speed and location displayed

## Source

- [Source Code](https://github.com/OpenHUTB/nn/tree/main/src/carla_aeb)
