# CARLA Development Notes

Technical notes and best practices for CARLA autonomous driving simulation development.

---

## Table of Contents
- [Simulation Modes](#simulation-modes)
- [Time-Step Configuration](#time-step-configuration)
- [Sensor Setup](#sensor-setup)
- [Performance Tips](#performance-tips)

---

## Simulation Modes

### Synchronous vs Asynchronous Mode

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| Control | Script controls simulation with `world.tick()` | Simulation runs freely |
| Sensor Data | All sensors synchronized | Data may have timing jitter |
| Determinism | Fully deterministic | Non-deterministic |
| Use Case | Dataset generation, sensor fusion | Real-time demos, quick testing |

#### Synchronous Mode

```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)

# Main loop
while running:
    world.tick()  # Advance simulation
```

**Best for:**
- ✅ Machine learning dataset collection
- ✅ Sensor fusion (LiDAR + RGB + Depth)
- ✅ Reproducible experiments
- ✅ Timing-critical analysis

#### Asynchronous Mode

```python
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)

# React to simulation events
world.wait_for_tick()
```

**Best for:**
- ✅ Real-time visualization demos
- ✅ Low-latency control testing
- ✅ Quick prototyping

---

## Time-Step Configuration

### Variable Time-Step (Default)

Simulation time is calculated by the server based on rendering speed.

```python
settings = world.get_settings()
settings.fixed_delta_seconds = None
world.apply_settings(settings)
```

### Fixed Time-Step with Physics Substeps

For precise physics simulation, use fixed time-steps with substepping.

**Rule:** `fixed_delta_seconds <= max_substep_delta_time × max_substeps`

```python
settings = world.get_settings()
settings.fixed_delta_seconds = 0.05    # 20 FPS simulation
settings.substepping = True
settings.max_substep_delta_time = 0.01  # 100 Hz physics
settings.max_substeps = 10
world.apply_settings(settings)
```

> 💡 **Tip:** For optimal physics accuracy, keep `max_substep_delta_time` below **0.01666** (60 Hz), ideally below **0.01** (100 Hz).

### Common Configurations

| Use Case | fixed_delta | substep_delta | max_substeps | Effective FPS |
|----------|-------------|---------------|--------------|---------------|
| Dataset Collection | 0.1 | 0.01 | 10 | 10 Hz |
| Real-time Demo | 0.033 | 0.01 | 10 | 30 Hz |
| High-Precision | 0.05 | 0.005 | 10 | 20 Hz |

---

## Sensor Setup

### Sensor Attachment

```python
# Attach sensor to ego vehicle
transform = carla.Transform(carla.Location(x=2.0, z=1.4))
sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
```

### Common Sensor Positions

| Sensor | Position (x, y, z) | Notes |
|--------|-------------------|-------|
| Front Camera | (2.0, 0, 1.4) | Hood-mounted view |
| BEV Camera | (0, 0, 20.0) | Top-down view |
| LiDAR | (0, 0, 2.4) | Roof-mounted |
| Rear Camera | (-2.0, 0, 1.4) | Backup camera |

---

## Performance Tips

1. **Enable `no_rendering_mode`** when collecting sensor data only:
   ```python
   settings.no_rendering_mode = True
   ```

2. **Limit active distance** to reduce actor computations:
   ```python
   settings.actor_active_distance = 2000
   ```

3. **Use Traffic Manager** for NPC vehicles:
   ```python
   traffic_manager = client.get_trafficmanager(8000)
   traffic_manager.set_synchronous_mode(True)
   ```

4. **Batch spawn actors** instead of spawning one by one.

5. **Destroy sensors** when done to free resources:
   ```python
   sensor.destroy()
   ```

---

## References

- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA Python API Reference](https://carla.readthedocs.io/en/latest/python_api/)