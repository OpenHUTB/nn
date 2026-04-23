# Autonomous Driving Data Recording System

This document describes the data recording system for collecting autonomous driving datasets in CARLA.

## Overview

The `DataRecorder` class provides a comprehensive solution for recording synchronized data from multiple sources during autonomous driving simulations. It captures RGB camera images, control signals, vehicle state, and timestamps at configurable sampling rates.

## Features

- **RGB Camera Recording**: Captures front-facing camera images at 400x224 resolution
- **Control Signals**: Records steering, throttle, and brake inputs
- **Vehicle State**: Captures speed and transform (location + rotation) data
- **Synchronization**: Timestamps and frame IDs for data alignment
- **Configurable Sampling**: 5-10 Hz sampling rate control
- **Thread-Safe**: Background recording with queue-based data handling
- **Easy Control**: Simple start/stop/toggle functionality

## Data Format

### Directory Structure
```
dataset/
└── session_YYYYMMDD_HHMMSS/
    ├── images/
    │   ├── frame_000000.jpg
    │   ├── frame_000001.jpg
    │   └── ...
    ├── metadata/
    │   ├── frame_000000.json
    │   ├── frame_000001.json
    │   └── ...
    └── session_summary.json
```

### Image Data
- **Format**: JPG images
- **Resolution**: 400x224 pixels (configurable)
- **Color Space**: RGB
- **Naming**: `frame_XXXXXX.jpg` (6-digit zero-padded frame numbers)

### Metadata Format
Each frame has a corresponding JSON metadata file:

```json
{
  "frame_id": 0,
  "timestamp": 1703123456.789,
  "image_filename": "frame_000000.jpg",
  "control_signals": {
    "steer": 0.15,
    "throttle": 0.7,
    "brake": 0.0
  },
  "vehicle_speed": 45.2,
  "vehicle_transform": {
    "location": {
      "x": 123.45,
      "y": 456.78,
      "z": 0.5
    },
    "rotation": {
      "pitch": 0.0,
      "yaw": 12.3,
      "roll": 0.0
    }
  }
}
```

## Usage

### Basic Integration

```python
from utils.DataRecorder import DataRecorder

# Initialize recorder
recorder = DataRecorder(
    output_dir="dataset",
    sampling_rate=10.0,  # 10 Hz
    image_size=(400, 224),
    enable_recording=False
)

# Start recording
recorder.start_recording()

# Update data sources
recorder.update_rgb_image(rgb_image)
recorder.update_control_signals(steer, throttle, brake)
recorder.update_vehicle_state(ego_vehicle)

# Record frame (called in main loop)
recorder.record_frame()

# Stop recording
recorder.stop_recording()
```

### Keyboard Controls (in Main.py)

- **R**: Toggle recording ON/OFF
- **S**: Show recording status
- **ESC**: Exit simulation

### Status Monitoring

```python
status = recorder.get_recording_status()
print(f"Recording: {status['is_recording']}")
print(f"Frames: {status['frame_count']}")
print(f"Duration: {status['session_duration']:.1f}s")
print(f"Queue size: {status['queue_size']}")
```

## Configuration Options

### DataRecorder Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | "dataset" | Base directory for data storage |
| `sampling_rate` | float | 10.0 | Recording frequency (5-10 Hz) |
| `image_size` | tuple | (400, 224) | RGB image dimensions (width, height) |
| `enable_recording` | bool | False | Start recording immediately |

### Sampling Rate Guidelines

- **5 Hz**: Lower computational load, suitable for basic training
- **10 Hz**: Higher fidelity, recommended for detailed analysis
- **Custom**: Can be adjusted between 5-10 Hz based on requirements

## Integration with CARLA

The system integrates seamlessly with the existing CARLA simulation:

1. **Sensor Integration**: RGB camera data is automatically fed to the recorder
2. **Vehicle Control**: Control signals are captured from the ego vehicle controller
3. **State Monitoring**: Vehicle speed and transform are updated each frame
4. **Synchronization**: All data is timestamped and frame-synchronized

## Performance Considerations

- **Thread-Safe Design**: Recording happens in background thread
- **Queue Management**: Prevents blocking of main simulation loop
- **Memory Efficient**: Images are processed and saved immediately
- **Error Handling**: Robust error handling prevents data loss

## Data Quality

### Synchronization
- All data sources are timestamped with the same reference
- Frame IDs provide sequential ordering
- Sampling rate ensures consistent temporal spacing

### Validation
- Image format validation (JPG compression)
- Control signal range checking (-1.0 to 1.0)
- Vehicle state validation (speed, transform bounds)

## Training Dataset Preparation

The recorded data is ready for autonomous driving model training:

1. **Image Preprocessing**: Images are already resized to target dimensions
2. **Label Format**: Control signals serve as training labels
3. **Temporal Consistency**: Frame sequences maintain temporal relationships
4. **Metadata**: Rich metadata enables advanced training techniques

## Example Usage

See `examples/data_recording_example.py` for a complete standalone example demonstrating:
- Recorder initialization
- Data simulation
- Recording control
- Status monitoring
- Cleanup procedures

## Troubleshooting

### Common Issues

1. **Recording Not Starting**: Check if RGB camera is enabled in sensor configuration
2. **Low Frame Rate**: Reduce sampling rate or check system performance
3. **Disk Space**: Monitor output directory size during long recordings
4. **Memory Issues**: Ensure adequate RAM for image processing queue

### Performance Tips

- Use SSD storage for better I/O performance
- Close unnecessary applications during recording
- Monitor system resources during long sessions
- Consider reducing image resolution for faster processing

## Future Enhancements

Potential improvements for the data recording system:

- **Multi-Camera Support**: Record from multiple camera angles
- **LIDAR Integration**: Include point cloud data
- **Semantic Segmentation**: Add segmentation masks
- **Real-time Visualization**: Live recording status display
- **Data Compression**: Advanced compression for storage efficiency
- **Cloud Upload**: Automatic backup to cloud storage
