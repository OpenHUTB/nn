#!/usr/bin/env python3
"""
Example script demonstrating how to use the DataRecorder class
for autonomous driving dataset collection.

This example shows how to:
1. Initialize the DataRecorder
2. Start/stop recording
3. Update data sources
4. Monitor recording status
"""

import time
import numpy as np
import cv2
import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.DataRecorder import DataRecorder

def create_sample_rgb_image(width=400, height=224):
    """Create a sample RGB image for testing."""
    # Create a simple test image with gradient
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient
    for i in range(height):
        for j in range(width):
            image[i, j] = [int(255 * j / width), int(255 * i / height), 128]
    
    # Add some text
    cv2.putText(image, "Sample Frame", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

def simulate_vehicle_data():
    """Simulate vehicle control and state data."""
    # Simulate control signals
    steer = np.sin(time.time() * 0.5) * 0.3  # Oscillating steering
    throttle = 0.5 + 0.3 * np.sin(time.time() * 0.2)  # Varying throttle
    brake = max(0, -np.sin(time.time() * 0.3))  # Occasional braking
    
    # Simulate vehicle speed (km/h)
    speed = 30 + 10 * np.sin(time.time() * 0.1)
    
    # Simulate vehicle transform
    transform = {
        'location': {
            'x': 100 + 50 * np.sin(time.time() * 0.1),
            'y': 200 + 30 * np.cos(time.time() * 0.1),
            'z': 0.5
        },
        'rotation': {
            'pitch': 0.0,
            'yaw': 45 + 10 * np.sin(time.time() * 0.2),
            'roll': 0.0
        }
    }
    
    return steer, throttle, brake, speed, transform

def main():
    """Main example function."""
    print("DataRecorder Example")
    print("=" * 40)
    
    # Initialize the data recorder
    recorder = DataRecorder(
        output_dir="example_dataset",
        sampling_rate=8.0,  # 8 Hz sampling rate
        image_size=(400, 224),
        enable_recording=False  # Start disabled
    )
    
    print(f"DataRecorder initialized:")
    print(f"  - Output directory: {recorder.output_dir}")
    print(f"  - Sampling rate: {recorder.sampling_rate} Hz")
    print(f"  - Image size: {recorder.image_size}")
    print(f"  - Session directory: {recorder.session_dir}")
    
    # Start recording
    print("\nStarting recording...")
    recorder.start_recording()
    
    # Simulate data collection for 10 seconds
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 10.0:
            # Create sample RGB image
            rgb_image = create_sample_rgb_image()
            
            # Get simulated vehicle data
            steer, throttle, brake, speed, transform = simulate_vehicle_data()
            
            # Update recorder with current data
            recorder.update_rgb_image(rgb_image)
            recorder.update_control_signals(steer, throttle, brake)
            recorder.current_data['vehicle_speed'] = speed
            recorder.current_data['vehicle_transform'] = transform
            
            # Record frame if conditions are met
            recorder.record_frame()
            
            # Print status every 2 seconds
            if frame_count % 16 == 0:  # Approximately every 2 seconds at 8 Hz
                status = recorder.get_recording_status()
                print(f"Recording... Frames: {status['frame_count']}, "
                      f"Duration: {status['session_duration']:.1f}s, "
                      f"Queue: {status['queue_size']}")
            
            frame_count += 1
            time.sleep(0.01)  # Small delay to prevent overwhelming the system
            
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    
    # Stop recording
    print("\nStopping recording...")
    recorder.stop_recording()
    
    # Show final status
    final_status = recorder.get_recording_status()
    print(f"\nFinal Status:")
    print(f"  - Total frames recorded: {final_status['frame_count']}")
    print(f"  - Session duration: {final_status['session_duration']:.1f} seconds")
    print(f"  - Average frame rate: {final_status['frame_count'] / final_status['session_duration']:.1f} Hz")
    print(f"  - Data saved to: {final_status['session_directory']}")
    
    # Clean up
    recorder.cleanup()
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
