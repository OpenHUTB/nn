import os
import json
import time
import threading
import queue
from copy import deepcopy
from pathlib import Path
import numpy as np
import cv2
import carla
from datetime import datetime
from typing import Dict, Any, Tuple
import logging

class DataRecorder:
    """
    Data recording system for autonomous driving dataset collection.
    Records RGB camera images, control signals, vehicle state, and timestamps
    at configurable sampling rates (5-10 Hz).
    """
    
    def __init__(self, 
                 output_dir: str = "dataset",
                 sampling_rate: float = 10.0,
                 image_size: Tuple[int, int] = (400, 224),
                 enable_recording: bool = False):
        """
        Initialize the data recorder.
        
        Args:
            output_dir: Directory to save recorded data
            sampling_rate: Recording frequency in Hz (5-10 Hz recommended)
            image_size: RGB camera image dimensions (width, height)
            enable_recording: Whether to start recording immediately
        """
        self.output_dir = Path(output_dir)
        self.sampling_rate = max(5.0, min(10.0, sampling_rate))  # Clamp between 5-10 Hz
        self.image_size = image_size
        self.enable_recording = enable_recording
        
        # Recording state
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        self.last_record_time = 0.0
        
        # Data queues for thread-safe recording
        self.data_queue = queue.Queue(maxsize=100)
        self.recording_thread = None
        self.stop_recording_event = threading.Event()
        
        # Current data cache
        self.current_data = {
            'rgb_image': None,
            'control_signals': {'steer': 0.0, 'throttle': 0.0, 'brake': 0.0},
            'vehicle_speed': 0.0,
            'vehicle_transform': None,
            'timestamp': 0.0,
            'frame_id': 0
        }
        
        # Thread lock for data access
        self.data_lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for data storage."""
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session directory with timestamp
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{session_name}"
        self.session_images_dir = self.session_dir / "images"
        self.session_metadata_dir = self.session_dir / "metadata"
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_images_dir.mkdir(parents=True, exist_ok=True)
        self.session_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Data recorder initialized. Session: {session_name}")
        
    def start_recording(self):
        """Start the data recording process."""
        if self.is_recording:
            self.logger.warning("Recording is already active")
            return
            
        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0
        self.last_record_time = 0.0
        self.stop_recording_event.clear()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.recording_thread.start()
        
        self.logger.info(f"Started recording at {self.sampling_rate} Hz")
        
    def stop_recording(self):
        """Stop the data recording process."""
        if not self.is_recording:
            self.logger.warning("Recording is not active")
            return
            
        self.is_recording = False
        self.stop_recording_event.set()
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.data_queue.join()
            self.recording_thread.join(timeout=5.0)
            
        # Save session summary
        self._save_session_summary()
        
        self.logger.info(f"Stopped recording. Total frames: {self.frame_count}")
        
    def update_rgb_image(self, image: np.ndarray):
        """Update the current RGB image data."""
        if not self.is_recording:
            return
            
        # Resize image to target size if needed
        if image.shape[:2] != (self.image_size[1], self.image_size[0]):
            image = cv2.resize(image, self.image_size)
            
        with self.data_lock:
            self.current_data['rgb_image'] = image.copy()
            
    def update_control_signals(self, steer: float, throttle: float, brake: float):
        """Update the current control signals."""
        if not self.is_recording:
            return
            
        with self.data_lock:
            self.current_data['control_signals'] = {
                'steer': float(steer),
                'throttle': float(throttle),
                'brake': float(brake)
            }
            
    def update_vehicle_state(self, ego_vehicle: carla.Vehicle):
        """Update vehicle speed and transform data."""
        if not self.is_recording or ego_vehicle is None:
            return
            
        try:
            # Get vehicle velocity and calculate speed
            velocity = ego_vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h
            
            # Get vehicle transform
            transform = ego_vehicle.get_transform()
            
            with self.data_lock:
                self.current_data['vehicle_speed'] = float(speed)
                self.current_data['vehicle_transform'] = {
                    'location': {
                        'x': float(transform.location.x),
                        'y': float(transform.location.y),
                        'z': float(transform.location.z)
                    },
                    'rotation': {
                        'pitch': float(transform.rotation.pitch),
                        'yaw': float(transform.rotation.yaw),
                        'roll': float(transform.rotation.roll)
                    }
                }
        except Exception as e:
            self.logger.error(f"Error updating vehicle state: {e}")
            
    def should_record_frame(self) -> bool:
        """Check if enough time has passed to record the next frame."""
        if not self.is_recording:
            return False
            
        current_time = time.time()
        time_since_last = current_time - self.last_record_time
        min_interval = 1.0 / self.sampling_rate
        
        return time_since_last >= min_interval
        
    def record_frame(self):
        """Record current frame data if conditions are met."""
        if not self.should_record_frame():
            return
            
        with self.data_lock:
            # Check if we have all required data
            if self.current_data['rgb_image'] is None:
                return
                
            # Update timestamp and frame ID
            current_time = time.time()
            self.current_data['timestamp'] = current_time
            self.current_data['frame_id'] = self.frame_count
            
            # Create a copy of current data for recording
            frame_data = {
                'rgb_image': self.current_data['rgb_image'].copy(),
                'control_signals': self.current_data['control_signals'].copy(),
                'vehicle_speed': self.current_data['vehicle_speed'],
                'vehicle_transform': deepcopy(self.current_data['vehicle_transform']),
                'timestamp': self.current_data['timestamp'],
                'frame_id': self.current_data['frame_id']
            }
            
        # Add to recording queue
        try:
            self.data_queue.put_nowait(frame_data)
        except queue.Full:
            self.logger.warning("Recording queue is full, dropping frame")
            
        self.last_record_time = current_time
        self.frame_count += 1
        
    def _recording_worker(self):
        """Worker thread for saving recorded data."""
        while not self.stop_recording_event.is_set() or not self.data_queue.empty():
            frame_data = None
            try:
                # Get data from queue with timeout
                frame_data = self.data_queue.get(timeout=1.0)
                self._save_frame_data(frame_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in recording worker: {e}")
            finally:
                if frame_data is not None:
                    self.data_queue.task_done()
                    
    def _save_frame_data(self, frame_data: Dict[str, Any]):
        """Save individual frame data to disk."""
        try:
            frame_id = frame_data['frame_id']
            
            # Save RGB image
            image_filename = f"frame_{frame_id:06d}.jpg"
            image_path = self.session_images_dir / image_filename
            if not cv2.imwrite(str(image_path), frame_data['rgb_image']):
                raise IOError(f"Failed to save image to {image_path}")
            
            # Prepare metadata
            metadata = {
                'frame_id': frame_id,
                'timestamp': frame_data['timestamp'],
                'image_filename': image_filename,
                'control_signals': frame_data['control_signals'],
                'vehicle_speed': frame_data['vehicle_speed'],
                'vehicle_transform': frame_data['vehicle_transform']
            }
            
            # Save metadata
            metadata_filename = f"frame_{frame_id:06d}.json"
            metadata_path = self.session_metadata_dir / metadata_filename
            
            with metadata_path.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving frame data: {e}")
            
    def _save_session_summary(self):
        """Save session summary and statistics."""
        try:
            session_duration = time.time() - self.start_time if self.start_time else 0
            
            summary = {
                'session_info': {
                    'start_time': self.start_time,
                    'duration_seconds': session_duration,
                    'total_frames': self.frame_count,
                    'sampling_rate': self.sampling_rate,
                    'image_size': self.image_size
                },
                'recording_settings': {
                    'output_directory': str(self.session_dir),
                    'images_directory': str(self.session_images_dir),
                    'metadata_directory': str(self.session_metadata_dir)
                },
                'data_format': {
                    'rgb_image': 'JPG format, resized to specified dimensions',
                    'control_signals': 'steer, throttle, brake (float values)',
                    'vehicle_speed': 'speed in km/h (float)',
                    'vehicle_transform': 'location (x,y,z) and rotation (pitch,yaw,roll)',
                    'timestamp': 'Unix timestamp (float)',
                    'frame_id': 'Sequential frame number (int)'
                }
            }
            
            summary_path = self.session_dir / "session_summary.json"
            with summary_path.open('w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
                
            self.logger.info(f"Session summary saved to {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving session summary: {e}")
            
    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status and statistics."""
        current_time = time.time()
        session_duration = current_time - self.start_time if self.start_time else 0
        
        return {
            'is_recording': self.is_recording,
            'frame_count': self.frame_count,
            'session_duration': session_duration,
            'sampling_rate': self.sampling_rate,
            'queue_size': self.data_queue.qsize(),
            'session_directory': str(self.session_dir)
        }
        
    def toggle_recording(self):
        """Toggle recording on/off."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def set_sampling_rate(self, rate: float):
        """Update the sampling rate."""
        self.sampling_rate = max(5.0, min(10.0, rate))
        self.logger.info(f"Sampling rate updated to {self.sampling_rate} Hz")
        
    def cleanup(self):
        """Clean up resources and stop recording."""
        if self.is_recording:
            self.stop_recording()
            
        # Clear any remaining data in queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
                self.data_queue.task_done()
            except queue.Empty:
                break
                
        self.logger.info("Data recorder cleaned up")