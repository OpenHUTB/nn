import carla
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import pygame
import time
import queue
import random
import cv2
import math
import logging
import logging.handlers
import sys
import traceback
import os
from matplotlib import pyplot as plt
from pathlib import Path

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class SimulationLogger:
    """
    Enhanced logging system for CARLA AV Simulation.
    
    Features:
    - RotatingFileHandler: auto-rotate at 10MB, keep 5 backups
    - Multi-level logging: DEBUG/INFO/WARNING/ERROR
    - Performance metrics: FPS, memory, queue sizes
    - Structured JSON log format for analysis
    - Separate log files for different concerns
    """
    
    _instance = None
    
    def __init__(self, log_dir='logs', level=logging.INFO):
        """
        Initialize enhanced logging system.
        
        Args:
            log_dir (str): Directory for log files
            level: Minimum log level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.level = level
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_fps_count = 0
        self.performance_log = []
        
        self._setup_loggers()
    
    @staticmethod
    def _fix_console_encoding():
        pass
    
    def _setup_loggers(self):
        """Set up all loggers with appropriate handlers."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(log_format, date_format)
        
        json_formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
            date_format
        )
        
        # Main logger - RotatingFileHandler (10MB, 5 backups)
        self.logger = logging.getLogger('carla_sim')
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
        )
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Rotating file handler (10MB per file, keep 5)
        main_log = self.log_dir / 'simulation.log'
        file_handler = logging.handlers.RotatingFileHandler(
            main_log,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Performance metrics logger (separate file)
        perf_log = self.log_dir / 'performance.log'
        self.perf_logger = logging.getLogger('carla_sim.perf')
        self.perf_logger.setLevel(logging.DEBUG)
        self.perf_logger.handlers.clear()
        self.perf_logger.propagate = False
        
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log,
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(json_formatter)
        self.perf_logger.addHandler(perf_handler)
        
        # Sensor data logger (separate file, JSON format)
        sensor_log = self.log_dir / 'sensor_events.log'
        self.sensor_logger = logging.getLogger('carla_sim.sensor')
        self.sensor_logger.setLevel(logging.DEBUG)
        self.sensor_logger.handlers.clear()
        self.sensor_logger.propagate = False
        
        sensor_handler = logging.handlers.RotatingFileHandler(
            sensor_log,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        sensor_handler.setLevel(logging.DEBUG)
        sensor_handler.setFormatter(json_formatter)
        self.sensor_logger.addHandler(sensor_handler)
        
        self.logger.info(f"📝 Enhanced logging initialized: {self.log_dir}")
        self.logger.info(f"   Main log: {main_log} (rotating, 10MB x 5)")
        self.logger.info(f"   Perf log: {perf_log} (JSON format)")
        self.logger.info(f"   Sensor log: {sensor_log} (JSON format)")
    
    def record_frame(self, fps=None, queue_sizes=None, memory_mb=None):
        """
        Record performance metrics for current frame.
        
        Args:
            fps (float): Current frames per second
            queue_sizes (dict): Queue sizes {name: size}
            memory_mb (float): Current memory usage in MB
        """
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if fps is None:
            now = time.time()
            dt = now - self.last_fps_time
            if dt >= 1.0:
                fps = (self.frame_count - self.last_fps_count) / dt
                self.last_fps_time = now
                self.last_fps_count = self.frame_count
            else:
                fps = 0
        
        metrics = {
            'elapsed_s': round(elapsed, 2),
            'frame': self.frame_count,
            'fps': round(fps, 1) if fps else 0,
            'memory_mb': round(memory_mb, 1) if memory_mb else 0,
            'queues': queue_sizes or {}
        }
        
        self.performance_log.append(metrics)
        self.perf_logger.info(json.dumps(metrics))
        
        return metrics
    
    def log_sensor_event(self, sensor_type, event_type, data=None):
        """
        Log a sensor event in structured JSON format.
        
        Args:
            sensor_type (str): Type of sensor (camera, lidar, radar)
            event_type (str): Event type (data_received, queue_full, etc.)
            data (dict): Additional event data
        """
        event = {
            'sensor': sensor_type,
            'event': event_type,
            'frame': self.frame_count,
            'timestamp': datetime.now().isoformat()
        }
        if data:
            event.update(data)
        self.sensor_logger.info(json.dumps(event))
    
    def log_weather_change(self, old_params, new_params, source='unknown'):
        """Log weather parameter changes."""
        self.logger.info(f"🌦️  Weather changed (source: {source})")
        for key in new_params:
            old_val = old_params.get(key, 'N/A')
            new_val = new_params[key]
            if old_val != new_val:
                self.logger.debug(f"   {key}: {old_val} → {new_val}")
    
    def log_traffic_stats(self, vehicles, pedestrians):
        """Log traffic statistics."""
        self.logger.info(f"🚗 Traffic: {vehicles} vehicles, {pedestrians} pedestrians")
    
    def log_error_with_context(self, error, context=None):
        """Log error with additional context information."""
        self.logger.error(f"❌ {str(error)}")
        if context:
            self.logger.debug(f"   Context: {json.dumps(context, default=str)}")
    
    def get_summary(self):
        """Get logging session summary."""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        return {
            'total_frames': self.frame_count,
            'elapsed_seconds': round(elapsed, 2),
            'avg_fps': round(avg_fps, 2),
            'log_dir': str(self.log_dir),
            'performance_entries': len(self.performance_log)
        }
    
    def print_summary(self):
        """Print logging session summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("📝 Logging Session Summary:")
        print("="*60)
        print(f"   Total Frames: {summary['total_frames']}")
        print(f"   Elapsed Time: {summary['elapsed_seconds']}s")
        print(f"   Average FPS: {summary['avg_fps']}")
        print(f"   Log Directory: {summary['log_dir']}")
        print(f"   Performance Entries: {summary['performance_entries']}")
        print("="*60 + "\n")


# Initialize enhanced logging
_sim_logger = SimulationLogger()


# Convenience functions for backward compatibility
def get_logger():
    """Get the main simulation logger."""
    return _sim_logger.logger


def get_perf_logger():
    """Get the performance metrics logger."""
    return _sim_logger


def log_sensor_event(sensor_type, event_type, data=None):
    """Log a sensor event."""
    _sim_logger.log_sensor_event(sensor_type, event_type, data)


def log_performance(fps=None, queue_sizes=None, memory_mb=None):
    """Record performance metrics."""
    return _sim_logger.record_frame(fps, queue_sizes, memory_mb)


class MemoryMonitor:
    """
    Memory monitoring and management system for long-running simulations.
    
    Features:
    - Real-time memory usage tracking (RSS)
    - Configurable memory threshold warnings
    - Automatic incremental export when memory exceeds threshold
    - Frame count limits per sensor type
    - Memory usage history logging
    """
    
    def __init__(self, max_memory_mb=4096, max_frames_per_sensor=5000,
                 check_interval_seconds=5, warn_threshold_pct=0.8):
        self.max_memory_mb = max_memory_mb
        self.max_frames_per_sensor = max_frames_per_sensor
        self.check_interval_seconds = check_interval_seconds
        self.warn_threshold_pct = warn_threshold_pct
        self.last_check_time = 0
        self.memory_history = []
        self.export_count = 0
        
    def get_memory_mb(self):
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            try:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except ImportError:
                return 0
    
    def should_check(self, current_time):
        """Check if it's time for a memory check."""
        return current_time - self.last_check_time >= self.check_interval_seconds
    
    def check_memory(self, sensor_data, current_time=None):
        """
        Check memory usage and sensor data sizes.
        
        Args:
            sensor_data (dict): The sensor_data dictionary from AVSimulation
            current_time (float): Current time (time.time())
            
        Returns:
            dict: Memory status with recommendations
        """
        if current_time is None:
            current_time = time.time()
            
        if not self.should_check(current_time):
            return {'status': 'ok', 'action': 'none'}
        
        self.last_check_time = current_time
        memory_mb = self.get_memory_mb()
        
        frame_counts = {}
        for sensor_type, data_list in sensor_data.items():
            frame_counts[sensor_type] = len(data_list)
        
        status = {
            'memory_mb': round(memory_mb, 1),
            'max_memory_mb': self.max_memory_mb,
            'memory_pct': round(memory_mb / self.max_memory_mb * 100, 1) if self.max_memory_mb > 0 else 0,
            'frame_counts': frame_counts,
            'status': 'ok',
            'action': 'none'
        }
        
        self.memory_history.append({
            'time': datetime.now().isoformat(),
            'memory_mb': status['memory_mb'],
            'frame_counts': frame_counts.copy()
        })
        
        if memory_mb >= self.max_memory_mb:
            status['status'] = 'critical'
            status['action'] = 'export_and_clear'
            logging.warning(f"CRITICAL: Memory {memory_mb:.0f}MB >= {self.max_memory_mb}MB! Forcing export.")
        elif memory_mb >= self.max_memory_mb * self.warn_threshold_pct:
            status['status'] = 'warning'
            status['action'] = 'export_and_clear'
            logging.warning(f"WARNING: Memory {memory_mb:.0f}MB >= {self.max_memory_mb * self.warn_threshold_pct:.0f}MB threshold")
        
        for sensor_type, count in frame_counts.items():
            if count >= self.max_frames_per_sensor:
                status['status'] = 'frame_limit'
                status['action'] = 'export_and_clear'
                status['limit_sensor'] = sensor_type
                logging.warning(f"{sensor_type} frames ({count}) >= limit ({self.max_frames_per_sensor}). Triggering export.")
                break
        
        if status['memory_mb'] > 0:
            logging.info(f"Memory: {status['memory_mb']}MB ({status['memory_pct']}%) | "
                        f"Camera: {frame_counts.get('camera', 0)} | "
                        f"LiDAR: {frame_counts.get('lidar', 0)} | "
                        f"Radar: {frame_counts.get('radar', 0)}")
        
        return status
    
    def clear_sensor_data(self, sensor_data, keep_last_n=100):
        """Clear sensor data after export, keeping the most recent N frames."""
        for sensor_type in sensor_data:
            if isinstance(sensor_data[sensor_type], list) and len(sensor_data[sensor_type]) > keep_last_n:
                old_count = len(sensor_data[sensor_type])
                sensor_data[sensor_type] = sensor_data[sensor_type][-keep_last_n:]
                logging.info(f"Cleared {sensor_type}: {old_count} -> {len(sensor_data[sensor_type])} frames")
    
    def get_summary(self):
        """Get memory monitoring summary."""
        return {
            'export_count': self.export_count,
            'memory_history_entries': len(self.memory_history),
            'peak_memory_mb': max(h['memory_mb'] for h in self.memory_history) if self.memory_history else 0,
            'max_memory_mb': self.max_memory_mb
        }


class AsyncSensorProcessor:
    """
    Asynchronous sensor data processing system.
    
    Features:
    - ThreadPoolExecutor for non-blocking sensor callback processing
    - Batch processing of sensor data to reduce overhead
    - Non-blocking queue operations with drop policy
    - Processing statistics and performance tracking
    """
    
    def __init__(self, max_workers=4, batch_size=5, batch_timeout=0.1):
        """
        Initialize async sensor processor.
        
        Args:
            max_workers (int): Number of worker threads for processing
            batch_size (int): Number of items to process per batch
            batch_timeout (float): Max seconds to wait before processing a partial batch
        """
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self._pending_camera = []
        self._pending_lidar = []
        self._pending_radar = []
        self._lock_camera = __import__('threading').Lock()
        self._lock_lidar = __import__('threading').Lock()
        self._lock_radar = __import__('threading').Lock()
        
        self.stats = {
            'camera_processed': 0,
            'camera_dropped': 0,
            'lidar_processed': 0,
            'lidar_dropped': 0,
            'radar_processed': 0,
            'radar_dropped': 0,
            'batches_flushed': 0
        }
        self._shutdown = False
        
    def process_camera(self, data, image_queue, sensor_data_list):
        """
        Process camera data asynchronously.
        
        Args:
            data: CARLA sensor data
            image_queue: Queue for visualization
            sensor_data_list: List to append processed data
        """
        def _process():
            try:
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                
                try:
                    if not image_queue.full():
                        image_queue.put((data.frame, array), block=False)
                    else:
                        with self._lock_camera:
                            self.stats['camera_dropped'] += 1
                except queue.Full:
                    with self._lock_camera:
                        self.stats['camera_dropped'] += 1
                
                with self._lock_camera:
                    self._pending_camera.append({
                        'timestamp': data.timestamp,
                        'frame': data.frame,
                        'data': array,
                        'transform': data.transform
                    })
                    self.stats['camera_processed'] += 1
                    
                    if len(self._pending_camera) >= self.batch_size:
                        sensor_data_list.extend(self._pending_camera)
                        self._pending_camera.clear()
                        self.stats['batches_flushed'] += 1
            except Exception as e:
                logging.error(f"Error processing camera data: {str(e)}")
        
        self.executor.submit(_process)
    
    def process_lidar(self, data, lidar_queue, sensor_data_list):
        """
        Process LiDAR data asynchronously.
        
        Args:
            data: CARLA sensor data
            lidar_queue: Queue for visualization
            sensor_data_list: List to append processed data
        """
        def _process():
            try:
                points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                
                try:
                    if not lidar_queue.full():
                        lidar_queue.put((data.frame, points), block=False)
                    else:
                        with self._lock_lidar:
                            self.stats['lidar_dropped'] += 1
                except queue.Full:
                    with self._lock_lidar:
                        self.stats['lidar_dropped'] += 1
                
                with self._lock_lidar:
                    self._pending_lidar.append({
                        'timestamp': data.timestamp,
                        'frame': data.frame,
                        'points': points,
                        'transform': data.transform
                    })
                    self.stats['lidar_processed'] += 1
                    
                    if len(self._pending_lidar) >= self.batch_size:
                        sensor_data_list.extend(self._pending_lidar)
                        self._pending_lidar.clear()
                        self.stats['batches_flushed'] += 1
            except Exception as e:
                logging.error(f"Error processing lidar data: {str(e)}")
        
        self.executor.submit(_process)
    
    def process_radar(self, data, radar_queue, sensor_data_list):
        """
        Process radar data asynchronously.
        
        Args:
            data: CARLA sensor data
            radar_queue: Queue for visualization
            sensor_data_list: List to append processed data
        """
        def _process():
            try:
                points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                
                try:
                    if not radar_queue.full():
                        radar_queue.put((data.frame, points), block=False)
                    else:
                        with self._lock_radar:
                            self.stats['radar_dropped'] += 1
                except queue.Full:
                    with self._lock_radar:
                        self.stats['radar_dropped'] += 1
                
                with self._lock_radar:
                    self._pending_radar.append({
                        'timestamp': data.timestamp,
                        'frame': data.frame,
                        'points': points,
                        'transform': data.transform
                    })
                    self.stats['radar_processed'] += 1
                    
                    if len(self._pending_radar) >= self.batch_size:
                        sensor_data_list.extend(self._pending_radar)
                        self._pending_radar.clear()
                        self.stats['batches_flushed'] += 1
            except Exception as e:
                logging.error(f"Error processing radar data: {str(e)}")
        
        self.executor.submit(_process)
    
    def flush_all(self, sensor_data):
        """
        Flush all pending data to sensor_data lists.
        Call this at the end of simulation or periodically.
        
        Args:
            sensor_data (dict): The sensor_data dictionary
        """
        with self._lock_camera:
            if self._pending_camera:
                sensor_data['camera'].extend(self._pending_camera)
                self._pending_camera.clear()
                self.stats['batches_flushed'] += 1
        
        with self._lock_lidar:
            if self._pending_lidar:
                sensor_data['lidar'].extend(self._pending_lidar)
                self._pending_lidar.clear()
                self.stats['batches_flushed'] += 1
        
        with self._lock_radar:
            if self._pending_radar:
                sensor_data['radar'].extend(self._pending_radar)
                self._pending_radar.clear()
                self.stats['batches_flushed'] += 1
    
    def get_stats(self):
        """Get processing statistics."""
        total_processed = (self.stats['camera_processed'] + 
                          self.stats['lidar_processed'] + 
                          self.stats['radar_processed'])
        total_dropped = (self.stats['camera_dropped'] + 
                        self.stats['lidar_dropped'] + 
                        self.stats['radar_dropped'])
        return {
            **self.stats,
            'total_processed': total_processed,
            'total_dropped': total_dropped,
            'drop_rate': f"{total_dropped / (total_processed + total_dropped) * 100:.1f}%" if (total_processed + total_dropped) > 0 else "0%"
        }
    
    def print_stats(self):
        """Print processing statistics."""
        stats = self.get_stats()
        try:
            print("\n" + "="*60)
            print("[AsyncSensor] Processor Statistics:")
            print("="*60)
            print(f"   Camera: {stats['camera_processed']} processed, {stats['camera_dropped']} dropped")
            print(f"   LiDAR:  {stats['lidar_processed']} processed, {stats['lidar_dropped']} dropped")
            print(f"   Radar:  {stats['radar_processed']} processed, {stats['radar_dropped']} dropped")
            print(f"   Total:  {stats['total_processed']} processed, {stats['total_dropped']} dropped ({stats['drop_rate']})")
            print(f"   Batches flushed: {stats['batches_flushed']}")
            print("="*60 + "\n")
        except UnicodeEncodeError:
            logging.info(f"[AsyncSensor] Stats: {stats}")
    
    def shutdown(self):
        """Shutdown the executor."""
        self._shutdown = True
        self.executor.shutdown(wait=True)


class ConfigLoader:
    """
    Configuration loader for CARLA AV Simulation.
    
    Loads settings from carla_settings.ini (YAML format) and provides
    easy access to all configuration parameters with fallback defaults.
    
    Features:
    - Load from YAML config file (carla_settings.ini)
    - Provide default values if config missing
    - Validate parameter ranges
    - Support runtime config updates
    """
    
    DEFAULT_CONFIG = {
        'simulation': {
            'version': '0.9.15',
            'tick_rate': 30,
            'duration': 600,
            'random_seed': 42,
            'synchronous_mode': True
        },
        'world': {
            'weather': {
                'rain_intensity': 100.0,
                'puddles': 100.0,
                'wetness': 100.0,
                'fog_density': 20.0,
                'wind_intensity': 50.0,
                'cloudiness': 100.0,
                'sun_altitude_angle': 45.0,
                'precipitation_deposits': 100.0
            }
        },
        'traffic': {
            'max_vehicles': 50,
            'min_vehicles': 30,
            'spawn_spacing': 2.0,
            'speed_variance': [-20, 10],
            'safe_distance': 0.5,
            'respect_traffic_lights': True,
            'ignore_lights_percentage': 0.0,
            'max_retry_attempts': 10
        },
        'pedestrians': {
            'max_pedestrians': 30,
            'min_pedestrians': 15,
            'speed_range': [0.8, 1.8],
            'crossing_percentage': 0.7,
            'max_spawn_attempts': 5,
            'safe_spawn_distance': 2.0
        }
    }
    
    def __init__(self, config_path='carla_settings.ini'):
        """
        Initialize config loader.
        
        Args:
            config_path (str): Path to configuration file (YAML format)
        """
        self.config_path = Path(config_path)
        self.config = None
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file with fallback to defaults."""
        try:
            if self.config_path.exists() and YAML_AVAILABLE:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logging.info(f"✅ Configuration loaded from {self.config_path}")
            else:
                if not YAML_AVAILABLE:
                    logging.warning("⚠️  PyYAML not available, using default configuration")
                else:
                    logging.warning(f"⚠️  Config file not found: {self.config_path}, using defaults")
                self.config = self.DEFAULT_CONFIG
        except Exception as e:
            logging.error(f"❌ Failed to load config: {str(e)}. Using defaults.")
            self.config = self.DEFAULT_CONFIG
            
    def reload(self):
        """Reload configuration from file (hot reload)."""
        self._load_config()
        logging.info("🔄 Configuration reloaded")
        
    def get(self, key_path, default=None):
        """
        Get configuration value by dotted path.
        
        Args:
            key_path (str): Dotted path to config value (e.g., 'world.weather.rain_intensity')
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def get_weather_params(self):
        """
        Get weather parameters from config.
        
        Returns:
            dict: Weather parameters dictionary
        """
        weather_config = self.get('world.weather', self.DEFAULT_CONFIG['world']['weather'])
        return {
            'cloudiness': float(weather_config.get('cloudiness', 100.0)),
            'precipitation': float(weather_config.get('rain_intensity', 100.0)),
            'precipitation_deposits': float(weather_config.get('precipitation_deposits', 100.0)),
            'wind_intensity': float(weather_config.get('wind_intensity', 50.0)),
            'fog_density': float(weather_config.get('fog_density', 20.0)),
            'wetness': float(weather_config.get('wetness', 100.0)),
            'sun_altitude_angle': float(weather_config.get('sun_altitude_angle', 45.0))
        }
    
    def get_traffic_params(self):
        """
        Get traffic parameters from config.
        
        Returns:
            dict: Traffic parameters with max_vehicles, max_pedestrians, etc.
        """
        traffic = self.get('traffic', self.DEFAULT_CONFIG['traffic'])
        pedestrians = self.get('pedestrians', self.DEFAULT_CONFIG['pedestrians'])
        
        return {
            'max_vehicles': int(traffic.get('max_vehicles', 50)),
            'min_vehicles': int(traffic.get('min_vehicles', 30)),
            'spawn_spacing': float(traffic.get('spawn_spacing', 2.0)),
            'safe_distance': float(traffic.get('safe_distance', 0.5)),
            'respect_traffic_lights': bool(traffic.get('respect_traffic_lights', True)),
            'max_pedestrians': int(pedestrians.get('max_pedestrians', 30)),
            'min_pedestrians': int(pedestrians.get('min_pedestrians', 15))
        }
    
    def get_sensor_config(self, config_name='minimal'):
        """
        Get sensor configuration by name.
        
        Args:
            config_name (str): Name of sensor config ('minimal', 'standard', 'advanced')
            
        Returns:
            dict: Sensor configuration or None if not found
        """
        sensor_configs = self.get('sensor_configurations', {})
        return sensor_configs.get(config_name, None)
    
    def validate_weather_params(self, params):
        """
        Validate weather parameters are within valid ranges.
        
        Args:
            params (dict): Weather parameters dictionary
            
        Returns:
            tuple: (is_valid: bool, errors: list)
        """
        errors = []
        ranges = {
            'cloudiness': (0, 100),
            'precipitation': (0, 100),
            'precipitation_deposits': (0, 100),
            'wind_intensity': (0, 100),
            'fog_density': (0, 100),
            'wetness': (0, 100),
            'sun_altitude_angle': (-90, 90)
        }
        
        for param, (min_val, max_val) in ranges.items():
            value = params.get(param)
            if value is not None:
                if not (min_val <= value <= max_val):
                    errors.append(f"{param}: {value} out of range [{min_val}, {max_val}]")
                    
        return len(errors) == 0, errors
    
    def print_summary(self):
        """Print configuration summary for debugging."""
        try:
            print("\n" + "="*60)
            print("[Config] Current Configuration Summary:")
            print("="*60)
            
            weather = self.get_weather_params()
            print("\n[Weather] Parameters:")
            for key, value in weather.items():
                print(f"   {key}: {value}")
                
            traffic = self.get_traffic_params()
            print("\n[Traffic] Parameters:")
            for key, value in traffic.items():
                print(f"   {key}: {value}")
                
            sim = self.get('simulation', {})
            print("\n[Simulation] Settings:")
            print(f"   Duration: {sim.get('duration', 600)}s")
            print(f"   Tick Rate: {sim.get('tick_rate', 30)} FPS")
            print("="*60 + "\n")
        except UnicodeEncodeError:
            logging.info("Configuration summary printed (console encoding skipped emoji)")

class SimulationError(Exception):
    """Custom exception for simulation-specific errors"""
    pass

class SensorConfiguration:
    """
    Class to hold sensor configuration specifications.
    """
    def __init__(self, name, sensors_specs):
        """
        Initialize a sensor configuration.

        Args:
            name (str): Name of the configuration
            sensors_specs (list): List of tuples (blueprint_name, attributes, transform)
                where attributes is a dict of sensor attributes
        """
        self.name = name
        self.sensors_specs = sensors_specs

class AVSimulation:
    def __init__(self, config_file='carla_settings.ini'):
        """
        Initialize CARLA AV Simulation with configuration file support.
        
        Args:
            config_file (str): Path to configuration file (YAML format)
        """
        try:
            # Load configuration from file
            self.config_loader = ConfigLoader(config_file)
            
            # Print configuration summary on startup
            try:
                self.config_loader.print_summary()
            except Exception:
                logging.info("Configuration loaded (summary display skipped)")
            
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(20.0)  # Increased timeout

            # Verify connection
            try:
                self.world = self.client.get_world()
                logging.info("Successfully connected to CARLA server")
            except RuntimeError as e:
                raise SimulationError(f"Failed to connect to CARLA server: {str(e)}")

            self.map = self.world.get_map()

            # Initialize Pygame
            if not pygame.get_init():
                pygame.init()
            try:
                viz_config = self.config_loader.get('visualization', {})
                window_size = viz_config.get('window_size', [1920, 1080])
                self.display = pygame.display.set_mode(tuple(window_size), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()

                self._viz_fps = viz_config.get('visualization_fps', 15)
                self._viz_interval = 1.0 / self._viz_fps
                self._last_viz_time = 0.0
                self._lidar_display_points = viz_config.get('lidar_display_points', 2000)
                self._skip_when_minimized = viz_config.get('skip_when_minimized', True)

                self._font = pygame.font.Font(None, 36)
                self._text_cache = {}
                self._text_cache_frame = -1
            except pygame.error as e:
                raise SimulationError(f"Failed to initialize Pygame display: {str(e)}")

            # Initialize queues with configurable max size
            perf_config = self.config_loader.get('performance', {})
            max_queue_size = perf_config.get('max_sensor_queue_size', 100)
            
            self.image_queue = queue.Queue(maxsize=max_queue_size)
            self.lidar_queue = queue.Queue(maxsize=max_queue_size)
            self.radar_queue = queue.Queue(maxsize=max_queue_size)

            # Sensor data storage with capacity checks
            self.sensor_data = {
                'camera': [],
                'lidar': [],
                'radar': [],
                'semantic': [],
                'depth': [],
                'weather': []
            }

            self.active_sensors = []
            self.active_actors = []

            # Memory monitor for long-running simulations
            perf_config = self.config_loader.get('performance', {})
            self.memory_monitor = MemoryMonitor(
                max_memory_mb=perf_config.get('max_memory_mb', 4096),
                max_frames_per_sensor=perf_config.get('max_frames_per_sensor', 5000),
                check_interval_seconds=perf_config.get('memory_check_interval', 5),
                warn_threshold_pct=perf_config.get('memory_warn_threshold', 0.8)
            )
            self.incremental_export_count = 0

            # Async sensor processor for non-blocking callbacks
            async_config = perf_config.get('async_sensor', {})
            self.async_processor = AsyncSensorProcessor(
                max_workers=async_config.get('max_workers', 4),
                batch_size=async_config.get('batch_size', 5),
                batch_timeout=async_config.get('batch_timeout', 0.1)
            )

            # Define sensor configurations (now can be overridden by config file)
            self.define_sensor_configurations()

            logging.info("✅ AVSimulation initialized successfully (with config file support + async sensor processing)")

        except Exception as e:
            logging.error(f"Failed to initialize AVSimulation: {str(e)}")
            raise
    def setup_sensors(self, vehicle, config_name):
        """
        Set up sensors based on the selected configuration.

        Args:
            vehicle: CARLA vehicle actor to attach sensors to
            config_name: Name of sensor configuration to use ('minimal', 'standard', or 'advanced')
        """
        sensors = []
        try:
            if config_name not in self.sensor_configurations:
                raise SimulationError(f"Invalid sensor configuration: {config_name}")

            config = self.sensor_configurations[config_name]
            logging.info(f"Setting up {config_name} sensor configuration")

            for blueprint_name, attributes, transform in config.sensors_specs:
                try:
                    # Get the blueprint
                    blueprint = self.world.get_blueprint_library().find(blueprint_name)
                    if blueprint is None:
                        raise SimulationError(f"Sensor blueprint not found: {blueprint_name}")

                    # Set attributes
                    for attr_name, attr_value in attributes.items():
                        blueprint.set_attribute(attr_name, attr_value)

                    # Spawn the sensor
                    sensor = self.world.spawn_actor(
                        blueprint,
                        transform,
                        attach_to=vehicle
                    )

                    if sensor is None:
                        raise SimulationError(f"Failed to spawn sensor: {blueprint_name}")

                    # Set up the appropriate callback
                    if 'camera.rgb' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'camera'))
                        logging.info(f"RGB camera set up at location: {transform.location}")

                    elif 'camera.semantic_segmentation' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'semantic'))
                        logging.info(f"Semantic segmentation camera set up at location: {transform.location}")

                    elif 'camera.depth' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'depth'))
                        logging.info(f"Depth camera set up at location: {transform.location}")

                    elif 'lidar' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'lidar'))
                        logging.info(f"LiDAR set up at location: {transform.location}")

                    elif 'radar' in blueprint_name:
                        sensor.listen(lambda data: self.sensor_callback(data, 'radar'))
                        logging.info(f"Radar set up at location: {transform.location}")

                    sensors.append(sensor)
                    self.active_sensors.append(sensor)

                except Exception as e:
                    logging.error(f"Failed to set up sensor {blueprint_name}: {str(e)}")
                    # Continue with other sensors even if one fails
                    continue

            if not sensors:
                raise SimulationError("No sensors were successfully set up")

            logging.info(f"Successfully set up {len(sensors)} sensors")
            return sensors

        except Exception as e:
            logging.error(f"Error in setup_sensors: {str(e)}")
            # Clean up any sensors that were created before the error
            for sensor in sensors:
                try:
                    if sensor is not None and sensor.is_alive:
                        sensor.destroy()
                        logging.info(f"Cleaned up sensor after setup error: {sensor.type_id}")
                except:
                    pass
            raise SimulationError(f"Sensor setup failed: {str(e)}")

    def define_sensor_configurations(self):
        """
        Define the sensor configurations available in the simulation.
        """
        self.sensor_configurations = {
            'minimal': SensorConfiguration('minimal', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '32',
                    'points_per_second': '100000',
                    'rotation_frequency': '20',
                    'range': '50'
                }, carla.Transform(carla.Location(x=0, z=1.8)))
            ]),

            'standard': SensorConfiguration('standard', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '64',
                    'points_per_second': '200000',
                    'rotation_frequency': '20',
                    'range': '70'
                }, carla.Transform(carla.Location(x=0, z=1.8))),
                ('sensor.other.radar', {
                    'horizontal_fov': '30',
                    'vertical_fov': '30',
                    'points_per_second': '1500',
                    'range': '100'
                }, carla.Transform(carla.Location(x=2.0, z=1.0)))
            ]),

            'advanced': SensorConfiguration('advanced', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.camera.semantic_segmentation', {
                    'image_size_x': '1920',
                    'image_size_y': '1080'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.camera.depth', {
                    'image_size_x': '1920',
                    'image_size_y': '1080'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '128',
                    'points_per_second': '500000',
                    'rotation_frequency': '20',
                    'range': '100'
                }, carla.Transform(carla.Location(x=0, z=1.8))),
                ('sensor.other.radar', {
                    'horizontal_fov': '45',
                    'vertical_fov': '45',
                    'points_per_second': '2000',
                    'range': '100'
                }, carla.Transform(carla.Location(x=2.0, z=1.0)))
            ])
        }
    def setup_weather(self, weather_params=None):
        """
        Set up weather conditions from config file or custom parameters.

        Args:
            weather_params (dict): Optional custom weather parameters (overrides config file)
                                   If None, loads from carla_settings.ini
        """
        try:
            # Load from config file if no custom params provided
            if weather_params is None:
                weather_params = self.config_loader.get_weather_params()
                logging.info("📂 Loading weather parameters from configuration file")
            
            # Validate parameters
            is_valid, errors = self.config_loader.validate_weather_params(weather_params)
            if not is_valid:
                logging.warning(f"⚠️  Weather parameter validation warnings: {errors}")
            
            # Set up weather with loaded/custom parameters
            weather = carla.WeatherParameters(
                cloudiness=weather_params.get('cloudiness', 100.0),
                precipitation=weather_params.get('precipitation', 100.0),
                precipitation_deposits=weather_params.get('precipitation_deposits', 100.0),
                wind_intensity=weather_params.get('wind_intensity', 50.0),
                fog_density=weather_params.get('fog_density', 20.0),
                wetness=weather_params.get('wetness', 100.0),
                sun_altitude_angle=weather_params.get('sun_altitude_angle', 45.0)
            )

            # Apply weather settings
            self.world.set_weather(weather)

            # Store weather state
            self.sensor_data['weather'].append({
                'timestamp': datetime.now().isoformat(),
                'source': 'config_file' if weather_params else 'custom',
                'params': {
                    'cloudiness': weather.cloudiness,
                    'precipitation': weather.precipitation,
                    'precipitation_deposits': weather.precipitation_deposits,
                    'wind_intensity': weather.wind_intensity,
                    'fog_density': weather.fog_density,
                    'wetness': weather.wetness,
                    'sun_altitude_angle': weather.sun_altitude_angle
                }
            })

            logging.info(f"✅ Weather configured: Rain={weather.precipitation:.1f}%, Fog={weather.fog_density:.1f}%, Wind={weather.wind_intensity:.1f}%")
            return weather

        except Exception as e:
            logging.error(f"Failed to setup weather: {str(e)}")
            raise SimulationError(f"Weather setup failed: {str(e)}")

    def setup_traffic(self, num_vehicles=None, num_pedestrians=None):
        """
        Set up traffic and pedestrians from config file or custom parameters.

        Args:
            num_vehicles (int): Number of vehicles (if None, loads from config)
            num_pedestrians (int): Number of pedestrians (if None, loads from config)
        """
        vehicles = []
        pedestrians = []
        controllers = []
        
        try:
            # Load defaults from config file if not provided
            if num_vehicles is None or num_pedestrians is None:
                traffic_params = self.config_loader.get_traffic_params()
                if num_vehicles is None:
                    num_vehicles = traffic_params['max_vehicles']
                    logging.info(f"📂 Loading vehicle count from config: {num_vehicles}")
                if num_pedestrians is None:
                    num_pedestrians = traffic_params['max_pedestrians']
                    logging.info(f"📂 Loading pedestrian count from config: {num_pedestrians}")
            
            # Set up traffic manager with config parameters
            traffic_manager = self.client.get_trafficmanager(8000)  # Port 8000
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(0)
            
            # Apply traffic settings from config
            safe_distance = self.config_loader.get('traffic.safe_distance', 0.5)
            traffic_manager.global_percentage_speed_difference(0)
            # Note: safe_distance can be applied per-vehicle later if needed
            
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise SimulationError("No spawn points found in map")
            
            # Shuffle spawn points
            random.shuffle(spawn_points)
            
            # Spawn vehicles with collision checking
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            for _ in range(num_vehicles):
                try:
                    blueprint = random.choice(vehicle_bps)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                    
                    vehicle = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
                    if vehicle is not None:
                        vehicle.set_autopilot(True, traffic_manager.get_port())
                        # Set behavior parameters
                        traffic_manager.update_vehicle_lights(vehicle, True)
                        traffic_manager.distance_to_leading_vehicle(vehicle, 0.5)
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-20, 10))
                        
                        vehicles.append(vehicle)
                        self.active_actors.append(vehicle)
                except Exception as e:
                    logging.warning(f"Failed to spawn vehicle: {str(e)}")
                    continue
                
            logging.info(f"Successfully spawned {len(vehicles)} vehicles")
            
            # Spawn pedestrians with collision checking
            pedestrian_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            
            for _ in range(num_pedestrians):
                try:
                    # Try multiple times to find a valid spawn point
                    for _ in range(5):  # Try 5 times per pedestrian
                        spawn_point = carla.Transform(
                            self.world.get_random_location_from_navigation(),
                            carla.Rotation()
                        )
                        
                        # Check if location is valid
                        if self.world.get_map().get_waypoint(spawn_point.location, project_to_road=False):
                            blueprint = random.choice(pedestrian_bps)
                            pedestrian = self.world.try_spawn_actor(blueprint, spawn_point)
                            
                            if pedestrian is not None:
                                # Spawn controller
                                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                                controller.start()
                                controller.set_max_speed(1.4)
                                
                                pedestrians.append(pedestrian)
                                controllers.append(controller)
                                self.active_actors.extend([pedestrian, controller])
                                break
                
                except Exception as e:
                    logging.warning(f"Failed to spawn pedestrian: {str(e)}")
                    continue
            
            logging.info(f"Successfully spawned {len(pedestrians)} pedestrians")
            
            # Wait a moment for physics to settle
            time.sleep(0.5)
            
            # Start pedestrian movement
            for controller in controllers:
                try:
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    # Add some randomization to pedestrian behavior
                    controller.set_max_speed(random.uniform(0.8, 1.8))
                except Exception as e:
                    logging.warning(f"Failed to set pedestrian destination: {str(e)}")
            
            return vehicles, pedestrians
            
        except Exception as e:
            logging.error(f"Error in setup_traffic: {str(e)}")
            self.cleanup_actors()
            raise

    def sensor_callback(self, data, sensor_type):
        try:
            if self.async_processor._shutdown:
                return
            if sensor_type == 'camera':
                self.async_processor.process_camera(
                    data, self.image_queue, self.sensor_data['camera']
                )
            elif sensor_type == 'lidar':
                self.async_processor.process_lidar(
                    data, self.lidar_queue, self.sensor_data['lidar']
                )
            elif sensor_type == 'radar':
                self.async_processor.process_radar(
                    data, self.radar_queue, self.sensor_data['radar']
                )
            elif sensor_type in ('semantic', 'depth'):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                self.sensor_data[sensor_type].append({
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'data': array,
                    'transform': data.transform
                })
        except Exception as e:
            logging.error(f"Error in sensor callback ({sensor_type}): {str(e)}")

    def cleanup_actors(self):
        """Clean up all actors spawned during simulation"""
        try:
            logging.info("Cleaning up actors...")
            for actor in self.active_actors:
                try:
                    if actor is not None and actor.is_alive:
                        actor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy actor: {str(e)}")
            self.active_actors.clear()

            # Clean up sensors specifically
            for sensor in self.active_sensors:
                try:
                    if sensor is not None and sensor.is_alive:
                        sensor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy sensor: {str(e)}")
            self.active_sensors.clear()

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def _scale_weather_params(self, base_params, intensity):
        scaled = {}
        for key, value in base_params.items():
            if key == 'sun_altitude_angle':
                scaled[key] = value
            else:
                scaled[key] = round(value * intensity, 1)
        return scaled

    def run_simulation(self, config_name, duration_seconds=600, weather_scenario='storm_front'):
        dashboard = None
        weather_scheduler = None
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.025
            self.world.apply_settings(settings)
            
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            
            vehicle = None
            sensors = []
            traffic_vehicles = []
            pedestrians = []

            weather_scheduler = WeatherScheduler(
                scenario_name=weather_scenario,
                duration=duration_seconds,
                enable_random_events=True
            )
            initial_params = weather_scheduler.update(0)
            self.world.set_weather(carla.WeatherParameters(**initial_params))
            logging.info(f"Weather scheduler started: {weather_scheduler.get_scenario_name()}")

            blueprint = self.world.get_blueprint_library().find('vehicle.yamaha.yzf')
            spawn_points = self.map.get_spawn_points()

            if not spawn_points:
                raise SimulationError("No spawn points available")

            spawn_success = False
            random.shuffle(spawn_points)

            for spawn_point in spawn_points:
                try:
                    if self.world.get_spectator().get_transform().location.distance(spawn_point.location) > 2.0:
                        vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                        if vehicle is not None:
                            spawn_success = True
                            break
                except Exception as e:
                    logging.warning(f"Failed to spawn at point {spawn_point}: {str(e)}")
                    continue

            if not spawn_success:
                raise SimulationError("Could not find a clear spawn point for ego vehicle")

            self.active_actors.append(vehicle)
            vehicle.set_autopilot(True)
            logging.info(f"Ego vehicle spawned successfully at {spawn_point}")

            time.sleep(0.5)

            traffic_vehicles, pedestrians = self.setup_traffic()
            logging.info("Traffic setup completed")

            sensors = self.setup_sensors(vehicle, config_name)
            self.active_sensors.extend(sensors)
            logging.info(f"Sensors setup completed for configuration: {config_name}")

            dashboard = SimulationDashboard(
                duration=duration_seconds,
                weather_name=weather_scheduler.get_scenario_name(),
                mode=f'full ({config_name})'
            )
            dashboard.vehicles_count = len(traffic_vehicles)
            dashboard.pedestrians_count = len(pedestrians)
            dashboard.start()

            paused = False
            debug_mode = False
            quit_requested = False

            start_time = time.time()
            frame_count = 0
            last_dash_update = 0
            last_weather_update = 0

            while time.time() - start_time < duration_seconds and not quit_requested:
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit_requested = True
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                paused = not paused
                                dashboard.paused = paused
                            elif event.key == pygame.K_q:
                                quit_requested = True
                                dashboard.quit_requested = True
                            elif event.key == pygame.K_d:
                                debug_mode = not debug_mode
                                dashboard.debug_mode = debug_mode

                    if paused:
                        time.sleep(0.05)
                        elapsed = time.time() - start_time
                        dashboard.update(elapsed=elapsed)
                        continue

                    self.world.tick()
                    
                    now = time.time()
                    elapsed = now - start_time

                    if now - last_weather_update >= 1.0:
                        weather_params = weather_scheduler.update(elapsed)
                        self.world.set_weather(carla.WeatherParameters(**weather_params))
                        last_weather_update = now

                    mem_status = self.memory_monitor.check_memory(
                        self.sensor_data, time.time()
                    )
                    if mem_status['action'] == 'export_and_clear':
                        logging.info("Memory threshold reached, performing incremental export...")
                        self.export_data_incremental(".")
                        self.memory_monitor.clear_sensor_data(self.sensor_data)
                        self.memory_monitor.export_count += 1
                        self.incremental_export_count += 1

                    if frame_count % 10 == 0:
                        self.async_processor.flush_all(self.sensor_data)
                    
                    self.visualize_data()
                    
                    frame_count += 1
                    
                    if now - last_dash_update >= 0.25:
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        mem_mb = self.memory_monitor.get_memory_mb()
                        dashboard.update(
                            elapsed=elapsed,
                            fps=fps,
                            memory_mb=mem_mb,
                            frame_count=frame_count,
                            camera_frames=len(self.sensor_data.get('camera', [])),
                            lidar_frames=len(self.sensor_data.get('lidar', [])),
                            radar_frames=len(self.sensor_data.get('radar', [])),
                            semantic_frames=len(self.sensor_data.get('semantic', [])),
                            depth_frames=len(self.sensor_data.get('depth', [])),
                            queue_sizes={
                                'image': self.image_queue.qsize(),
                                'lidar': self.lidar_queue.qsize(),
                                'radar': self.radar_queue.qsize(),
                            },
                            weather_phase=weather_scheduler.get_phase_label(),
                        )
                        last_dash_update = now
                
                except Exception as e:
                    logging.error(f"Error in simulation loop: {str(e)}")
                    break
                
        finally:
            if dashboard:
                dashboard.stop()
            self.cleanup_actors()
            self.async_processor.flush_all(self.sensor_data)
            self.async_processor.print_stats()
            self.async_processor.shutdown()
            self.export_data(".")

            mem_summary = self.memory_monitor.get_summary()
            logging.info(f"Memory Monitor Summary:")
            logging.info(f"   Incremental exports: {mem_summary['export_count']}")
            logging.info(f"   Peak memory: {mem_summary['peak_memory_mb']:.1f}MB / {mem_summary['max_memory_mb']}MB")

            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def export_data_incremental(self, output_dir):
        """
        Export current sensor data incrementally (for memory management).
        
        Exports data with a unique batch number, then the caller should
        clear the sensor_data to free memory.
        
        Args:
            output_dir (str): Directory to save incremental data
        """
        try:
            batch_id = self.incremental_export_count
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = Path(output_dir) / f"incremental_batch_{batch_id:03d}_{timestamp}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            frame_counts = {}
            for sensor_type, data_list in self.sensor_data.items():
                frame_counts[sensor_type] = len(data_list)
            
            metadata = {
                'batch_id': batch_id,
                'export_timestamp': timestamp,
                'frame_counts': frame_counts,
                'memory_monitor': self.memory_monitor.get_summary()
            }
            
            with open(batch_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            if self.sensor_data['camera']:
                cam_dir = batch_dir / 'camera'
                cam_dir.mkdir(exist_ok=True)
                for idx, frame in enumerate(self.sensor_data['camera']):
                    img = frame.get('data')
                    if img is not None:
                        cv2.imwrite(str(cam_dir / f'frame_{idx:06d}.png'), img)
            
            if self.sensor_data['lidar']:
                lidar_data = {
                    'timestamps': [f['timestamp'] for f in self.sensor_data['lidar']],
                    'points': [f['points'].tolist() for f in self.sensor_data['lidar']]
                }
                with open(batch_dir / 'lidar_data.json', 'w') as f:
                    json.dump(lidar_data, f, default=str)
            
            logging.info(f"📦 Incremental export #{batch_id}: {frame_counts} → {batch_dir}")
            
        except Exception as e:
            logging.error(f"Failed incremental export: {str(e)}")

    def export_data(self, output_dir):
        """
        Export collected sensor data to the specified output directory.
        
        Args:
            output_dir (Path or str): Directory to save the exported data
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure output directory exists
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export metadata as JSON
            try:
                metadata = {
                    'timestamp': timestamp,
                    'weather_conditions': self.sensor_data['weather'],
                    'num_frames': {
                        'camera': len(self.sensor_data['camera']),
                        'lidar': len(self.sensor_data['lidar']),
                        'radar': len(self.sensor_data['radar']),
                        'semantic': len(self.sensor_data['semantic']),
                        'depth': len(self.sensor_data['depth'])
                    }
                }
                
                with open(output_dir / f'metadata_{timestamp}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                logging.info("Metadata export completed")
                
            except Exception as e:
                logging.error(f"Failed to export metadata: {str(e)}")
            
            # Export camera data (images)
            if self.sensor_data['camera']:
                try:
                    camera_dir = output_dir / 'camera_data'
                    camera_dir.mkdir(exist_ok=True)
                    
                    for idx, frame in enumerate(self.sensor_data['camera']):
                        # Save as PNG to preserve quality
                        img_path = camera_dir / f'frame_{idx:06d}.png'
                        cv2.imwrite(str(img_path), cv2.cvtColor(frame['data'], cv2.COLOR_RGB2BGR))
                    
                    logging.info(f"Exported {len(self.sensor_data['camera'])} camera frames")
                except Exception as e:
                    logging.error(f"Failed to export camera data: {str(e)}")
            
            # Export LiDAR points with serializable transforms
            if self.sensor_data['lidar']:
                try:
                    lidar_data = {
                        'timestamps': [frame['timestamp'] for frame in self.sensor_data['lidar']],
                        'points': [frame['points'] for frame in self.sensor_data['lidar']],
                        # Convert Transform objects to dictionaries
                        'transforms': [{
                            'location': {
                                'x': frame['transform'].location.x,
                                'y': frame['transform'].location.y,
                                'z': frame['transform'].location.z
                            },
                            'rotation': {
                                'pitch': frame['transform'].rotation.pitch,
                                'yaw': frame['transform'].rotation.yaw,
                                'roll': frame['transform'].rotation.roll
                            }
                        } for frame in self.sensor_data['lidar']]
                    }
                    
                    with open(output_dir / f'lidar_data_{timestamp}.pkl', 'wb') as f:
                        pickle.dump(lidar_data, f)
                    logging.info(f"Exported {len(self.sensor_data['lidar'])} LiDAR frames")
                except Exception as e:
                    logging.error(f"Failed to export LiDAR data: {str(e)}")
            
            # Export radar data as CSV
            if self.sensor_data['radar']:
                try:
                    radar_frames = []
                    for frame in self.sensor_data['radar']:
                        df = pd.DataFrame(
                            frame['points'],
                            columns=['velocity', 'azimuth', 'altitude', 'depth']
                        )
                        df['timestamp'] = frame['timestamp']
                        df['frame'] = frame.get('frame', 0)
                        radar_frames.append(df)
                    
                    if radar_frames:
                        radar_df = pd.concat(radar_frames, ignore_index=True)
                        radar_df.to_csv(output_dir / f'radar_data_{timestamp}.csv', index=False)
                        logging.info(f"Exported {len(radar_frames)} radar frames")
                except Exception as e:
                    logging.error(f"Failed to export radar data: {str(e)}")
            
            # Export semantic segmentation data if available
            if self.sensor_data['semantic']:
                try:
                    semantic_dir = output_dir / 'semantic_data'
                    semantic_dir.mkdir(exist_ok=True)
                    
                    for idx, frame in enumerate(self.sensor_data['semantic']):
                        semantic_path = semantic_dir / f'frame_{idx:06d}.png'
                        cv2.imwrite(str(semantic_path), frame['data'])
                    
                    logging.info(f"Exported {len(self.sensor_data['semantic'])} semantic segmentation frames")
                except Exception as e:
                    logging.error(f"Failed to export semantic segmentation data: {str(e)}")
            
            # Export depth data if available
            if self.sensor_data['depth']:
                try:
                    depth_dir = output_dir / 'depth_data'
                    depth_dir.mkdir(exist_ok=True)
                    
                    for idx, frame in enumerate(self.sensor_data['depth']):
                        depth_path = depth_dir / f'frame_{idx:06d}.png'
                        # Normalize depth data for visualization
                        depth_data = frame['data'].astype(np.float32)
                        depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                        cv2.imwrite(str(depth_path), depth_data.astype(np.uint8))
                    
                    logging.info(f"Exported {len(self.sensor_data['depth'])} depth frames")
                except Exception as e:
                    logging.error(f"Failed to export depth data: {str(e)}")
            
            logging.info(f"Data export completed successfully to {output_dir}")
            
        except Exception as e:
            logging.error(f"Error during data export: {str(e)}")
            raise SimulationError(f"Data export failed: {str(e)}")
    def visualize_data(self):
        now = time.time()
        if now - self._last_viz_time < self._viz_interval:
            return
        self._last_viz_time = now

        if self._skip_when_minimized:
            try:
                import ctypes
                hwnd = pygame.display.get_wm_info()['window']
                if ctypes.windll.user32.IsIconic(hwnd):
                    return
            except Exception:
                pass

        try:
            if not self.image_queue.empty():
                frame, image = self.image_queue.get()
                if not np.isnan(image).any():
                    image = np.rot90(image)
                    image = pygame.surfarray.make_surface(image)
                    self.display.blit(image, (0, 0))

            if not self.lidar_queue.empty():
                frame, points = self.lidar_queue.get()
                lidar_surface = pygame.Surface((400, 400))
                lidar_surface.fill((0, 0, 0))

                if points.size > 0 and not np.isnan(points).any():
                    points_2d = points[:, :2]
                    points_2d = points_2d * 10
                    points_2d += np.array([200, 200])

                    mask = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < 400) &
                           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < 400))
                    points_2d = points_2d[mask]

                    if points_2d.size > 0:
                        heights = points[mask, 2]
                        heights_norm = np.clip((heights - np.min(heights)) /
                                             (np.max(heights) - np.min(heights) + 1e-6), 0, 1)

                        total = len(points_2d)
                        if total > self._lidar_display_points:
                            indices = np.linspace(0, total - 1, self._lidar_display_points, dtype=int)
                            points_2d = points_2d[indices]
                            heights_norm = heights_norm[indices]

                        colors_r = (255 * heights_norm).astype(int)
                        colors_b = (255 * (1 - heights_norm)).astype(int)

                        for i in range(len(points_2d)):
                            x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
                            color = (colors_r[i], 0, colors_b[i])
                            lidar_surface.set_at((x, y), color)

                self.display.blit(lidar_surface, (0, self.display.get_height() - 400))

            if not self.radar_queue.empty():
                frame, radar_data = self.radar_queue.get()
                radar_surface = pygame.Surface((200, 200))
                radar_surface.fill((0, 0, 0))

                if radar_data.size > 0 and not np.isnan(radar_data).any():
                    for detection in radar_data:
                        velocity, azimuth, altitude, depth = detection
                        if not any(np.isnan([velocity, azimuth, altitude, depth])):
                            x = depth * math.cos(azimuth) * 2
                            y = depth * math.sin(azimuth) * 2

                            screen_x = int(100 + x)
                            screen_y = int(100 + y)

                            if 0 <= screen_x < 200 and 0 <= screen_y < 200:
                                color = (255, 0, 0) if velocity < 0 else (0, 255, 0)
                                pygame.draw.circle(radar_surface, color,
                                                (screen_x, screen_y), 3)

                self.display.blit(radar_surface, (self.display.get_width() - 200,
                                                self.display.get_height() - 200))

            try:
                text_color = (255, 255, 255)
                current_frame = len(self.sensor_data.get('camera', []))

                if current_frame != self._text_cache_frame:
                    self._text_cache_frame = current_frame
                    self._text_cache = {
                        'camera': self._font.render(f"Camera Frames: {len(self.sensor_data['camera'])}", True, text_color),
                        'lidar': self._font.render(f"LiDAR Frames: {len(self.sensor_data['lidar'])}", True, text_color),
                        'radar': self._font.render(f"Radar Frames: {len(self.sensor_data['radar'])}", True, text_color),
                        'viz_fps': self._font.render(f"Viz FPS: {self._viz_fps}", True, (200, 200, 0)),
                    }

                self.display.blit(self._text_cache['camera'], (self.display.get_width() - 300, 10))
                self.display.blit(self._text_cache['lidar'], (self.display.get_width() - 300, 50))
                self.display.blit(self._text_cache['radar'], (self.display.get_width() - 300, 90))
                self.display.blit(self._text_cache['viz_fps'], (self.display.get_width() - 300, 130))

            except Exception as e:
                logging.warning(f"Failed to render text overlay: {str(e)}")

            pygame.display.flip()

        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
class WeatherScheduler:
    WEATHER_PRESETS = {
        'clear_noon': {
            'cloudiness': 10.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0,
            'wind_intensity': 5.0, 'fog_density': 0.0, 'wetness': 0.0,
            'sun_altitude_angle': 75.0
        },
        'cloudy': {
            'cloudiness': 70.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0,
            'wind_intensity': 15.0, 'fog_density': 5.0, 'wetness': 5.0,
            'sun_altitude_angle': 60.0
        },
        'light_rain': {
            'cloudiness': 50.0, 'precipitation': 30.0, 'precipitation_deposits': 15.0,
            'wind_intensity': 20.0, 'fog_density': 8.0, 'wetness': 30.0,
            'sun_altitude_angle': 55.0
        },
        'heavy_rain': {
            'cloudiness': 100.0, 'precipitation': 100.0, 'precipitation_deposits': 100.0,
            'wind_intensity': 60.0, 'fog_density': 15.0, 'wetness': 100.0,
            'sun_altitude_angle': 40.0
        },
        'storm': {
            'cloudiness': 100.0, 'precipitation': 100.0, 'precipitation_deposits': 100.0,
            'wind_intensity': 100.0, 'fog_density': 40.0, 'wetness': 100.0,
            'sun_altitude_angle': 15.0
        },
        'fog_morning': {
            'cloudiness': 80.0, 'precipitation': 5.0, 'precipitation_deposits': 0.0,
            'wind_intensity': 5.0, 'fog_density': 85.0, 'wetness': 15.0,
            'sun_altitude_angle': 20.0
        },
        'fog_dense': {
            'cloudiness': 95.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0,
            'wind_intensity': 3.0, 'fog_density': 100.0, 'wetness': 10.0,
            'sun_altitude_angle': 10.0
        },
        'sunset': {
            'cloudiness': 20.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0,
            'wind_intensity': 10.0, 'fog_density': 3.0, 'wetness': 0.0,
            'sun_altitude_angle': 5.0
        },
    }

    SCENARIOS = {
        'commute_to_work': {
            'name': 'Commute to Work',
            'timeline': [
                (0, 'clear_noon'),
                (0.15, 'cloudy'),
                (0.30, 'light_rain'),
                (0.50, 'heavy_rain'),
                (0.70, 'fog_morning'),
                (0.85, 'cloudy'),
                (1.0, 'clear_noon'),
            ],
            'transition_duration': 0.05,
        },
        'storm_front': {
            'name': 'Storm Front',
            'timeline': [
                (0, 'clear_noon'),
                (0.10, 'cloudy'),
                (0.20, 'light_rain'),
                (0.35, 'heavy_rain'),
                (0.50, 'storm'),
                (0.65, 'heavy_rain'),
                (0.80, 'light_rain'),
                (0.90, 'cloudy'),
                (1.0, 'clear_noon'),
            ],
            'transition_duration': 0.04,
        },
        'fog_dissipation': {
            'name': 'Fog Dissipation',
            'timeline': [
                (0, 'fog_dense'),
                (0.20, 'fog_morning'),
                (0.40, 'cloudy'),
                (0.55, 'light_rain'),
                (0.70, 'cloudy'),
                (0.85, 'clear_noon'),
                (1.0, 'sunset'),
            ],
            'transition_duration': 0.05,
        },
        'clear_to_storm': {
            'name': 'Clear to Storm',
            'timeline': [
                (0, 'clear_noon'),
                (0.25, 'cloudy'),
                (0.50, 'light_rain'),
                (0.75, 'storm'),
                (1.0, 'heavy_rain'),
            ],
            'transition_duration': 0.06,
        },
    }

    RANDOM_EVENTS = [
        {'name': 'Sudden Gust', 'preset': 'storm', 'duration_pct': 0.05, 'probability': 0.002},
        {'name': 'Fog Patch', 'preset': 'fog_morning', 'duration_pct': 0.08, 'probability': 0.001},
        {'name': 'Rain Burst', 'preset': 'heavy_rain', 'duration_pct': 0.04, 'probability': 0.003},
    ]

    def __init__(self, scenario_name='storm_front', duration=60, enable_random_events=True):
        self.scenario_name = scenario_name
        self.duration = duration
        self.enable_random_events = enable_random_events
        self.scenario = self.SCENARIOS.get(scenario_name, self.SCENARIOS['storm_front'])
        self.timeline = self.scenario['timeline']
        self.transition_duration = self.scenario['transition_duration']
        self.current_phase = self.timeline[0][1]
        self.next_phase = self.timeline[0][1]
        self.phase_progress = 0.0
        self.current_params = dict(self.WEATHER_PRESETS[self.timeline[0][1]])
        self.active_random_event = None
        self.random_event_start = 0.0
        self.random_event_end = 0.0
        self.random_event_base_params = None
        self._last_logged_phase = None

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_params(self, params_a, params_b, t):
        result = {}
        for key in params_a:
            result[key] = round(self._lerp(params_a[key], params_b.get(key, params_a[key]), t), 1)
        return result

    def _get_timeline_segment(self, progress):
        for i in range(len(self.timeline) - 1):
            start_pct, start_preset = self.timeline[i]
            end_pct, end_preset = self.timeline[i + 1]
            if start_pct <= progress < end_pct:
                segment_progress = (progress - start_pct) / (end_pct - start_pct) if end_pct > start_pct else 1.0
                return start_preset, end_preset, segment_progress
        last_preset = self.timeline[-1][1]
        return last_preset, last_preset, 1.0

    def _check_random_event(self, progress, elapsed):
        if not self.enable_random_events or self.active_random_event is not None:
            return
        for event in self.RANDOM_EVENTS:
            if random.random() < event['probability']:
                self.active_random_event = event
                self.random_event_start = progress
                self.random_event_end = min(progress + event['duration_pct'], 1.0)
                self.random_event_base_params = dict(self.current_params)
                logging.info(f"Weather random event: {event['name']} (until {self.random_event_end*100:.0f}%)")
                break

    def _apply_random_event(self, base_params, progress):
        if self.active_random_event is None:
            return base_params
        if progress >= self.random_event_end:
            logging.info(f"Random event ended: {self.active_random_event['name']}")
            self.active_random_event = None
            self.random_event_base_params = None
            return base_params
        event_progress = (progress - self.random_event_start) / (self.random_event_end - self.random_event_start)
        event_t = math.sin(event_progress * math.pi)
        event_params = self.WEATHER_PRESETS[self.active_random_event['preset']]
        return self._lerp_params(base_params, event_params, event_t * 0.7)

    def update(self, elapsed):
        progress = min(elapsed / self.duration, 1.0) if self.duration > 0 else 0.0
        self._check_random_event(progress, elapsed)
        start_preset, end_preset, segment_progress = self._get_timeline_segment(progress)
        self.current_phase = start_preset
        self.next_phase = end_preset
        self.phase_progress = segment_progress
        start_params = self.WEATHER_PRESETS[start_preset]
        end_params = self.WEATHER_PRESETS[end_preset]
        trans_start = 1.0 - self.transition_duration / max(self._get_segment_duration(progress), 0.001)
        if segment_progress < trans_start:
            self.current_params = dict(start_params)
        else:
            t = (segment_progress - trans_start) / (1.0 - trans_start) if trans_start < 1.0 else 1.0
            self.current_params = self._lerp_params(start_params, end_params, t)
        self.current_params = self._apply_random_event(self.current_params, progress)
        if self.current_phase != self._last_logged_phase:
            logging.info(f"Weather phase: {start_preset} -> {end_preset} ({progress*100:.0f}%)")
            self._last_logged_phase = self.current_phase
        return self.current_params

    def _get_segment_duration(self, progress):
        for i in range(len(self.timeline) - 1):
            start_pct, _ = self.timeline[i]
            end_pct, _ = self.timeline[i + 1]
            if start_pct <= progress < end_pct:
                return (end_pct - start_pct) * self.duration
        return self.duration

    def get_phase_label(self):
        phase_names = {
            'clear_noon': 'Clear', 'cloudy': 'Cloudy', 'light_rain': 'Light Rain',
            'heavy_rain': 'Heavy Rain', 'storm': 'Storm', 'fog_morning': 'Fog',
            'fog_dense': 'Dense Fog', 'sunset': 'Sunset',
        }
        current = phase_names.get(self.current_phase, self.current_phase)
        next_name = phase_names.get(self.next_phase, self.next_phase)
        if self.active_random_event:
            return f"{current}->{next_name} [{self.active_random_event['name']}]"
        if self.current_phase == self.next_phase:
            return current
        return f"{current}->{next_name}"

    def get_scenario_name(self):
        return self.scenario['name']

    def get_progress_pct(self):
        return self.phase_progress * 100


class SimulationDashboard:
    def __init__(self, duration, weather_name='rain', mode='quick'):
        self.duration = duration
        self.weather_name = weather_name
        self.mode = mode
        self.weather_intensity = 1.0
        self.weather_phase = ''
        self.paused = False
        self.debug_mode = False
        self.quit_requested = False
        self.elapsed = 0.0
        self.fps = 0.0
        self.memory_mb = 0.0
        self.frame_count = 0
        self.camera_frames = 0
        self.lidar_frames = 0
        self.radar_frames = 0
        self.semantic_frames = 0
        self.depth_frames = 0
        self.vehicles_count = 0
        self.pedestrians_count = 0
        self.queue_sizes = {}
        self._console = Console(force_terminal=True) if RICH_AVAILABLE else None
        self._live = None
        self._status = "Running"

    def start(self):
        if not RICH_AVAILABLE:
            return
        self._live = Live(
            self._build_panel(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._status = "PAUSED" if self.paused else "Running"
        if self._live:
            self._live.update(self._build_panel())

    def check_keyboard(self):
        try:
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch()
                self._handle_key(key)
        except ImportError:
            pass

    def _handle_key(self, key):
        if key == b' ':
            self.paused = not self.paused
            self._status = "PAUSED" if self.paused else "Running"
        elif key in (b'w', b'W'):
            self.weather_intensity = min(1.0, round(self.weather_intensity + 0.1, 1))
        elif key in (b's', b'S'):
            self.weather_intensity = max(0.0, round(self.weather_intensity - 0.1, 1))
        elif key in (b'q', b'Q'):
            self.quit_requested = True
            self._status = "Stopping..."
        elif key in (b'd', b'D'):
            self.debug_mode = not self.debug_mode
        if self._live:
            self._live.update(self._build_panel())

    def get_scaled_weather_params(self, base_params):
        scaled = {}
        for key, value in base_params.items():
            if key == 'sun_altitude_angle':
                scaled[key] = value
            else:
                scaled[key] = round(value * self.weather_intensity, 1)
        return scaled

    def _get_weather_label(self):
        i = self.weather_intensity
        if i <= 0.05:
            return "Clear"
        elif i <= 0.25:
            return "Light Rain"
        elif i <= 0.5:
            return "Moderate Rain"
        elif i <= 0.75:
            return "Heavy Rain"
        else:
            return "Storm"

    def _build_panel(self):
        if not RICH_AVAILABLE:
            return ""

        table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
        table.add_column(justify="left", no_wrap=True)

        status_style = "bold green" if not self.paused else "bold yellow"
        status_icon = ">" if not self.paused else "||"

        table.add_row(f"[bold cyan]CARLA AV Simulation[/]  [{status_style}]{status_icon} {self._status}[/]")
        table.add_row("[dim]──────────────────────────────────────────[/]")

        progress_pct = min(self.elapsed / self.duration, 1.0) if self.duration > 0 else 0
        bar_len = 20
        filled = int(bar_len * progress_pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        table.add_row(f"  Progress: [cyan][{bar}][/]{progress_pct*100:5.1f}%  {self.elapsed:.1f}/{self.duration}s")

        table.add_row(f"  FPS: [cyan]{self.fps:6.1f}[/] | Memory: [cyan]{self.memory_mb:.0f}MB[/]")

        if self.weather_phase:
            table.add_row(f"  Weather: [cyan]{self.weather_phase}[/]")
        else:
            weather_label = self._get_weather_label()
            table.add_row(f"  Weather: [cyan]{weather_label} ({self.weather_intensity*100:.0f}%)[/]")

        sensor_parts = [f"Camera: [cyan]{self.camera_frames}[/]"]
        sensor_parts.append(f"LiDAR: [cyan]{self.lidar_frames}[/]")
        if self.radar_frames > 0:
            sensor_parts.append(f"Radar: [cyan]{self.radar_frames}[/]")
        table.add_row("  " + " | ".join(sensor_parts))

        table.add_row("[dim]──────────────────────────────────────────[/]")
        table.add_row("[dim][Space] Pause  [W/S] Weather  [Q] Quit  [D] Debug[/]")

        if self.debug_mode:
            table.add_row("[dim]──────────────────────────────────────────[/]")
            debug_parts = [f"Frame: {self.frame_count}"]
            debug_parts.append(f"Vehicles: {self.vehicles_count}")
            debug_parts.append(f"Pedestrians: {self.pedestrians_count}")
            table.add_row(f"[dim]{' | '.join(debug_parts)}[/]")
            if self.queue_sizes:
                qs = " | ".join(f"{k}: {v}" for k, v in self.queue_sizes.items())
                table.add_row(f"[dim]Queues: {qs}[/]")
            table.add_row(f"[dim]Mode: {self.mode} | Weather base: {self.weather_name}[/]")

        return Panel(table, border_style="cyan", padding=(1, 2), title="[bold]Dashboard[/]", title_align="left")


def main():
    simulation = None
    try:
        simulation = AVSimulation()
        configs = ['minimal', 'standard', 'advanced']
        sensor_desc = {
            'minimal': 'RGB Camera + LiDAR',
            'standard': 'RGB Camera + LiDAR + Radar',
            'advanced': 'RGB + Semantic + Depth + LiDAR + Radar'
        }

        if RICH_AVAILABLE:
            console = Console(force_terminal=True)
            table = Table(title="Sensor Configurations", show_header=True)
            table.add_column("Option", style="cyan", justify="center")
            table.add_column("Configuration", style="green")
            table.add_column("Sensors", style="yellow")
            for i, config in enumerate(configs):
                table.add_row(str(i + 1), config, sensor_desc.get(config, ''))
            table.add_row("0", "Exit", "")
            console.print(table)
        else:
            print("\nAvailable sensor configurations:")
            for i, config in enumerate(configs):
                print(f"{i+1}. {config}")

        while True:
            try:
                choice = input("\nSelect configuration (1-3) or 0 to exit: ")
                if not choice.strip():
                    continue

                choice = int(choice)
                if choice == 0:
                    break
                if 1 <= choice <= len(configs):
                    logging.info(f"Starting simulation with {configs[choice-1]} configuration...")
                    simulation.run_simulation(configs[choice-1])
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                logging.info("Simulation interrupted by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                break

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        traceback.print_exc()

    finally:
        if simulation is not None:
            simulation.cleanup_actors()
        pygame.quit()
        logging.info("Simulation ended")

if __name__ == '__main__':
    main()
