from __future__ import annotations

import csv
import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional


class DataLogger:
    """Data logger for recording simulation data during runs.

    This class handles recording of simulation data including:
    - Time series data: time, error, steering angle, throttle, detection surface area
    - Metadata: controller type, PID parameters, map name, run ID
    """

    def __init__(self, folder: str, run_id: str) -> None:
        """Constructor

        Args:
            folder (str): Path to the folder where data will be saved.
            run_id (str): Unique identifier for this run.
        """
        self.folder = folder
        self.run_id = run_id
        self.data: List[Dict] = []
        self.metadata: Dict = {}
        self.start_time = datetime.now()

        if not os.path.exists(folder):
            os.makedirs(folder)

    def set_metadata(self, **kwargs) -> None:
        """Set metadata for the run.

        Args:
            **kwargs: Metadata key-value pairs (e.g., controller, pid_params, map_name)
        """
        self.metadata.update(kwargs)
        self.metadata['run_id'] = self.run_id
        self.metadata['start_time'] = self.start_time.isoformat()

    def record_step(self, step: int, error: float, steer: float, 
                    throttle: float, detection_surface_area: float) -> None:
        """Record data for one simulation step.

        Args:
            step (int): Step number.
            error (float): Difference to the center of the detected lane.
            steer (float): Steering angle applied.
            throttle (float): Throttle value applied.
            detection_surface_area (float): Detected surface area.
        """
        self.data.append({
            'step': step,
            'time': (datetime.now() - self.start_time).total_seconds(),
            'error': error,
            'steer': steer,
            'throttle': throttle,
            'detection_surface_area': detection_surface_area
        })

    def save_csv(self, filename: Optional[str] = None) -> str:
        """Save recorded data to a CSV file.

        Args:
            filename (Optional[str]): Custom filename. Defaults to {run_id}_data.csv.

        Returns:
            str: Path to the saved CSV file.
        """
        if not filename:
            filename = f'{self.run_id}_data.csv'
        
        filepath = os.path.join(self.folder, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if self.data:
                fieldnames = self.data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data)
        
        return filepath

    def save_npy(self, filename: Optional[str] = None) -> str:
        """Save recorded data to a NumPy file.

        Args:
            filename (Optional[str]): Custom filename. Defaults to {run_id}_data.npy.

        Returns:
            str: Path to the saved NumPy file.
        """
        if not filename:
            filename = f'{self.run_id}_data.npy'
        
        filepath = os.path.join(self.folder, filename)
        
        if self.data:
            structured_data = {
                'steps': np.array([d['step'] for d in self.data]),
                'times': np.array([d['time'] for d in self.data]),
                'errors': np.array([d['error'] for d in self.data]),
                'steers': np.array([d['steer'] for d in self.data]),
                'throttles': np.array([d['throttle'] for d in self.data]),
                'detection_areas': np.array([d['detection_surface_area'] for d in self.data])
            }
            np.save(filepath, structured_data)
        
        return filepath

    def save_metadata(self, filename: Optional[str] = None) -> str:
        """Save metadata to a JSON file.

        Args:
            filename (Optional[str]): Custom filename. Defaults to {run_id}_metadata.json.

        Returns:
            str: Path to the saved JSON file.
        """
        if not filename:
            filename = f'{self.run_id}_metadata.json'
        
        filepath = os.path.join(self.folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)
        
        return filepath

    def save_all(self) -> None:
        """Save all recorded data and metadata."""
        self.save_csv()
        self.save_npy()
        self.save_metadata()
        print(f"Data saved to {self.folder}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the recorded data.

        Returns:
            Dict: Summary statistics including mean, std, max, min for key metrics.
        """
        if not self.data:
            return {}
        
        errors = np.array([d['error'] for d in self.data])
        steers = np.array([d['steer'] for d in self.data])
        
        return {
            'total_steps': len(self.data),
            'duration': self.data[-1]['time'] if self.data else 0,
            'error_mean': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'error_max': float(np.max(errors)),
            'error_min': float(np.min(errors)),
            'error_rmse': float(np.sqrt(np.mean(errors ** 2))),
            'steer_mean': float(np.mean(steers)),
            'steer_std': float(np.std(steers)),
            'steer_max': float(np.max(steers)),
            'steer_min': float(np.min(steers))
        }

    def save_summary(self, filename: Optional[str] = None) -> str:
        """Save summary statistics to a JSON file.

        Args:
            filename (Optional[str]): Custom filename. Defaults to {run_id}_summary.json.

        Returns:
            str: Path to the saved JSON file.
        """
        if not filename:
            filename = f'{self.run_id}_summary.json'
        
        filepath = os.path.join(self.folder, filename)
        summary = self.get_summary_stats()
        summary.update(self.metadata)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        
        return filepath
