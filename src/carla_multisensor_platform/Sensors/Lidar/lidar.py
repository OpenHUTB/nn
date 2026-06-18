import carla
import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor

class LidarManager:
    def __init__(self, min_dist=0.5, max_dist=50.0):
        self.points = None
        self.min_dist = min_dist
        self.max_dist = max_dist
        
    def filter_points_by_range(self, points):
        distances = np.linalg.norm(points[:, 3], axis=1)
        mask = (distances > self.min_dist) & (distances < self.max_dist)
        return points[mask]
    
    def normalize_points(points):
        norm_points = np.copy(points)
        norm_points[:, :3] = (norm_points[:, :3] - norm_points[:, :3].min(axis=0)) / (norm_points[:, :3].ptp(axis=0) + 1e-8)
        norm_points[:, 3] = norm_points[:, 3] / (norm_points[:, 3].max() + 1e-8)
        return norm_points
        
    def open3d_visualizer(points):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])
        
        if points.shape[1] >= 4:
            intensity = points[:, 3]
            colors = (intensity / 255.0).reshape(-1, 1)
            colors = np.repeat(colors, 3, axis=1)
            pc.colors = o3d.utility.Vector3dVector(colors)
            
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Open3D PointCloud", width=960, height=540)
        vis.add_geometry(pc)
        vis.run()
        vis.destroy_window()