import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from config import Config

class NavigationVisualizer:
    def __init__(self):
        self.config = Config()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.path_x = []
        self.path_y = []
        self.rewards = []
        self.distances = []
        self.episode_rewards = []
        
    def init_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1, self.config.MAP_WIDTH + 1)
        self.ax.set_ylim(-1, self.config.MAP_HEIGHT + 1)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Robot Navigation Simulation')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        self.obstacle_patches = []
        self.robot_patch = None
        self.target_patch = None
        self.path_line = None
        self.lidar_lines = []
        
    def draw_obstacles(self, obstacles):
        for obs in obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
            self.ax.add_patch(circle)
            self.obstacle_patches.append(circle)
    
    def draw_robot(self, x, y, theta):
        if self.robot_patch:
            self.robot_patch.remove()
        
        robot_body = plt.Circle((x, y), self.config.ROBOT_RADIUS, color='blue')
        self.ax.add_patch(robot_body)
        
        arrow_length = self.config.ROBOT_RADIUS * 1.5
        arrow_x = x + arrow_length * np.cos(theta)
        arrow_y = y + arrow_length * np.sin(theta)
        self.ax.arrow(x, y, (arrow_x - x) * 0.8, (arrow_y - y) * 0.8, 
                     head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        self.robot_patch = robot_body
    
    def draw_target(self, x, y):
        if self.target_patch:
            self.target_patch.remove()
        
        target = plt.Circle((x, y), 0.4, color='green', alpha=0.7)
        self.ax.add_patch(target)
        self.target_patch = target
    
    def draw_path(self, x, y):
        self.path_x.append(x)
        self.path_y.append(y)
        
        if self.path_line:
            self.path_line.remove()
        
        if len(self.path_x) > 1:
            self.path_line, = self.ax.plot(self.path_x, self.path_y, 'g--', linewidth=1.5)
    
    def draw_lidar(self, x, y, theta, lidar_data):
        for line in self.lidar_lines:
            line.remove()
        self.lidar_lines.clear()
        
        for i, dist in enumerate(lidar_data):
            angle = np.radians(i) + theta
            end_x = x + dist * np.cos(angle)
            end_y = y + dist * np.sin(angle)
            line, = self.ax.plot([x, end_x], [y, end_y], 'gray', linewidth=0.5, alpha=0.3)
            self.lidar_lines.append(line)
    
    def update_reward_plot(self, reward, distance):
        self.rewards.append(reward)
        self.distances.append(distance)
    
    def show(self):
        plt.show(block=False)
        plt.pause(0.001)
    
    def save_figure(self, filename='navigation_result.png'):
        if not os.path.exists(self.config.RESULT_DIR):
            os.makedirs(self.config.RESULT_DIR)
        plt.savefig(os.path.join(self.config.RESULT_DIR, filename), dpi=150)
    
    def plot_training_history(self, rewards, distances, episode_lengths):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Reward History')
        axes[0].grid(True)
        
        axes[1].plot(distances)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Final Distance to Target')
        axes[1].set_title('Distance to Target')
        axes[1].grid(True)
        
        axes[2].plot(episode_lengths)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Episode Length')
        axes[2].set_title('Episode Length History')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if not os.path.exists(self.config.RESULT_DIR):
            os.makedirs(self.config.RESULT_DIR)
        plt.savefig(os.path.join(self.config.RESULT_DIR, 'training_history.png'), dpi=150)
        plt.show()
    
    def plot_lidar_heatmap(self, lidar_data):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        theta = np.linspace(0, 2*np.pi, len(lidar_data))
        ax.plot(theta, lidar_data)
        ax.set_rmax(self.config.LIDAR_RANGE)
        ax.set_title('Lidar Data Visualization')
        
        if not os.path.exists(self.config.RESULT_DIR):
            os.makedirs(self.config.RESULT_DIR)
        plt.savefig(os.path.join(self.config.RESULT_DIR, 'lidar_heatmap.png'), dpi=150)
        plt.show()
    
    def reset_path(self):
        self.path_x = []
        self.path_y = []
        self.path_line = None