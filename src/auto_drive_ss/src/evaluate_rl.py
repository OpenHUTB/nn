import carla
import numpy as np
import torch
import queue
import time
import os
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import logging
import json
from pathlib import Path

# Import local modules
from RL_main_baseline import CarlaRLEnv, Parameters, EpisodeStats
from src.models.perception_system import PerceptionSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='evaluation.log'
)
logger = logging.getLogger('carla_evaluation')

class EvaluationMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = 0
        self.collision_rate = 0
        self.off_road_rate = 0
        self.avg_speed = 0
        self.avg_lane_deviation = 0
        self.waypoints_reached = 0
        self.total_distance = 0  # Add total distance tracking
        self.total_episodes = 0
        
    def update(self, stats, termination_reason):
        summary = stats.get_summary()
        self.episode_rewards.append(summary['total_reward'])
        self.episode_lengths.append(summary['steps'])
        self.avg_speed += summary['avg_speed']
        self.avg_lane_deviation += summary['avg_lane_deviation']
        self.waypoints_reached += summary['waypoints_reached']
        self.total_distance += summary['total_distance']  # Update total distance
        self.total_episodes += 1
        
        if termination_reason == 'success':
            self.success_rate += 1
        elif termination_reason == 'collision':
            self.collision_rate += 1
        elif termination_reason == 'off_road':
            self.off_road_rate += 1
            
    def get_summary(self):
        if self.total_episodes == 0:
            return {}
            
        return {
            'avg_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'avg_episode_length': np.mean(self.episode_lengths),
            'success_rate': self.success_rate / self.total_episodes * 100,
            'collision_rate': self.collision_rate / self.total_episodes * 100,
            'off_road_rate': self.off_road_rate / self.total_episodes * 100,
            'avg_speed': self.avg_speed / self.total_episodes,
            'avg_lane_deviation': self.avg_lane_deviation / self.total_episodes,
            'avg_waypoints_reached': self.waypoints_reached / self.total_episodes,
            'avg_distance': self.total_distance / self.total_episodes  # Add average distance to summary
        }

def evaluate_model(checkpoint_path, num_episodes=50, towns=['Town01', 'Town02'], 
                  weather_conditions=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon']):
    """Evaluate a trained model across different towns and weather conditions."""
    
    # Load parameters
    params = Parameters()
    
    # Initialize metrics
    metrics = {town: {weather: EvaluationMetrics() for weather in weather_conditions}
              for town in towns}
    
    # Create output directory for evaluation results
    eval_dir = Path('evaluation_results')
    eval_dir.mkdir(exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=str(eval_dir / 'tensorboard'))
    
    # Load perception system
    perception = PerceptionSystem(
        feature_dim=params.FEATURE_DIM,
        fusion_dim=params.FUSION_DIM,
        output_dim=params.OUTPUT_DIM,
        num_frames=params.NUM_FRAMES
    ).to(device)
    perception.eval()
    
    # Load trained model
    model = PPO.load(checkpoint_path)
    
    for town in towns:
        for weather in weather_conditions:
            logger.info(f"\nEvaluating on {town} with {weather} conditions")
            
            # Create environment
            env = CarlaRLEnv(town=town)
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(f"{os.path.dirname(checkpoint_path)}/vec_normalize.pkl", env)
            env.training = False
            env.norm_reward = False
            
            # Set weather
            weather_preset = getattr(carla.WeatherParameters, weather)
            env.get_attr('world')[0].set_weather(weather_preset)
            
            for episode in range(num_episodes):
                obs = env.reset()
                done = False
                episode_stats = EpisodeStats()
                termination_reason = 'incomplete'
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    
                    # Update episode statistics
                    episode_stats.add_step_info(
                        reward=reward[0],
                        speed=info[0].get('speed', 0),
                        distance=info[0].get('distance', 0),
                        lane_deviation=info[0].get('lane_deviation', 0),
                        waypoint_reached=info[0].get('waypoint_reached', False)
                    )
                    
                    if done:
                        if info[0].get('collision', False):
                            termination_reason = 'collision'
                        elif info[0].get('off_road', False):
                            termination_reason = 'off_road'
                        elif episode_stats.waypoints_reached >= params.WAYPOINT_ROUTE_LENGTH:
                            termination_reason = 'success'
                
                # Update metrics
                metrics[town][weather].update(episode_stats, termination_reason)
                episode_stats.print_summary(episode, termination_reason)
                
                # Log to tensorboard
                writer.add_scalar(f"{town}/{weather}/episode_reward", 
                                episode_stats.total_reward, episode)
            
            # Save town/weather specific results
            results = metrics[town][weather].get_summary()
            with open(eval_dir / f"{town}_{weather}_results.json", 'w') as f:
                json.dump(results, f, indent=4)
    
    # Aggregate and save overall results
    overall_results = {
        'towns': {},
        'weather_conditions': {},
        'overall': {
            'success_rate': [],
            'collision_rate': [],
            'avg_episode_reward': []
        }
    }
    
    for town in towns:
        town_metrics = [metrics[town][w].get_summary() for w in weather_conditions]
        overall_results['towns'][town] = {
            'success_rate': np.mean([m['success_rate'] for m in town_metrics]),
            'collision_rate': np.mean([m['collision_rate'] for m in town_metrics]),
            'avg_episode_reward': np.mean([m['avg_episode_reward'] for m in town_metrics])
        }
    
    for weather in weather_conditions:
        weather_metrics = [metrics[t][weather].get_summary() for t in towns]
        overall_results['weather_conditions'][weather] = {
            'success_rate': np.mean([m['success_rate'] for m in weather_metrics]),
            'collision_rate': np.mean([m['collision_rate'] for m in weather_metrics]),
            'avg_episode_reward': np.mean([m['avg_episode_reward'] for m in weather_metrics])
        }
    
    # Save overall results
    with open(eval_dir / 'overall_results.json', 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    writer.close()
    env.close()
    
    return overall_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained RL model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of evaluation episodes per condition')
    parser.add_argument('--towns', nargs='+', default=['Town01', 'Town02'],
                        help='List of towns to evaluate on')
    parser.add_argument('--weather', nargs='+', 
                        default=['ClearNoon', 'ClearSunset', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon'],
                        help='List of weather conditions to evaluate on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        towns=args.towns,
        weather_conditions=args.weather
    )
    
    # Print summary
    logger.info("\nEvaluation Complete! Summary of results:")
    logger.info("\nPer-Town Performance:")
    for town, metrics in results['towns'].items():
        logger.info(f"\n{town}:")
        logger.info(f"  Success Rate: {metrics['success_rate']:.2f}%")
        logger.info(f"  Collision Rate: {metrics['collision_rate']:.2f}%")
        logger.info(f"  Off-Road Rate: {metrics['off_road_rate']:.2f}%")
        logger.info(f"  Average Episode Reward: {metrics['avg_episode_reward']:.2f}")
        logger.info(f"  Average Speed: {metrics['avg_speed']:.2f} km/h")
        logger.info(f"  Average Lane Deviation: {metrics['avg_lane_deviation']:.2f} m")
        logger.info(f"  Average Waypoints Reached: {metrics['avg_waypoints_reached']:.2f}")
        logger.info(f"  Average Distance Traveled: {metrics['avg_distance']:.2f} m")
    
    logger.info("\nPer-Weather Performance:")
    for weather, metrics in results['weather_conditions'].items():
        logger.info(f"\n{weather}:")
        logger.info(f"  Success Rate: {metrics['success_rate']:.2f}%")
        logger.info(f"  Collision Rate: {metrics['collision_rate']:.2f}%")
        logger.info(f"  Off-Road Rate: {metrics['off_road_rate']:.2f}%")
        logger.info(f"  Average Episode Reward: {metrics['avg_episode_reward']:.2f}")
        logger.info(f"  Average Speed: {metrics['avg_speed']:.2f} km/h")
        logger.info(f"  Average Lane Deviation: {metrics['avg_lane_deviation']:.2f} m")
        logger.info(f"  Average Waypoints Reached: {metrics['avg_waypoints_reached']:.2f}")
        logger.info(f"  Average Distance Traveled: {metrics['avg_distance']:.2f} m")