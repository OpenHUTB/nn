import carla
import numpy as np
import torch
import queue
import random
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
import re
import glob
from collections import deque
import logging
import argparse


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='training.log'
)
logger = logging.getLogger('carla_training')

# Set number of threads based on HPC allocation
if 'OMP_NUM_THREADS' in os.environ:
    num_threads = int(os.environ['OMP_NUM_THREADS'])
    torch.set_num_threads(num_threads)
    logger.info(f"Setting torch threads to {num_threads}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

low_speed_timer = 0
max_distance = 3.0  # Max distance from center before terminating
target_speed = 20.0  # km/h

# Import perception system
from src.models.perception_system import PerceptionSystem
# Define parameters
class Parameters:
    # Environment settings
    TOWN = "Town01"
    SYNC_MODE = True
    FIXED_DELTA_SECONDS = 0.05  # ~33 FPS for smoother visualization
    SENSOR_WIDTH = 640
    SENSOR_HEIGHT = 480
    RENDER_PRIORITY = False
    RENDER = False  # Set this to False for headless operation
    
    # Perception settings
    FEATURE_DIM = 256
    FUSION_DIM = 256
    OUTPUT_DIM = 256
    NUM_FRAMES = 5
    
    # SSL Checkpoint path - NEW PARAMETER
    SSL_CHECKPOINT_PATH = "checkpoints/ssl/pretrained_perception.pth"
    
    # Training settings
    TOTAL_TIMESTEPS = 20000000
    LEARNING_RATE = 3e-4
    N_STEPS = 2048
    BATCH_SIZE = 128
    N_EPOCHS = 6
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    MAX_EPISODE_STEPS = 2000
    SEED = 42
    
    # Reward settings
    COLLISION_PENALTY = -10.0  # Reduced from -60.0
    COLLISION_GRACE_FRAMES = 30  # Grace period before collision penalty applies
    COLLISION_WARNING_THRESHOLD = 5  # Warn after this many collision events
    LANE_KEEPING_REWARD = 1.0
    LANE_DEVIATION_PENALTY = -0.5  # Reduced from -1.0
    OFF_ROAD_PENALTY = -15.0  # Reduced from -40.0
    OFF_ROAD_GRACE_STEPS = 30  # Grace steps before off-road penalty
    STABLE_STEERING_REWARD = 0.5
    SPEED_REWARD_FACTOR = 0.3
    OPTIMAL_SPEED_MIN = 5.0
    OPTIMAL_SPEED_MAX = 35.0
    RED_LIGHT_PENALTY = -3.0  # Reduced from -5.0
    RED_LIGHT_STOP_REWARD = 5.0
    STOPPED_GRACE_FRAMES_EARLY = 400  # Increased from 300
    STOPPED_GRACE_FRAMES_MID = 300  # Increased from 200
    STOPPED_GRACE_FRAMES_NORMAL = 200  # Increased from 150
    
    # Waypoint settings
    WAYPOINT_ROUTE_LENGTH = 100
    WAYPOINT_DISTANCE = 2.0
    
    # Checkpoints and visualization
    CHECKPOINT_DIR = "./checkpoints/trained_ssl_rl"
    TENSORBOARD_LOG = "./tensorboard_logs/trained_ssl_rl"
    CHECKPOINT_FREQ = 10  
    EVAL_FREQ = 5000

params = Parameters()

class EpisodeStats:
    """Simple helper class to track per-episode statistics"""
    def __init__(self):
        self.total_reward = 0
        self.steps = 0
        self.start_time = time.time()
        self.speeds = []
        self.distances = []
        self.lane_deviations = []
        self.waypoints_reached = 0
        self.collision_count = 0
        self.off_road_count = 0
        self.stopped_count = 0

    def add_step_info(self, reward, speed, distance, lane_deviation, waypoint_reached=False, collision=False, off_road=False):
        self.total_reward += reward
        self.steps += 1
        self.speeds.append(speed)
        self.distances.append(distance)
        self.lane_deviations.append(lane_deviation)
        if waypoint_reached:
            self.waypoints_reached += 1
        if collision:
            self.collision_count += 1
        if off_road:
            self.off_road_count += 1

    def get_summary(self):
        duration = time.time() - self.start_time
        return {
            "total_reward": self.total_reward,
            "steps": self.steps,
            "duration_seconds": duration,
            "avg_speed": np.mean(self.speeds) if self.speeds else 0,
            "max_speed": np.max(self.speeds) if self.speeds else 0,
            "total_distance": sum(self.distances),
            "avg_lane_deviation": np.mean(self.lane_deviations) if self.lane_deviations else 0,
            "waypoints_reached": self.waypoints_reached,
            "collision_count": self.collision_count,
            "off_road_count": self.off_road_count
        }

    def print_summary(self, episode_num, reason):
        summary = self.get_summary()
        termination_details = {
            "collision": f"Collisions: {summary['collision_count']}",
            "timeout": f"Steps: {summary['steps']}/{params.MAX_EPISODE_STEPS}",
            "off_road": f"Off-road events: {summary['off_road_count']}",
            "completed": f"Waypoints: {summary['waypoints_reached']}"
        }
        detail = termination_details.get(reason, "Unknown")

        logger.info("\n" + "="*60)
        logger.info(f"Episode {episode_num} Summary:")
        logger.info(f"  Termination Reason: {reason.upper()} ({detail})")
        logger.info(f"  Total Reward: {summary['total_reward']:.2f}")
        logger.info(f"  Steps: {summary['steps']}")
        logger.info(f"  Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info(f"  Average Speed: {summary['avg_speed']:.2f} km/h")
        logger.info(f"  Max Speed: {summary['max_speed']:.2f} km/h")
        logger.info(f"  Total Distance: {summary['total_distance']:.2f} meters")
        logger.info(f"  Average Lane Deviation: {summary['avg_lane_deviation']:.2f} meters")
        logger.info(f"  Waypoints Reached: {summary['waypoints_reached']}")
        logger.info("="*60 + "\n")

        print(f"\n🎯 Episode {episode_num} | Reason: {reason.upper()} | Reward: {summary['total_reward']:.1f} | Steps: {summary['steps']} | Speed: {summary['avg_speed']:.1f} km/h")

# Dummy implementation of CarlaViewer for headless operation
class CarlaViewer:
    def __init__(self, width=1280, height=720):
        pass
        
    def update(self, rgb_image, info=None):
        return True
        
    def close(self):
        pass

class CarlaRLEnv(gym.Env):
    """
    CARLA RL Environment that synchronizes RGB, Depth, and Segmentation sensors,
    and processes them using our Perception System. Compatible with Gymnasium.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, town=params.TOWN, render_mode=None, debug_route=False):
        super().__init__()
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),  # Throttle, steer, brake
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(params.OUTPUT_DIM,),
            dtype=np.float32
        )
        
        self.render_mode = None  # Always set to None for headless operation
        self.viewer = None
        self.debug_route = debug_route
        
        # CARLA-specific attributes
        self.client = None
        self.world = None
        self.sensor_queues = {'rgb': queue.Queue(), 'depth': queue.Queue(), 'segmentation': queue.Queue()}
        self.vehicle = None
        self.rgb_sensor = None
        self.depth_sensor = None
        self.segmentation_sensor = None
        self.collision_sensor = None
        self.town = town
        self.total_episodes = 0
        self.stopped_frames = 0
        self.off_road_steps = 0
        self.collision_frame_count = 0
        self.off_road_grace_steps_remaining = params.OFF_ROAD_GRACE_STEPS
        self.step_log_interval = 10  # Log every N steps
        
        # Waypoint tracking
        self.waypoints = []
        self.current_waypoint_index = 0
        self.target_waypoint = None
        self.route_length = params.WAYPOINT_ROUTE_LENGTH
        self.waypoint_distance = params.WAYPOINT_DISTANCE
        
        # Episode tracking
        self.collision_occurred = False
        self.steps_taken = 0
        self.max_steps = params.MAX_EPISODE_STEPS
        self.episode_rewards = []
        self.total_reward = 0
        self.last_location = None
        self.distance_traveled = 0
        
        # Initialize episode stats
        self.current_episode_stats = EpisodeStats()
        
        # Initialize perception system
        self.perception = PerceptionSystem(
            feature_dim=params.FEATURE_DIM,
            fusion_dim=params.FUSION_DIM,
            output_dim=params.OUTPUT_DIM,
            num_frames=params.NUM_FRAMES
        )

        # Load pre-trained SSL weights if available
        if os.path.exists(params.SSL_CHECKPOINT_PATH):
            logger.info(f"Loading pre-trained SSL weights from {params.SSL_CHECKPOINT_PATH}")
            try:
                ssl_state_dict = torch.load(params.SSL_CHECKPOINT_PATH, map_location=device)
                # Check if this is the perception system state dict or the full trainer state dict
                if 'memory' in ssl_state_dict:
                    # Direct perception system state dict
                    self.perception.load_state_dict(ssl_state_dict)
                else:
                    # Get just the perception part from the trainer state dict
                    # This handles the case where the SSL checkpoint is the full EnhancedSSLPerceptionTrainer
                    perception_state_dict = {}
                    for key, value in ssl_state_dict.items():
                        if key.startswith('perception.'):
                            perception_state_dict[key[11:]] = value  # Remove 'perception.' prefix
                    
                    if perception_state_dict:
                        self.perception.load_state_dict(perception_state_dict)
                    else:
                        logger.warning("Could not extract perception weights from SSL checkpoint")
                logger.info("✅ Successfully loaded pre-trained SSL weights")
            except Exception as e:
                logger.error(f"Error loading SSL weights: {e}")
                logger.info("⚠️ Using default initialization for perception system")
        else:
            logger.warning(f"SSL checkpoint not found at {params.SSL_CHECKPOINT_PATH}")
            logger.info("⚠️ Using default initialization for perception system")

        self.perception.to(device)
        
        # Prevent Perception System from training
        for param in self.perception.parameters():
            param.requires_grad = False
        self.perception.eval()
        
        # Connect to CARLA
        self._connect_to_carla()
    
    def _connect_to_carla(self):
        """Connect to CARLA server and set up the environment."""
        for attempt in range(3):  # Try 3 times
            try:
                self.client = carla.Client("localhost", 2000)
                self.client.set_timeout(20.0)
                self.world = self.client.load_world(self.town)
                self._setup_world()
                self._spawn_vehicle()
                self._setup_sensors()
                self._setup_collision_sensor()
                self._setup_waypoints()
                logger.info(f"✅ Connected to CARLA and loaded {self.town}")
                return  # Success!
            except Exception as e:
                logger.error(f"⚠️ CARLA connection attempt {attempt+1} failed: {e}")
                time.sleep(5)  # Wait before retrying

        raise RuntimeError("❌ Could not connect to CARLA after multiple attempts!")

    def _setup_world(self):
        """Set CARLA to synchronous mode for consistent sensor readings."""
        settings = self.world.get_settings()
        settings.synchronous_mode = params.SYNC_MODE
        settings.fixed_delta_seconds = params.FIXED_DELTA_SECONDS
        settings.no_rendering_mode = True  # Force no rendering for HPC
        
        # Adjust quality level for better visuals if rendering is prioritized
        if params.RENDER_PRIORITY:
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
            
        success = False
        for attempt in range(30):
            self.world.apply_settings(settings)
            new_settings = self.world.get_settings()
            if new_settings.synchronous_mode == params.SYNC_MODE:
                success = True
                break
            time.sleep(0.1)  # Small delay to allow settings to apply
        
        if not success:
            raise RuntimeError("Failed to apply synchronous mode after multiple attempts!")

    def _spawn_vehicle(self):
        """Improved vehicle spawning with road placement and physics stabilization."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]
        
        # Get map and spawn points
        map = self.world.get_map()
        spawn_points = map.get_spawn_points()
        
        if not spawn_points:
            raise RuntimeError("No spawn points found!")
        
        # Try to spawn at the first spawn point for consistency
        spawn_point = spawn_points[24]
        
        # Ensure spawn point is slightly above the road to prevent falling
        # First check if this is a valid driving waypoint
        waypoint = map.get_waypoint(spawn_point.location)
        if waypoint.lane_type != carla.LaneType.Driving:
            # If not on a driving lane, find a nearby driving waypoint
            waypoints = map.get_waypoints(spawn_point.location, 50.0)  # 50m radius
            driving_waypoints = [wp for wp in waypoints if wp.lane_type == carla.LaneType.Driving]
            
            if driving_waypoints:
                # Use the closest driving waypoint
                waypoint = min(driving_waypoints, key=lambda wp: wp.transform.location.distance(spawn_point.location))
                # Update spawn point to use this waypoint
                spawn_point = waypoint.transform
        
        # Ensure spawn height is correct (slightly above the waypoint to avoid collision)
        spawn_point.location.z += 0.2  # 20cm above the road surface
        
        # Try to spawn vehicle a few times
        for attempt in range(5):
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                
                # Immediately apply handbrake to prevent rolling
                control = carla.VehicleControl()
                control.hand_brake = True
                self.vehicle.apply_control(control)
                
                # Wait for physics to settle by advancing simulation
                for _ in range(10):  # Tick several times
                    self.world.tick()
                
                # Now release handbrake and ensure vehicle is in neutral position
                control.hand_brake = False
                control.throttle = 0.0
                control.brake = 0.0
                control.steer = 0.0
                self.vehicle.apply_control(control)
                
                # Store initial location
                self.last_location = self.vehicle.get_location()
                logger.info("🚗 Vehicle spawned and stabilized!")
                return
                
            except Exception as e:
                logger.error(f"Spawn attempt {attempt+1} failed: {e}")
                if hasattr(self, 'vehicle') and self.vehicle:
                    self.vehicle.destroy()
                time.sleep(0.1)
        
        raise RuntimeError("Failed to spawn vehicle after multiple attempts")

    def _setup_sensors(self):
        """Attach RGB, Depth, and Segmentation sensors to the vehicle with 3rd person view."""
        blueprint_library = self.world.get_blueprint_library()

        # 3rd person camera position (behind and above the vehicle)
        camera_pos = (-5.0, 0.0, 3.0)  # X=back/front, Y=left/right, Z=up/down
        camera_rot = (-15, 0, 0)  # Pitch, Yaw, Roll
        
        # RGB Camera - 3rd person
        self.rgb_sensor = self._create_sensor(
            blueprint_library, 
            "sensor.camera.rgb", 
            camera_pos, 
            "rgb",
            camera_rot
        )

        # Depth Camera - 3rd person
        self.depth_sensor = self._create_sensor(
            blueprint_library, 
            "sensor.camera.depth", 
            camera_pos, 
            "depth",
            camera_rot
        )

        # Segmentation Camera - 3rd person
        self.segmentation_sensor = self._create_sensor(
            blueprint_library, 
            "sensor.camera.semantic_segmentation", 
            camera_pos, 
            "segmentation",
            camera_rot
        )

    def _setup_collision_sensor(self):
        """Set up collision detection sensor."""
        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        """Callback for collision events."""
        self.collision_frame_count += 1
        self.collision_occurred = True
        if self.collision_frame_count <= params.COLLISION_WARNING_THRESHOLD:
            logger.info(f"⚠️ Collision #{self.collision_frame_count} detected with {event.other_actor.type_id}!")
        elif self.collision_frame_count % 20 == 0:
            logger.info(f"⚠️ Collision #{self.collision_frame_count} detected (continuing episode)")

    def _create_sensor(self, blueprint_library, sensor_type, position, queue_name, rotation=(0, 0, 0)):
        """Helper function to spawn sensors with specific attributes."""
        bp = blueprint_library.find(sensor_type)
        bp.set_attribute("image_size_x", str(params.SENSOR_WIDTH))
        bp.set_attribute("image_size_y", str(params.SENSOR_HEIGHT))
        
        transform = carla.Transform(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2])
        )
        
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        
        # Attach callback function to sensor
        sensor.listen(lambda data: self.sensor_queues[queue_name].put(data))
        return sensor
    
    def _calculate_reward(self):
        """
        Simplified reward function using speed, centering, and angle alignment.
        Replaces the complex handcrafted reward with a clean multiplicative version.
        Implements grace periods for collisions and off-road to reduce early termination.
        """

        def angle_diff(v0, v1):
            angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
            if angle > np.pi: angle -= 2 * np.pi
            elif angle <= -np.pi: angle += 2 * np.pi
            return angle

        def vector(v):
            return np.array([v.x, v.y, v.z])

        done = False
        reward = 0.0

        vehicle = self.vehicle
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)

        # Grace period: collision alone doesn't immediately end episode
        # Only penalize if collision persists beyond grace frames
        if self.collision_occurred:
            if self.collision_frame_count > params.COLLISION_GRACE_FRAMES:
                # Gradual penalty instead of immediate termination
                collision_penalty = params.COLLISION_PENALTY * (self.collision_frame_count / params.COLLISION_GRACE_FRAMES)
                return collision_penalty, True
            else:
                # During grace period, minor penalty but continue
                reward += params.COLLISION_PENALTY * 0.1 * (self.collision_frame_count / params.COLLISION_GRACE_FRAMES)

        # Off-road check with grace period
        is_off_road = self._is_off_road()
        if is_off_road:
            if self.off_road_grace_steps_remaining > 0:
                self.off_road_grace_steps_remaining -= 1
                reward += params.OFF_ROAD_PENALTY * 0.1  # Minor penalty during grace
            else:
                return params.OFF_ROAD_PENALTY, True
        else:
            # Reset grace when back on road
            self.off_road_grace_steps_remaining = params.OFF_ROAD_GRACE_STEPS

        # Fail: Stopped too long - more forgiving with increased grace frames
        if speed_kmh < 1.0:
            self.stopped_frames += 1

            if self.total_episodes < 5:
                max_stopped_frames = params.STOPPED_GRACE_FRAMES_EARLY
            elif self.steps_taken < 50:
                max_stopped_frames = params.STOPPED_GRACE_FRAMES_MID
            else:
                max_stopped_frames = params.STOPPED_GRACE_FRAMES_NORMAL

            if self.stopped_frames > max_stopped_frames:
                return -10.0, True
        else:
            self.stopped_frames = 0

        # Get current waypoint and direction
        waypoint = self.world.get_map().get_waypoint(vehicle_location)
        distance_from_center = vehicle_location.distance(waypoint.transform.location) if waypoint else 0.0

        vehicle_forward = vector(vehicle.get_velocity())
        wp_forward = vector(waypoint.transform.rotation.get_forward_vector()) if waypoint else np.array([1.0, 0.0, 0.0])
        angle = angle_diff(vehicle_forward, wp_forward)

        # Speed reward
        min_speed = 15.0
        target_speed = 20.0
        max_speed = 25.0

        if speed_kmh < min_speed:
            speed_reward = speed_kmh / min_speed
        elif speed_kmh > target_speed:
            speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
        else:
            speed_reward = 1.0

        # Centering and angle
        centering_factor = max(1.0 - distance_from_center / 3.0, 0.0)
        angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

        # Multiply
        reward += speed_reward * centering_factor * angle_factor

        # Step logging every N steps for monitoring
        if self.steps_taken % self.step_log_interval == 0:
            logger.info(
                f"Step {self.steps_taken} | "
                f"Speed: {speed_kmh:.1f} km/h | "
                f"Pos: ({vehicle_location.x:.1f}, {vehicle_location.y:.1f}, {vehicle_location.z:.1f}) | "
                f"Dist: {distance_from_center:.2f}m | "
                f"Reward: {reward:.3f} | "
                f"Collisions: {self.collision_frame_count} | "
                f"Waypoints: {self.current_waypoint_index}/{len(self.waypoints)}"
            )

        return reward, done

    def _is_off_road(self):
        """Off-road detection with tolerance for initial vehicle placement"""
        try:
            current_location = self.vehicle.get_location()
            
            # Use project_to_road=True to allow small deviations from road center
            # This is more forgiving for vehicle placement
            car_waypoint = self.world.get_map().get_waypoint(current_location, project_to_road=True)
            
            # If waypoint is None, vehicle is truly off-road
            if car_waypoint is None:
                return True
            
            # Check if the waypoint is on a valid driving surface
            valid_lane_types = [
                carla.LaneType.Driving,
                carla.LaneType.Shoulder,
                carla.LaneType.Sidewalk,
                carla.LaneType.Bidirectional
            ]
            
            if car_waypoint.lane_type not in valid_lane_types:
                return True
                
            # Calculate distance to the waypoint (projected position)
            # Allow up to 5 meters deviation before considering off-road
            waypoint_location = car_waypoint.transform.location
            distance_to_road = current_location.distance(waypoint_location)
            
            # Be more lenient in early steps to allow vehicle to stabilize
            if self.steps_taken < 50:
                max_distance = 8.0  # More lenient for first 50 steps
            else:
                max_distance = 5.0  # Normal tolerance
            
            if distance_to_road > max_distance:
                #logger.info(f"Off-road: distance={distance_to_road:.2f}m")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Exception in off-road check: {e}")
            # If there's an exception, be forgiving during early steps
            if self.steps_taken < 100:
                return False
            return True

    def _setup_waypoints(self):
        """Generate a dynamic route with varied paths and junction handling."""
        self.waypoints = []
        self.current_waypoint_index = 0
        
        # Get the current vehicle waypoint
        vehicle_location = self.vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(vehicle_location)
        
        # Route complexity parameters that scale with training progress
        # More complex routes as training progresses
        base_route_length = 30
        max_bonus_length = 70
        route_length_scale = min(1.0, self.total_episodes / 500)  # Scales from 0 to 1 over 500 episodes
        target_route_length = int(base_route_length + (max_bonus_length * route_length_scale))
        
        # Junction handling parameters
        # Higher probability of taking turns as training progresses
        base_turn_probability = 0.3
        max_turn_probability = 0.7
        turn_probability_scale = min(1.0, self.total_episodes / 300)  # Scales from 0 to 1 over 300 episodes
        junction_turn_probability = base_turn_probability + ((max_turn_probability - base_turn_probability) * turn_probability_scale)
        
        # Keep track of turns and junctions for logging
        turn_count = 0
        junction_count = 0
        
        # Cache previous direction to avoid U-turns
        prev_direction = None
        
        # Generate a route
        for i in range(target_route_length):
            # Get next waypoints (could be multiple at junctions)
            next_wps = current_waypoint.next(self.waypoint_distance)
            
            if not next_wps:
                logger.info(f"⚠️ Route generation ended early: No more waypoints available")
                break
            
            # Check if we're at a junction
            is_junction = current_waypoint.is_junction
            if is_junction:
                junction_count += 1
            
            # Handle path selection based on junction status
            if is_junction and len(next_wps) > 1 and random.random() < junction_turn_probability:
                # We're at a junction with multiple options and decide to turn
                
                # Get direction vectors for each option
                directions = []
                for wp in next_wps:
                    wp_forward = wp.transform.get_forward_vector()
                    # Convert to 2D unit vector
                    direction = np.array([wp_forward.x, wp_forward.y])
                    direction_length = np.linalg.norm(direction)
                    if direction_length > 0:
                        direction = direction / direction_length
                    directions.append(direction)
                
                # Calculate angles between current direction and each option
                current_forward = current_waypoint.transform.get_forward_vector()
                current_direction = np.array([current_forward.x, current_forward.y])
                current_direction_length = np.linalg.norm(current_direction)
                if current_direction_length > 0:
                    current_direction = current_direction / current_direction_length
                
                # Avoid the path that leads to a U-turn
                valid_indices = []
                for idx, direction in enumerate(directions):
                    # Calculate dot product (negative means sharp turn)
                    dot_product = np.dot(current_direction, direction)
                    # Avoid U-turns (angle > 120 degrees)
                    if dot_product > -0.5:  # cos(120°) ≈ -0.5
                        valid_indices.append(idx)
                
                if valid_indices:
                    # Randomly choose among valid options
                    chosen_idx = random.choice(valid_indices)
                    next_wp = next_wps[chosen_idx]
                    if chosen_idx != 0:  # If not taking the "straight" option
                        turn_count += 1
                else:
                    # If all options would cause a U-turn, take the straightest path
                    dot_products = [np.dot(current_direction, d) for d in directions]
                    chosen_idx = np.argmax(dot_products)
                    next_wp = next_wps[chosen_idx]
            else:
                # Default to first path (usually straight ahead)
                next_wp = next_wps[0]
            
            # Store the waypoint
            self.waypoints.append(next_wp)
            
            # Update current waypoint for next iteration
            current_waypoint = next_wp
            
            # Update previous direction
            wp_forward = next_wp.transform.get_forward_vector()
            prev_direction = np.array([wp_forward.x, wp_forward.y])
            direction_length = np.linalg.norm(prev_direction)
            if direction_length > 0:
                prev_direction = prev_direction / direction_length
        
        # Set initial target waypoint
        if self.waypoints:
            self.target_waypoint = self.waypoints[0]
        
        # Calculate route complexity metrics
        route_length = len(self.waypoints)
        turn_percentage = (turn_count / max(1, junction_count)) * 100 if junction_count > 0 else 0
        
        logger.info(f"✅ Generated {route_length} waypoints for navigation")
        logger.info(f"   Complexity: {junction_count} junctions, {turn_count} turns ({turn_percentage:.1f}% turn rate)")
        
        return route_length > 0  # Return success status

    def _update_waypoints(self):
        """Update target waypoint based on vehicle progress."""
        if not self.waypoints:
            return False  # No waypoints to update
        
        # Get current vehicle location
        vehicle_location = self.vehicle.get_location()
        
        # Distance to current target waypoint
        target_location = self.target_waypoint.transform.location
        distance = vehicle_location.distance(target_location)
        
        # Check if we're close enough to the target waypoint
        if distance < 3.0:  # Within 3 meters
            # Mark current waypoint as reached
            waypoint_reached = True
            
            # Advance to next waypoint
            self.current_waypoint_index = min(self.current_waypoint_index + 1, len(self.waypoints) - 1)
            self.target_waypoint = self.waypoints[self.current_waypoint_index]
            
            return True  # Waypoint reached
        
        return False  # Waypoint not reached yet

    def _get_sensor_data(self, timeout=2.0):
        """Retrieve synchronized sensor data and convert them into tensors."""
        try:
            rgb_data = self.sensor_queues["rgb"].get(timeout=timeout)
            depth_data = self.sensor_queues["depth"].get(timeout=timeout)
            seg_data = self.sensor_queues["segmentation"].get(timeout=timeout)
        except queue.Empty:
            logger.info("⚠ Warning: Sensor data missing! Restarting CARLA...")
            self._connect_to_carla()  # Restart CARLA connection
            return None

        return {
            "rgb": self._convert_to_tensor(rgb_data, sensor_type="rgb"),
            "depth": self._convert_to_tensor(depth_data, sensor_type="depth"),
            "segmentation": self._convert_to_tensor(seg_data, sensor_type="segmentation"),
        }

    def step(self, action):
        """
        Apply the chosen action (Throttle, Steer, Brake) to the vehicle.
        Fetch sensor data and return the processed state, reward, done flag, and info.
        """
        # Increment step counter
        self.steps_taken += 1
        
        # Force movement in early episodes to help bootstrap learning
        if self.total_episodes < 10:  # For the first 5 episodes
            action = action.copy()
            action[0] = max(action[0], 0.5)  # Force at least 0.5 throttle
            action[2] = 0.0  # No braking allowed

        action = action.copy()

        # Process action: values come in as [-1, 1] for throttle and steer, [0, 1] for brake
        throttle = float(max(0.0, action[0]))  # Convert to [0, 1]
        steer = float(action[1])  # Keep as [-1, 1]
        brake = float(max(0.0, action[2]))  # Keep as [0, 1]
        
        # Apply control
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)

        # Advance simulation
        try:
            self.world.tick()
        except RuntimeError as e:
            logger.error(f"Runtime error during world tick: {e}")
            return np.zeros(params.OUTPUT_DIM), 0, True, True, {"error": "Simulation error"}

        # Fetch new state
        sensor_data = self._get_sensor_data()
        if sensor_data is None:
            logger.warning("⚠️ Sensor data missing, attempting recovery...")
            # Try to recover by resetting sensors
            self._cleanup()
            self._spawn_vehicle()
            self._setup_sensors()
            self._setup_collision_sensor()
            self.world.tick()
            sensor_data = self._get_sensor_data()
            if sensor_data is None:
                # If still no data, return with proper info structure
                return np.zeros(params.OUTPUT_DIM), 0, True, False, {
                    "error": "Sensor data missing",
                    "termination_reason": "sensor_failure",
                    "speed": 0,
                    "distance_traveled": self.distance_traveled,
                    "step": self.steps_taken
                }
        
        # Update waypoints and check if we reached one
        waypoint_reached = self._update_waypoints()

        # Calculate lane deviation for stats
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        waypoint = self.world.get_map().get_waypoint(vehicle_location)
        lane_deviation = vehicle_location.distance(waypoint.transform.location) if waypoint else 0.0

        # Get state through perception system
        with torch.no_grad():
            state = self.perception(sensor_data).squeeze(0).cpu().numpy()

        # Calculate reward
        reward_value, reward_done = self._calculate_reward()
        reward = reward_value

        # Check off-road status for episode stats
        is_off_road = self._is_off_road()

        self.total_reward += reward
        
        # Update distance traveled
        current_location = self.vehicle.get_location()

        distance = 0
        if self.last_location:
            distance = current_location.distance(self.last_location)
            self.distance_traveled += distance
        
        # Get vehicle info for monitoring
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

        # Update episode stats
        self.current_episode_stats.add_step_info(
            reward, speed, distance, lane_deviation, waypoint_reached,
            collision=self.collision_frame_count > 0,
            off_road=is_off_road
        )

        self.last_location = current_location
        
        # Determine if episode is done
        timeout = self.steps_taken >= self.max_steps
        done = timeout or reward_done
        
        # Prepare info dict with comprehensive information
        termination_reason = "in_progress"
        if timeout:
            termination_reason = "timeout"
        elif reward_done:
            termination_reason = "reward_based"
        elif self.collision_occurred:
            termination_reason = "collision_warning"
        else:
            termination_reason = "completed"
        
        # Get stats from episode tracker
        episode_stats = self.current_episode_stats.get_summary()

        # Comprehensive info dictionary with all relevant metrics for callbacks
        info = {
            # Key performance metrics
            "speed": speed,
            "avg_speed": episode_stats["avg_speed"],
            "total_reward": self.total_reward,
            "distance_traveled": self.distance_traveled,
            "lane_deviation": lane_deviation,
            "avg_lane_deviation": episode_stats["avg_lane_deviation"],
            "waypoints_reached": episode_stats["waypoints_reached"],
            
            # Episode state
            "step": self.steps_taken,
            "collision": self.collision_occurred,
            "termination_reason": termination_reason,
            "waypoint_index": self.current_waypoint_index,
            
            # These keys match Monitor's expected format
            "r": reward,  # Current step reward (for Monitor)
            "l": self.steps_taken,  # Episode length (for Monitor)
            "t": time.time() - self.episode_start_time  # Episode duration (for Monitor)
        }
        
        # If episode is done, print stats
        if done:
            self.current_episode_stats.print_summary(self.total_episodes, termination_reason)
        
        return state, reward, done, False, info

    def _convert_to_tensor(self, raw_data, sensor_type):
        """Convert raw CARLA sensor data into PyTorch tensor."""
        array = np.frombuffer(raw_data.raw_data, dtype=np.uint8)
        if sensor_type == "rgb":
            array = array.reshape((params.SENSOR_HEIGHT, params.SENSOR_WIDTH, 4))[:, :, :3]  # Extract RGB channels
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0  # Normalize
            return tensor.unsqueeze(0).to(device)  # Add batch dim

        elif sensor_type == "depth":
            array = array.reshape((params.SENSOR_HEIGHT, params.SENSOR_WIDTH, 4))
            # Extract depth value (single channel)
            depth = (array[..., 2] + array[..., 1] * 256 + array[..., 0] * 256 * 256) / (256**3 - 1)
            tensor = torch.from_numpy(depth.copy()).float().unsqueeze(0).unsqueeze(0)  # Make it (1, 1, H, W)
            return tensor.to(device)  # Return as single channel tensor

        elif sensor_type == "segmentation":
            array = array.reshape((params.SENSOR_HEIGHT, params.SENSOR_WIDTH, 4))[:, :, 2]  # Extract segmentation labels
            tensor = torch.from_numpy(array.copy()).float().unsqueeze(0)  # Convert to float and add batch dim
            
            # Convert to 3-channel format for segmentation model
            tensor = tensor.unsqueeze(0)  # Now (1, 1, H, W)
            tensor = tensor.repeat(1, 3, 1, 1)  # Expand to (1, 3, H, W)
            return tensor.to(device)
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        # Create a new episode stats object
        self.current_episode_stats = EpisodeStats()
        
        # Reset episode tracking
        self.collision_occurred = False
        self.collision_frame_count = 0
        self.steps_taken = 0
        self.total_reward = 0
        self.stopped_frames = 0
        self.off_road_frames = 0
        self.off_road_grace_steps_remaining = params.OFF_ROAD_GRACE_STEPS
        self.distance_traveled = 0
        self.total_episodes += 1
        self.last_location = None
        self.episode_start_time = time.time()  # Track when this episode started
        
        # Clean up existing objects
        self._cleanup()
        
        # Spawn new vehicle and sensors
        try:
            self._spawn_vehicle()
            self._setup_sensors()
            self._setup_collision_sensor()
            self._setup_waypoints()  # Generate new waypoints for this episode
            
            # Clear perception system memory
            self.perception.memory = []
            
            # Tick simulation to get initial state
            for _ in range(3):
                self.world.tick()
            time.sleep(0.5)  # Wait for sensors to initialize
            
            # Get initial state
            sensor_data = self._get_sensor_data()
            if sensor_data is None:
                logger.info("⚠️ No sensor data available after reset. Using zero state.")
                return np.zeros(params.OUTPUT_DIM), {}
            
            # Initialize perception system memory if needed
            if len(self.perception.memory) == 0:
                with torch.no_grad():
                    rgb_feat = self.perception.rgb_encoder(sensor_data['rgb'])
                    depth_feat = self.perception.depth_encoder(sensor_data['depth'])
                    seg_feat = self.perception.segmentation_parser(sensor_data['segmentation'])
                    fused_feat = self.perception.fusion_transformer(rgb_feat, depth_feat, seg_feat)
                    
                    # Initialize memory with different versions of the first frame
                    for i in range(self.perception.num_frames - 1):
                        # Add small noise to create variation
                        noise = torch.randn_like(fused_feat).to(device) * 0.01 * (i + 1)
                        self.perception.memory.append((fused_feat + noise).detach())
            
            # Get state through perception system
            with torch.no_grad():
                state = self.perception(sensor_data).squeeze(0).cpu().numpy()
            
            return state, {}
            
        except Exception as e:
            logger.error(f"⚠️ Error during reset: {e}")
            return np.zeros(params.OUTPUT_DIM), {"error": str(e)}
        
    def _cleanup(self):
        """Destroy actors to clean up resources."""
        if hasattr(self, 'collision_sensor') and self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        
        if hasattr(self, 'rgb_sensor') and self.rgb_sensor:
            self.rgb_sensor.destroy()
            self.rgb_sensor = None
            
        if hasattr(self, 'depth_sensor') and self.depth_sensor:
            self.depth_sensor.destroy()
            self.depth_sensor = None
            
        if hasattr(self, 'segmentation_sensor') and self.segmentation_sensor:
            self.segmentation_sensor.destroy()
            self.segmentation_sensor = None
            
        if hasattr(self, 'vehicle') and self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        
        # Clear the queues
        for queue_name in self.sensor_queues:
            while not self.sensor_queues[queue_name].empty():
                try:
                    self.sensor_queues[queue_name].get(block=False)
                except queue.Empty:
                    pass

    def render(self):
        """This method is called when render_mode is set to 'human'."""
        # No-op for headless operation
        pass

    def close(self):
        """Clean up all resources."""
        self._cleanup()
        
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        
        if self.world:
            # Restore original settings
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            logger.info("🔄 Restored asynchronous mode")



class TrainingMonitor:
    """Simplified monitor class for tracking training progress"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize simple metrics tracking as lists
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.episode_distances = []
        self.termination_reasons = []
        self.waypoints_reached = []
        self.collision_rates = []
        self.off_road_rates = []
        
        # Initialize episode counter
        self.episode_count = 0
        
        # Set up TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1)
            self.has_tensorboard = True
            logger.info("✅ TensorBoard writer initialized successfully")
        except ImportError:
            logger.warning("⚠️ TensorBoard not available")
            self.has_tensorboard = False
    
    def update(self, info):
        """Update metrics with information from completed episode."""
        # Increment episode counter
        self.episode_count += 1
        episode = self.episode_count  # For clarity
        
        # Extract info (with defaults if missing)
        reward = info.get('total_reward', info.get('r', 0.0))
        length = info.get('length', info.get('l', 0))
        distance = info.get('distance_traveled', 0.0)
        speed = info.get('avg_speed', 0.0)
        waypoints = info.get('waypoints_reached', 0)
        reason = info.get('termination_reason', 'unknown')
        
        # Calculate derived metrics
        is_collision = 1.0 if reason == 'collision' else 0.0
        is_off_road = 1.0 if reason == 'off_road' else 0.0
        
        # Store metrics
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_distances.append(distance)
        self.episode_speeds.append(speed)
        self.termination_reasons.append(reason)
        self.waypoints_reached.append(waypoints)
        self.collision_rates.append(is_collision)
        self.off_road_rates.append(is_off_road)
        
        # Log to TensorBoard
        if self.has_tensorboard:
            try:
                self.writer.add_scalar('training/reward', reward, episode)
                self.writer.add_scalar('training/length', length, episode)
                self.writer.add_scalar('vehicle/distance', distance, episode) 
                self.writer.add_scalar('vehicle/avg_speed', speed, episode)
                self.writer.add_scalar('vehicle/waypoints_reached', waypoints, episode)
                self.writer.add_scalar('safety/collision', is_collision, episode)
                self.writer.add_scalar('safety/off_road', is_off_road, episode)
                
                # Add moving averages if we have enough data
                window = min(10, len(self.episode_rewards))
                if window > 1:
                    self.writer.add_scalar('training/reward_avg', 
                                        np.mean(self.episode_rewards[-window:]), 
                                        episode)
                    self.writer.add_scalar('vehicle/distance_avg', 
                                        np.mean(self.episode_distances[-window:]), 
                                        episode)
                
                # Force flush to disk
                self.writer.flush()
            except Exception as e:
                logger.error(f"⚠️ Error logging to TensorBoard: {e}")
        
        # Save backup data every 10 episodes and at key milestones
        if episode % 10 == 0:
            self._save_backup(episode)
        
        # Return the episode number for verification
        return episode

    def _save_backup(self, episode):
        """Save backup at specific episode"""
        import pickle
        
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(self.log_dir, f'metrics_backup_ep{episode}_{timestamp}.pkl')
        
        # Create backup data dictionary
        backup_data = {
            'episode': episode,
            'timestamp': timestamp,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_speeds': self.episode_speeds,
            'episode_distances': self.episode_distances,
            'termination_reasons': self.termination_reasons,
            'waypoints_reached': self.waypoints_reached,
            'collision_rates': self.collision_rates,
            'off_road_rates': self.off_road_rates
        }
        
        # Save the backup
        try:
            with open(backup_path, 'wb') as f:
                pickle.dump(backup_data, f)
            logger.info(f"✅ Metrics backup saved: Episode {episode}")
            
            # Also save backup as CSV for easy viewing
            try:
                import pandas as pd
                csv_path = os.path.join(self.log_dir, f'metrics_ep{episode}.csv')
                
                # Create DataFrame with episode metrics
                metrics_df = pd.DataFrame({
                    'episode': list(range(1, episode + 1)),
                    'reward': self.episode_rewards,
                    'length': self.episode_lengths,
                    'distance': self.episode_distances,
                    'avg_speed': self.episode_speeds,
                    'waypoints': self.waypoints_reached,
                    'termination': self.termination_reasons,
                    'collision': self.collision_rates,
                    'off_road': self.off_road_rates
                })
                
                metrics_df.to_csv(csv_path, index=False)
            except Exception as e:
                logger.error(f"⚠️ Error saving CSV: {e}")
                
        except Exception as e:
            logger.error(f"⚠️ Failed to save metrics backup: {e}")

    def close(self):
        """Close TensorBoard writer"""
        if self.has_tensorboard:
            try:
                self.writer.close()
            except Exception as e:
                logger.error(f"⚠️ Error closing TensorBoard writer: {e}")

class SimpleEpisodeCallback(CheckpointCallback):
    """Simplified callback for tracking episodes and saving checkpoints."""
    
    def __init__(self, monitor, total_episodes=10000, check_freq=10, save_path="./models", verbose=1):
        super().__init__(save_freq=check_freq, save_path=save_path, name_prefix="ppo_carla", verbose=verbose)
        self.monitor = monitor
        self.check_freq = check_freq
        self.last_time = time.time()
        self.last_time_steps = 0
        self.episode_count = 0
        self.total_episodes = total_episodes
        self.last_mean_reward = -float('inf')
        
    def _on_step(self):
        # Log training speed every 1000 steps
        if self.num_timesteps % 1000 == 0:
            fps = int((self.num_timesteps - self.last_time_steps) / max(time.time() - self.last_time, 1e-8))
            self.last_time = time.time()
            self.last_time_steps = self.num_timesteps
            if self.verbose > 0:
                logger.info(f"\nTraining speed: {fps} fps | Total steps: {self.num_timesteps}")

        # Check if a new episode ended using the 'dones' flag
        if self.locals.get('dones') is not None:
            for done, info in zip(self.locals['dones'], self.locals['infos']):
                if done and info:
                    # Get termination_reason with fallback
                    term_reason = info.get("termination_reason", "unknown")
                    current_episode = self.monitor.update(info)

                    if self.verbose > 0:
                        print(f"✅ Episode {current_episode} completed (reason: {term_reason})")

                    if current_episode % self.check_freq == 0:
                        logger.info(f"💾 Saving checkpoint at episode {current_episode}...")
                        try:
                            checkpoint_path = os.path.join(self.save_path, f"model_ep{current_episode}")
                            self.model.save(checkpoint_path)
                            logger.info(f"✅ Model saved: {checkpoint_path}")

                            # Save environment stats
                            if hasattr(self.model, 'get_env') and hasattr(self.model.get_env(), 'save'):
                                env_path = os.path.join(self.save_path, f"env_ep{current_episode}.pkl")
                                self.model.get_env().save(env_path)
                                logger.info(f"✅ Env stats saved: {env_path}")
                        except Exception as e:
                            logger.error(f"❌ Error saving checkpoint: {e}")

                    # Final save if total episodes reached
                    if current_episode >= self.total_episodes:
                        logger.info(f"\n🏁 Reached target of {self.total_episodes} episodes! Saving final model...")

                        try:
                            run_name = self.model.tensorboard_log.split('/')[-1]
                            final_model_path = os.path.join(self.save_path, f"{run_name}_final")
                            self.model.save(final_model_path)

                            env_stats_path = os.path.join(self.save_path, f"{run_name}_env_final.pkl")
                            self.model.get_env().save(env_stats_path)

                            logger.info(f"✅ Final model saved to {final_model_path}")
                            logger.info(f"✅ Env stats saved to {env_stats_path}")
                        except Exception as e:
                            logger.error(f"❌ Final save error: {e}")

                        return False  # Stop training

        return True




def train(total_episodes=10000, debug_route=False, resume_from=None):
    """Main training function with simplified monitoring approach and resume capability for HPC."""
    # Create output directories
    os.makedirs(params.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(params.TENSORBOARD_LOG, exist_ok=True)
    
    # Set random seed
    set_random_seed(params.SEED)
    
    # Create timestamp for this training run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # For adjusting episode count in callback if we're resuming
    starting_episode = 0
    original_tb_log = None
    
    # If resuming, use existing run name for continuity and determine starting episode
    if resume_from:
        resume_path = resume_from
        # Remove .zip extension if present
        if resume_path.endswith('.zip'):
            resume_path = resume_path[:-4]
            
        # Extract run name from the checkpoint path
        if "_final" in resume_path:
            # This is a final model
            run_name = os.path.basename(resume_path).replace("_final", "")
            
            # Try to find the original tensorboard log directory
            potential_tb_dirs = glob.glob(os.path.join(params.TENSORBOARD_LOG, run_name + "*"))
            if potential_tb_dirs:
                original_tb_log = potential_tb_dirs[0]
                logger.info(f"Found original tensorboard log: {original_tb_log}")
        elif "model_ep" in resume_path:
            # This is an episode checkpoint - find the parent run name
            checkpoint_dir = os.path.dirname(resume_path)
            # Try to extract episode number
            match = re.search(r'model_ep(\d+)', os.path.basename(resume_path))
            if match:
                starting_episode = int(match.group(1))
                logger.info(f"Starting from episode {starting_episode}")
                
            # Look for tensorboard logs with matching patterns
            tb_pattern = "_".join(os.path.basename(resume_path).split("_")[:3])
            potential_tb_dirs = glob.glob(os.path.join(params.TENSORBOARD_LOG, tb_pattern + "*"))
            if potential_tb_dirs:
                original_tb_log = potential_tb_dirs[0]
                run_name = os.path.basename(original_tb_log)
                logger.info(f"Found original tensorboard log: {original_tb_log}")
            else:
                # Just use a generic name with timestamp
                run_name = f"carla_ppo_resumed_{timestamp}"
        else:
            # Generic backup case
            run_name = f"carla_ppo_resumed_{timestamp}"
    else:
        run_name = f"carla_ppo_{timestamp}"
    
    # If continuing an existing run, use the same log directory
    if original_tb_log and os.path.exists(original_tb_log):
        log_dir = original_tb_log
        logger.info(f"Continuing with existing log directory: {log_dir}")
    else:
        log_dir = os.path.join(params.TENSORBOARD_LOG, run_name)
        logger.info(f"Creating new log directory: {log_dir}")
    
    # Create simplified monitor
    monitor = TrainingMonitor(log_dir=log_dir)
    
    # If resuming, try to load previous metrics
    if resume_from and starting_episode > 0:
        # Look for metrics backup files
        metrics_files = glob.glob(os.path.join(log_dir, f"metrics_ep*.csv"))
        if metrics_files:
            try:
                # Find the highest episode number
                metrics_files.sort(key=lambda x: int(re.search(r'metrics_ep(\d+)', x).group(1)))
                latest_metrics = metrics_files[-1]
                logger.info(f"Loading previous metrics from {latest_metrics}")
                
                # Load metrics using pandas
                import pandas as pd
                metrics_df = pd.read_csv(latest_metrics)
                
                # Preload monitor with metrics data
                monitor.episode_rewards = metrics_df['reward'].tolist()
                monitor.episode_lengths = metrics_df['length'].tolist()
                monitor.episode_distances = metrics_df['distance'].tolist()
                monitor.episode_speeds = metrics_df['avg_speed'].tolist() if 'avg_speed' in metrics_df else []
                monitor.termination_reasons = metrics_df['termination'].tolist() if 'termination' in metrics_df else []
                monitor.waypoints_reached = metrics_df['waypoints'].tolist() if 'waypoints' in metrics_df else []
                monitor.collision_rates = metrics_df['collision'].tolist() if 'collision' in metrics_df else []
                monitor.off_road_rates = metrics_df['off_road'].tolist() if 'off_road' in metrics_df else []
                
                # Set episode counter
                monitor.episode_count = len(monitor.episode_rewards)
                logger.info(f"Loaded metrics for {monitor.episode_count} episodes")
                
                # Verify consistency with starting_episode
                if monitor.episode_count != starting_episode:
                    logger.warning(f"Metrics count ({monitor.episode_count}) doesn't match expected episode count ({starting_episode})")
                    # Trust the metrics file as source of truth
                    starting_episode = monitor.episode_count
            except Exception as e:
                logger.error(f"Error loading previous metrics: {e}")
                logger.info("Will start with fresh metrics tracking")
    
    # Print training configuration
    logger.info("\n" + "="*50)
    logger.info("CARLA RL Training Configuration:")
    logger.info(f"- Total Episodes Target: {starting_episode + total_episodes}")
    logger.info(f"- Current Episode: {starting_episode}")
    logger.info(f"- Remaining Episodes: {total_episodes}")
    logger.info(f"- Town: {params.TOWN}")
    logger.info(f"- Device: {device}")
    logger.info(f"- Learning Rate: {params.LEARNING_RATE}")
    logger.info(f"- Batch Size: {params.BATCH_SIZE}")
    logger.info(f"- Debug Route: {debug_route}")
    logger.info(f"- Log Directory: {log_dir}")
    
    # Add resume information if applicable
    if resume_from:
        logger.info(f"- Resuming from: {resume_from}")
    
    logger.info("="*50 + "\n")
    
    # Create environment
    logger.info("Creating environment...")
    env = CarlaRLEnv(town=params.TOWN, render_mode=None, debug_route=debug_route)
    
    # Wrap environment with Monitor to collect episode statistics
    env = Monitor(
        env, 
        filename=os.path.join(log_dir, "monitor.csv"),
        info_keywords=(
            'termination_reason', 
            'distance_traveled', 
            'avg_speed',
            'avg_lane_deviation',
            'waypoints_reached',
            'collision',
            'speed'
        )
    )
    
    # Create dummy vector environment
    env = DummyVecEnv([lambda: env])
    
    # Create normalized environment with appropriate normalization settings
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=params.GAMMA
    )
    
    # Load environment normalization stats if resuming
    if resume_from:
        # Find corresponding environment stats file
        env_stats_path = None
        
        # Check for environment stats with matching name pattern
        if "_final" in resume_from:
            env_stats_path = resume_from.replace("_final", "_env_final.pkl")
        else:
            # Try to find environment stats that correspond to the model checkpoint
            checkpoint_dir = os.path.dirname(resume_from)
            model_basename = os.path.basename(resume_from)
            
            # Extract episode number if it exists in the filename
            ep_num = None
            if "ep" in model_basename:
                try:
                    ep_num = int(re.search(r'ep(\d+)', model_basename).group(1))
                    env_stats_path = os.path.join(checkpoint_dir, f"env_ep{ep_num}.pkl")
                except:
                    logger.warning("Could not extract episode number from checkpoint filename")
        
        # Load environment stats if found
        if env_stats_path and os.path.exists(env_stats_path):
            logger.info(f"Loading environment stats from {env_stats_path}")
            env = VecNormalize.load(env_stats_path, env)
            # Keep collecting running statistics
            env.training = True
        else:
            logger.warning(f"No environment stats file found for {resume_from}. Using fresh normalization.")
    
    # Create or load PPO model
    if resume_from and (os.path.exists(resume_from + ".zip") or os.path.exists(resume_from)):
        model_path = resume_from + ".zip" if os.path.exists(resume_from + ".zip") else resume_from
        logger.info(f"Loading PPO model from {model_path}")
        model = PPO.load(
            model_path,
            env=env,
            device=device,
            tensorboard_log=log_dir
        )
        
        # Update model hyperparameters if needed
        model.learning_rate = lambda _:params.LEARNING_RATE
        model.n_steps = lambda _:params.N_STEPS
        model.batch_size = lambda _:params.BATCH_SIZE
        model.n_epochs = lambda _:params.N_EPOCHS
        model.gamma = lambda _:params.GAMMA
        model.gae_lambda = lambda _: params.GAE_LAMBDA
        model.clip_range = lambda _: params.CLIP_RANGE
        
        # Initialize the episode info buffer with proper size if it doesn't exist
        if not hasattr(model, 'ep_info_buffer') or model.ep_info_buffer is None:
            model.ep_info_buffer = deque(maxlen=10000)
            logger.info("Initialized new episode info buffer")
        elif len(model.ep_info_buffer) < starting_episode:
            # If buffer is smaller than our expected episode count, add padding
            padding_needed = starting_episode - len(model.ep_info_buffer)
            logger.info(f"Padding episode buffer with {padding_needed} empty entries")
            # Add padding with empty info entries
            for _ in range(padding_needed):
                model.ep_info_buffer.append({})
    else:
        logger.info("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            learning_rate=params.LEARNING_RATE,
            n_steps=params.N_STEPS,
            batch_size=params.BATCH_SIZE,
            n_epochs=params.N_EPOCHS,
            gamma=params.GAMMA,
            gae_lambda=params.GAE_LAMBDA,
            clip_range=params.CLIP_RANGE,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            seed=params.SEED
        )
    
    # Create simplified callback with adjusted episode count
    callback = SimpleEpisodeCallback(
        monitor=monitor,
        total_episodes=10000,  
        check_freq=params.CHECKPOINT_FREQ,
        save_path=params.CHECKPOINT_DIR,
        verbose=2
    )
    
    # If we're resuming, set the episode count in the callback
    if starting_episode > 0:
        callback.episode_count = starting_episode
        # Also update the monitor's episode count if not already set
        if monitor.episode_count < starting_episode:
            monitor.episode_count = starting_episode

    try:
        # Start training
        logger.info(f"\nContinuing training for {total_episodes} more episodes...")
        
        # Set a high number for total_timesteps
        max_timesteps = 20000000  # 20 million timesteps
        
        model.learn(
            total_timesteps=max_timesteps,
            callback=callback,
            tb_log_name=run_name,
            reset_num_timesteps=False  # Don't reset timesteps when resuming
        )
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted! Saving current model...\n")
    except Exception as e:
        logger.error(f"\n\nError during training: {e}\n")
        import traceback
        traceback.print_exc()
    finally:
        # Save the final model
        try:
            final_model_path = os.path.join(params.CHECKPOINT_DIR, f"{run_name}_final")
            model.save(final_model_path)
            logger.info(f"✅ Final model saved to {final_model_path}")
            
            # Save the environment normalization stats
            env_stats_path = os.path.join(params.CHECKPOINT_DIR, f"{run_name}_env_final.pkl")
            env.save(env_stats_path)
            logger.info(f"✅ Environment normalization stats saved to {env_stats_path}")
        except Exception as e:
            logger.error(f"❌ Error saving final model: {e}")

        # Generate and save final metrics
        try:
            current_episode = starting_episode + (callback.episode_count - starting_episode)
            monitor._save_backup(current_episode)
            logger.info(f"✅ Final metrics saved for episode {current_episode}.")
        except Exception as e:
            logger.error(f"❌ Error saving final metrics: {e}")

        # Close environment and monitoring
        try:
            env.close()
            monitor.close()
            logger.info("✅ Environment and monitor closed.")
        except Exception as e:
            logger.error(f"❌ Error closing environment: {e}")

        logger.info("\n🎉 Training completed!")
        
        return model, env


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CARLA RL Training')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--debug', action='store_true', help='Enable route debugging')
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Run training with specified parameters
    train(total_episodes=10000, debug_route=args.debug, resume_from=args.resume)