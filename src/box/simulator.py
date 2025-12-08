<<<<<<< HEAD
import sys
import os
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# 将根目录加入Python路径
sys.path.insert(0, project_root)
=======
>>>>>>> origin/main
import gymnasium as gym
from gymnasium import spaces
import pygame
import mujoco
<<<<<<< HEAD
import numpy as np
import scipy
import matplotlib
=======
import os
import numpy as np
import scipy
import matplotlib
import sys
>>>>>>> origin/main
import importlib
import shutil
import inspect
import pathlib
from datetime import datetime
import copy
from collections import defaultdict
import xml.etree.ElementTree as ET

<<<<<<< HEAD
from stable_baselines3 import PPO  # required to load a trained LLC policy in HRL approach

# -------------------------- 修复导入路径 --------------------------
# 统一使用项目根目录的绝对导入
from src.box.perception.base import Perception
# 若rendering.py中无Camera/Context，需确认文件是否存在或类是否正确定义
from src.box.utils.rendering import Camera, Context  # 确保rendering.py中包含这两个类
from src.box.utils.functions import output_path, parent_path, is_suitable_package_name, parse_yaml, write_yaml
# ------------------------------------------------------------------


class Simulator(gym.Env):
    """
    The Simulator class contains functionality to build a standalone Python package from a config file. 
    The built package integrates a biomechanical model, a task model, and a perception model into one 
    simulator that implements a gym.Env interface.
    
    Key features:
    - Support for Hierarchical Reinforcement Learning (HRL) with Low-Level Controller (LLC)
    - MuJoCo simulation integration with Gymnasium API
    - Configurable rendering (rgb_array, rgb_array_list, human)
    - Modular design for perception, biomechanical and task models
    """
    # Version format: X.Y.Z (major.minor.patch)
    version = "1.1.0"

    @classmethod
    def get_class(cls, *args):
        """ 
        Returns a class from given module path strings. 
        The last element in args should contain the class name.
        
        Args:
            *args: Module path components + class name
        
        Returns:
            type: Requested class object
        """
        # Handle module path and class name parsing
        modules = ".".join(args[:-1])
        if "." in args[-1]:
            splitted = args[-1].split(".")
            if modules == "":
                modules = ".".join(splitted[:-1])
            else:
                modules += "." + ".".join(splitted[:-1])
            cls_name = splitted[-1]
        else:
            cls_name = args[-1]
        
        module = cls.get_module(modules)
        return getattr(module, cls_name)

    @classmethod
    def get_module(cls, *args):
        """ 
        Returns a module from given path strings.
        
        Args:
            *args: Module path components
        
        Returns:
            module: Imported module object
        """
        src = __name__.split(".")[0]
        modules = ".".join(args)
        return importlib.import_module(src + "." + modules)

    @classmethod
    def build(cls, config):
        """ 
        Builds a simulator package from a configuration dict or YAML file path.
        
        Args:
            config: Either a dict with configuration or path to YAML config file
        
        Returns:
            str: Path to the built simulator folder
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            AssertionError: If required config fields are missing
            NameError: If package name is invalid
        """
        # Parse config file if path is provided
        if isinstance(config, str):
            if not os.path.isfile(config):
                raise FileNotFoundError(f"Config file {config} does not exist")
            config = parse_yaml(config)

        # Validate required config fields
        required_fields = [
            ("simulation", "Simulation specs must be defined in config"),
            ("simulation.bm_model", "Biomechanical model must be defined in config"),
            ("simulation.task", "Task must be defined in config"),
            ("simulation.run_parameters", "Run parameters must be defined in config"),
            ("simulation.run_parameters.action_sample_freq", 
             "Action sampling frequency must be defined in run parameters")
        ]
        
        for field, msg in required_fields:
            keys = field.split(".")
            current = config
            try:
                for key in keys:
                    current = current[key]
            except (KeyError, TypeError):
                raise AssertionError(msg)

        # Prepare config
        run_parameters = config["simulation"]["run_parameters"].copy()
        config["version"] = cls.version

        # Determine simulator folder location
        if "simulator_folder" in config:
            simulator_folder = os.path.normpath(config["simulator_folder"])
        else:
            simulator_folder = os.path.join(output_path(), config["simulator_name"])

        # Validate and set package name
        if "package_name" not in config:
            config["package_name"] = config["simulator_name"]
        
        if not is_suitable_package_name(config["package_name"]):
            raise NameError(
                "Package name (package_name/simulator_name) is invalid. "
                "Use lowercase letters and underscores only, cannot start with a number."
            )

        # Set gym registration name
        config["gym_name"] = f"uitb:{config['package_name']}-v0"

        # Create simulator package structure
        cls._clone(simulator_folder, config["package_name"])

        # Initialize and integrate task model
        task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
        task_kwargs = config["simulation"]["task"].get("kwargs", {})
        unity_exec = task_kwargs.get("unity_executable", None)
        task_cls.clone(simulator_folder, config["package_name"], app_executable=unity_exec)
        simulation = task_cls.initialise(task_kwargs)

        # Set MuJoCo compiler defaults
        compiler_defaults = {
            "inertiafromgeom": "auto",
            "balanceinertia": "true",
            "boundmass": "0.001",
            "boundinertia": "0.001",
            "inertiagrouprange": "0 1"
        }
        compiler = simulation.find("compiler")
        if compiler is None:
            ET.SubElement(simulation, "compiler", compiler_defaults)
        else:
            compiler.attrib.update(compiler_defaults)

        # Integrate biomechanical model
        bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
        bm_cls.clone(simulator_folder, config["package_name"])
        bm_cls.insert(simulation)

        # Integrate perception modules
        for module_cfg in config["simulation"].get("perception_modules", []):
            module_cls = cls.get_class("perception", module_cfg["cls"])
            module_kwargs = module_cfg.get("kwargs", {})
            module_cls.clone(simulator_folder, config["package_name"])
            module_cls.insert(simulation, **module_kwargs)

        # Clone RL library files
        if "rl" in config:
            rl_cls = cls.get_class("rl", config["rl"]["algorithm"])
            rl_cls.clone(simulator_folder, config["package_name"])

        # Save initial simulation XML
        simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")
        with open(f"{simulation_file}.xml", 'w') as file:
            simulation.write(file, encoding='unicode')

        # Initialize simulator to get final model state
        model, _, _, _, _, _ = cls._initialise(
            config, simulator_folder, {**run_parameters, "build": True}
        )

        # Save updated XML and binary model (faster loading)
        mujoco.MjModel.from_xml_path(f"{simulation_file}.xml")
        mujoco.mj_saveLastXML(f"{simulation_file}.xml", model)
        mujoco.mj_saveModel(model, f"{simulation_file}.mjcf", None)

        # Record build time and save config
        config["built"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_yaml(config, os.path.join(simulator_folder, "config.yaml"))

        return simulator_folder

    @classmethod
    def _clone(cls, simulator_folder, package_name):
        """ 
        Creates simulator package directory structure and copies required files.
        
        Args:
            simulator_folder: Base folder for the simulator
            package_name: Name of the Python package
        """
        # Create package directory
        pkg_dir = os.path.join(simulator_folder, package_name)
        os.makedirs(pkg_dir, exist_ok=True)

        # Copy simulator class file
        src_file = pathlib.Path(inspect.getfile(cls))
        shutil.copyfile(src_file, os.path.join(pkg_dir, src_file.name))

        # Create __init__.py with gym registration
        init_content = f"""from .simulator import Simulator

from gymnasium.envs.registration import register
import pathlib

module_folder = pathlib.Path(__file__).parent
simulator_folder = module_folder.parent
kwargs = {{'simulator_folder': simulator_folder}}
register(
    id=f'{{module_folder.stem}}-v0',
    entry_point=f'{{module_folder.stem}}.simulator:Simulator',
    kwargs=kwargs
)
"""
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as file:
            file.write(init_content)

        # Copy utility modules
        utils_src = os.path.join(parent_path(src_file), "utils")
        utils_dst = os.path.join(pkg_dir, "utils")
        shutil.copytree(utils_src, utils_dst, dirs_exist_ok=True)

        # Copy train/test modules
        for subdir in ["train", "test"]:
            src = os.path.join(parent_path(src_file), subdir)
            dst = os.path.join(pkg_dir, subdir)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    @classmethod
    def _initialise(cls, config, simulator_folder, run_parameters):
        """ 
        Initializes MuJoCo model/data and all simulator components.
        
        Args:
            config: Simulator configuration dict
            simulator_folder: Path to simulator folder
            run_parameters: Runtime parameters
        
        Returns:
            tuple: (model, data, task, bm_model, perception, callbacks)
        """
        # Load core components
        task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
        task_kwargs = config["simulation"]["task"].get("kwargs", {})
        
        bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
        bm_kwargs = config["simulation"]["bm_model"].get("kwargs", {})

        # Initialize perception modules
        perception_modules = {}
        for module_cfg in config["simulation"].get("perception_modules", []):
            module_cls = cls.get_class("perception", module_cfg["cls"])
            module_kwargs = module_cfg.get("kwargs", {})
            perception_modules[module_cls] = module_kwargs

        # Load MuJoCo model (try binary first, fall back to XML)
        simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")
        try:
            model = mujoco.MjModel.from_binary_path(f"{simulation_file}.mjcf")
        except (FileNotFoundError, mujoco.MujocoException):
            model = mujoco.MjModel.from_xml_path(f"{simulation_file}.xml")

        # Initialize MuJoCo data
        data = mujoco.MjData(model)

        # Calculate frame skip and dt
        run_parameters["frame_skip"] = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
        run_parameters["dt"] = model.opt.timestep * run_parameters["frame_skip"]

        # Initialize rendering context
        max_res = run_parameters.get("max_resolution", [1280, 960])
        run_parameters["rendering_context"] = Context(model, max_resolution=max_res)

        # Initialize callbacks
        callbacks = {}
        for cb in run_parameters.get("callbacks", []):
            cb_cls = cls.get_class(cb["cls"])
            callbacks[cb["name"]] = cb_cls(cb["name"], **cb["kwargs"])

        # Merge parameters for component initialization
        common_kwargs = {**run_parameters, **callbacks}
        
        # Initialize core components
        task = task_cls(model, data, **{**task_kwargs, **common_kwargs})
        bm_model = bm_cls(model, data,** {**bm_kwargs, **common_kwargs})
        perception = Perception(
            model, data, bm_model, perception_modules, common_kwargs
        )

        return model, data, task, bm_model, perception, callbacks

    @classmethod
    def get(cls, simulator_folder, render_mode="rgb_array", 
            render_mode_perception="embed", render_show_depths=False, 
            run_parameters=None, use_cloned=True):
        """ 
        Loads a pre-built simulator from folder.
        
        Args:
            simulator_folder: Path to simulator folder
            render_mode: Render mode (rgb_array, rgb_array_list, human)
            render_mode_perception: How to render perception modules (embed/separate/None)
            render_show_depths: Whether to show depth images
            run_parameters: Runtime parameters to override
            use_cloned: Whether to use cloned files or original source
        
        Returns:
            Simulator: Initialized simulator instance
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            RuntimeError: If simulator not built or version mismatch
        """
        # Load config
        config_path = os.path.join(simulator_folder, "config.yaml")
        try:
            config = parse_yaml(config_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load config: {e}")

        # Check if simulator is built
        if "built" not in config:
            raise RuntimeError("Simulator has not been built (missing 'built' timestamp in config)")

        # Add simulator folder to path
        if simulator_folder not in sys.path:
            sys.path.insert(0, simulator_folder)

        # Load simulator class
        try:
            gen_cls_cloned = getattr(importlib.import_module(config["package_name"]), "Simulator")
        except ImportError as e:
            raise RuntimeError(f"Failed to import simulator package: {e}")

        # Handle version checking
        _legacy_mode = False
        gen_cls_cloned_version = "0.0.0"
        
        if hasattr(gen_cls_cloned, "version"):
            gen_cls_cloned_version = gen_cls_cloned.version
        else:
            _legacy_mode = True
            gen_cls_cloned_version = gen_cls_cloned.id.split("-v")[-1]

        # Select class to use (cloned vs original)
        if use_cloned:
            gen_cls = gen_cls_cloned
        else:
            gen_cls = cls
            # Version compatibility check
            cloned_ver = gen_cls_cloned_version.split(".")
            current_ver = gen_cls.version.split(".")
            
            if cloned_ver[0] < current_ver[0]:
                raise RuntimeError(
                    f"Major version mismatch: Simulator v{gen_cls_cloned_version}, "
                    f"UITB v{gen_cls.version}. Use use_cloned=True or rebuild simulator."
                )
            elif cloned_ver[1] < current_ver[1]:
                print(f"Warning: Minor version mismatch - Simulator v{gen_cls_cloned_version}, "
                      f"UITB v{gen_cls.version}")

        # Initialize simulator
        try:
            if _legacy_mode:
                simulator = gen_cls(simulator_folder, run_parameters=run_parameters)
            else:
                simulator = gen_cls(
                    simulator_folder,
                    render_mode=render_mode,
                    render_mode_perception=render_mode_perception,
                    render_show_depths=render_show_depths,
                    run_parameters=run_parameters
                )
        except TypeError:
            # Fallback for older versions without perception render mode
            simulator = gen_cls(
                simulator_folder,
                render_mode=render_mode,
                render_show_depths=render_show_depths,
                run_parameters=run_parameters
            )

        return simulator

    def __init__(self, simulator_folder, render_mode="rgb_array", 
                 render_mode_perception="embed", render_show_depths=False, 
                 run_parameters=None):
        """ 
        Initializes a Simulator instance.
        
        Args:
            simulator_folder: Path to simulator folder
            render_mode: Render mode (rgb_array, rgb_array_list, human)
            render_mode_perception: How to render perception modules (embed/separate/None)
            render_show_depths: Whether to show depth images
            run_parameters: Runtime parameters to override
        
        Raises:
            FileNotFoundError: If simulator folder doesn't exist
        """
        super().__init__()
        
        # Validate simulator folder
        if not os.path.exists(simulator_folder):
            raise FileNotFoundError(f"Simulator folder {simulator_folder} does not exist")
        self._simulator_folder = simulator_folder

        # Load config
        self._config = parse_yaml(os.path.join(self._simulator_folder, "config.yaml"))

        # Merge run parameters (config defaults + runtime overrides)
        self._run_parameters = self._config["simulation"]["run_parameters"].copy()
        self._run_parameters.update(run_parameters or {})

        # Initialize core simulation components
        init_results = self._initialise(
            self._config, self._simulator_folder, self._run_parameters
        )
        self._model, self._data, self.task, self.bm_model, self.perception, self.callbacks = init_results

        # Initialize action/observation spaces
        self.action_space = self._initialise_action_space()
        self.observation_space = self._initialise_observation_space()

        # Episode statistics
        self._episode_statistics = {
            "length (seconds)": 0,
            "length (steps)": 0,
            "reward": 0
        }

        # Rendering setup
        self._render_mode = render_mode
        self._render_mode_perception = render_mode_perception
        self._render_show_depths = render_show_depths
        self._render_stack = []
        self._render_stack_perception = defaultdict(list)
        self._render_stack_pop = True
        self._render_stack_clean_at_reset = True
        self._render_screen_size = None
        self._render_window = None
        self._render_clock = None

        # Initialize GUI camera
        self._GUI_camera = Camera(
            self._run_parameters["rendering_context"],
            self._model,
            self._data,
            camera_id='for_testing',
            dt=self._run_parameters["dt"]
        )

        # HRL (Hierarchical RL) setup
        self._setup_hrl()

    def _setup_hrl(self):
        """Sets up Hierarchical RL components if configured"""
        if 'llc' not in self.config:
            return

        # Load LLC simulator
        llc_sim_name = self.config["llc"]["simulator_name"]
        llc_sim_folder = os.path.join(output_path(), llc_sim_name)
        
        if llc_sim_folder not in sys.path:
            sys.path.insert(0, llc_sim_folder)
        
        if not os.path.exists(llc_sim_folder):
            raise FileNotFoundError(f"LLC simulator folder {llc_sim_folder} does not exist")

        # Load LLC policy
        llc_checkpoint_dir = os.path.join(llc_sim_folder, 'checkpoints')
        llc_checkpoint = self.config["llc"]["checkpoint"]
        llc_checkpoint_path = os.path.join(llc_checkpoint_dir, llc_checkpoint)
        
        try:
            print(f"Loading LLC model: {llc_checkpoint_path}")
            self.llc_model = PPO.load(llc_checkpoint_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"LLC checkpoint {llc_checkpoint} not found in {llc_checkpoint_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLC model: {str(e)}")

        # Update action space for HRL
        self.action_space = self._initialise_HRL_action_space()

        # HRL parameters (from config with defaults)
        self._max_steps = self.config["llc"].get("llc_ratio", 100)
        self._dwell_threshold = self.config["llc"].get(
            "dwell_threshold", int(0.5 * self._max_steps)
        )
        self._target_radius = self.config["llc"].get("target_radius", 0.05)

        # Precompute joint information (performance optimization)
        joints = self.config["llc"]["joints"]
        self._independent_jnt_ids = []
        self._independent_dofs = []
        
        for joint in joints:
            joint_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, joint
            )
            jnt_type = self._model.jnt_type[joint_id]
            
            if jnt_type not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                raise NotImplementedError(
                    f"Only hinge/slide joints are supported. Joint {joint} is "
                    f"{mujoco.mjtJoint(jnt_type).name}"
                )
            
            self._independent_jnt_ids.append(joint_id)
            self._independent_dofs.append(self._model.jnt_qposadr[joint_id])

        # Precompute joint ranges (vectorized operations)
        self._jnt_range = self._model.jnt_range[self._independent_jnt_ids].astype(np.float32)
        self._jnt_range_diff = self._jnt_range[:, 1] - self._jnt_range[:, 0]
        # Avoid division by zero
        self._jnt_range_diff[self._jnt_range_diff == 0] = 1e-6

    def _normalise_qpos(self, qpos):
        """
        Normalizes joint positions to [-1, 1] range using precomputed joint ranges.
        
        Args:
            qpos: Raw joint positions
        
        Returns:
            np.ndarray: Normalized joint positions
        """
        # Normalize to [0, 1] then to [-1, 1] (vectorized for performance)
        norm_01 = (qpos - self._jnt_range[:, 0]) / self._jnt_range_diff
        return (norm_01 - 0.5) * 2

    def _initialise_action_space(self):
        """
        Initializes default action space (all actuators [-1, 1]).
        
        Returns:
            spaces.Box: Action space definition
        """
        num_actuators = self.bm_model.nu + self.perception.nu
        low = np.full(num_actuators, -1.0, dtype=np.float32)
        high = np.full(num_actuators, 1.0, dtype=np.float32)
        return spaces.Box(low=low, high=high)

    def _initialise_HRL_action_space(self):
        """
        Initializes action space for HRL (combines BM and perception actions).
        
        Returns:
            spaces.Box: HRL action space definition
        """
        # Biomechanical model action space
        bm_low = np.full(self.bm_model.nu, -1.0, dtype=np.float32)
        bm_high = np.full(self.bm_model.nu, 1.0, dtype=np.float32)
        
        # Perception module action space
        perception_low = np.full(self.perception.nu, -1.0, dtype=np.float32)
        perception_high = np.full(self.perception.nu, 1.0, dtype=np.float32)
        
        # Combine spaces
        low = np.concatenate([bm_low, perception_low])
        high = np.concatenate([bm_high, perception_high])
        
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _initialise_observation_space(self):
        """
        Initializes observation space (perception modules + stateful info).
        
        Returns:
            spaces.Dict: Observation space definition
        """
        # Get sample observation to build space
        observation = self.get_observation()
        obs_dict = {}

        # Add perception modules
        for module in self.perception.perception_modules:
            obs_dict[module.modality] = spaces.Box(
                dtype=np.float32,** module.get_observation_space_params()
            )

        # Add stateful information (ensure non-zero shape)
        if "stateful_information" in observation:
            state_params = self.task.get_stateful_information_space_params()
            # Ensure minimum shape of (1,) to avoid SB3 errors
            if state_params["shape"] == (0,):
                state_params["shape"] = (1,)
            obs_dict["stateful_information"] = spaces.Box(
                dtype=np.float32,** state_params
            )

        return spaces.Dict(obs_dict)

    def _get_qpos(self):
        """
        Gets normalized joint positions for independent joints.
        
        Returns:
            np.ndarray: Normalized joint positions
        """
        qpos = self._data.qpos[self._independent_dofs].copy()
        return self._normalise_qpos(qpos)

    def step(self, action):
        """
        Advances simulation by one step (supports HRL and normal RL).
        
        Args:
            action: Action vector from policy
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # HRL mode (with LLC)
        if 'llc' in self.config:
            self.task._target_qpos = action
            self._steps = 0
            total_reward = 0

            # LLC control loop
            while self._steps < self._max_steps:
                # Get LLC observation and predict action
                llc_obs = self.get_llcobservation(action)
                llc_action, _ = self.llc_model.predict(llc_obs, deterministic=True)

                # Apply controls
                self.bm_model.set_ctrl(self._model, self._data, llc_action)
                self.perception.set_ctrl(
                    self._model, self._data, action[self.bm_model.nu:]
                )

                # Step simulation
                mujoco.mj_step(
                    self._model, self._data, nstep=self._run_parameters["frame_skip"]
                )

                # Update components
                self.bm_model.update(self._model, self._data)
                self.perception.update(self._model, self._data)
                
                # Calculate reward and check termination
                reward, terminated, truncated, info = self.task.update(
                    self._model, self._data
                )
                reward -= self.bm_model.get_effort_cost(self._model, self._data)
                total_reward += reward

                # Get observation
                obs = self.get_observation(info)

                # Render if needed
                if self._render_mode == "rgb_array_list":
                    self._render_stack.append(self._GUI_rendering())
                elif self._render_mode == "human":
                    self._GUI_rendering_pygame()

                # Check termination conditions
                if terminated or truncated:
                    break
                
                # Task-specific termination checks
                if "target_spawned" in info or "new_button_generated" in info:
                    if info.get("target_hit", False):
                        break

                # Check joint position error threshold
                qpos = self._get_qpos()
                dist = np.abs(action - qpos)
                if np.all(dist < self._target_radius):
                    break

                self._steps += 1

            # Use total reward from LLC loop
            reward = total_reward

        # Normal RL mode
        else:
            # Apply controls
            self.bm_model.set_ctrl(self._model, self._data, action[:self.bm_model.nu])
            self.perception.set_ctrl(
                self._model, self._data, action[self.bm_model.nu:]
            )

            # Step simulation
            mujoco.mj_step(
                self._model, self._data, nstep=self._run_parameters["frame_skip"]
            )

            # Update components
            self.bm_model.update(self._model, self._data)
            self.perception.update(self._model, self._data)

            # Calculate reward and termination
            reward, terminated, truncated, info = self.task.update(
                self._model, self._data
            )
            
            # Add effort cost
            effort_cost = self.bm_model.get_effort_cost(self._model, self._data)
            info["EffortCost"] = effort_cost
            reward -= effort_cost

            # Get observation
            obs = self.get_observation(info)

            # Render if needed
            if self._render_mode == "rgb_array_list":
                self._render_stack.append(self._GUI_rendering())
            elif self._render_mode == "human":
                self._GUI_rendering_pygame()

        # Update episode statistics
        self._episode_statistics["length (seconds)"] += self._run_parameters["dt"]
        self._episode_statistics["length (steps)"] += 1
        self._episode_statistics["reward"] += reward

        return obs, reward, terminated, truncated, info

    def get_observation(self, info=None):
        """
        Gets observation from perception modules and task state.
        
        Args:
            info: Additional info from task update
        
        Returns:
            dict: Observation dictionary
        """
        # Get perception observations
        observation = self.perception.get_observation(self._model, self._data, info)

        # Add stateful information (ensure non-empty)
        stateful_info = self.task.get_stateful_information(self._model, self._data)
        if stateful_info.size > 0:
            observation["stateful_information"] = stateful_info
        elif "stateful_information" not in observation:
            # Add dummy state to avoid SB3 errors
            observation["stateful_information"] = np.array([0.0], dtype=np.float32)

        return observation

    def get_llcobservation(self, action):
        """
        Gets observation for LLC (no vision, joint position error).
        
        Args:
            action: Target joint position from HRL
        
        Returns:
            dict: LLC observation dictionary
        """
        # Get base observation and remove vision
        observation = self.perception.get_observation(self._model, self._data)
        observation.pop("vision", None)

        # Calculate joint position error
        qpos = self._get_qpos()
        qpos_diff = action - qpos

        # Add stateful information (joint error)
        observation["stateful_information"] = qpos_diff.astype(np.float32)

        return observation

    def reset(self, seed=None, options=None):
        """
        Resets simulator to initial state.
        
        Args:
            seed: Random seed
            options: Reset options
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)

        # Reset MuJoCo data
        mujoco.mj_resetData(self._model, self._data)

        # Reset components
        self.bm_model.reset(self._model, self._data)
        self.perception.reset(self._model, self._data)
        info = self.task.reset(self._model, self._data)

        # Forward pass to update state
        mujoco.mj_forward(self._model, self._data)

        # Reset episode statistics
        self._episode_statistics = {
            "length (seconds)": 0,
            "length (steps)": 0,
            "reward": 0
        }

        # Render initial frame
        if self._render_mode == "rgb_array_list":
            if self._render_stack_clean_at_reset:
                self._render_stack = []
                self._render_stack_perception = defaultdict(list)
            self._render_stack.append(self._GUI_rendering())
        elif self._render_mode == "human":
            self._GUI_rendering_pygame()

        return self.get_observation(), info

    def render(self):
        """
        Renders simulation frame based on render mode.
        
        Returns:
            np.ndarray/list/None: Rendered frame(s) or None
        """
        if self._render_mode == "rgb_array_list":
            render_stack = self._render_stack.copy()
            if self._render_stack_pop:
                self._render_stack = []
            return render_stack
        elif self._render_mode == "rgb_array":
            return self._GUI_rendering()
        return None

    def get_render_stack_perception(self):
        """
        Gets perception render stack (for separate perception rendering).
        
        Returns:
            defaultdict: Perception render stack
        """
        return copy.deepcopy(self._render_stack_perception)

    def _GUI_rendering(self):
        """
        Renders GUI frame with optional perception module overlays.
        
        Returns:
            np.ndarray: RGB image array
        """
        # Render main camera
        img, _ = self._GUI_camera.render()

        # Embed perception images into main frame
        if self._render_mode_perception == "embed":
            perception_images = self.perception.get_renders()
            
            if len(perception_images) > 0:
                img_h, img_w = img.shape[:2]
                desired_h = np.round(img_h / len(perception_images)).astype(int)
                max_w = np.round(0.2 * img_w).astype(int)

                # Resize and embed perception images
                resampled_imgs = []
                for perc_img in perception_images:
                    # Convert depth maps to RGB heatmaps if enabled
                    if perc_img.ndim == 2:
                        if self._render_show_depths:
                            # Create heatmap from depth
                            cmap = matplotlib.cm.jet
                            norm = matplotlib.colors.Normalize(
                                vmin=perc_img.min(), vmax=perc_img.max()
                            )
                            perc_img = cmap(norm(perc_img))[:, :, :3] * 255
                            perc_img = perc_img.astype(np.uint8)
                        else:
                            continue

                    # Calculate resize factor
                    h, w = perc_img.shape[:2]
                    scale = min(desired_h / h, max_w / w)
                    new_h = np.round(h * scale).astype(int)
                    new_w = np.round(w * scale).astype(int)

                    # Resample image (vectorized for performance)
                    resampled = np.zeros((new_h, new_w, perc_img.shape[2]), dtype=np.uint8)
                    for c in range(perc_img.shape[2]):
                        resampled[:, :, c] = scipy.ndimage.zoom(perc_img[:, :, c], scale, order=0)
                    
                    resampled_imgs.append(resampled)

                # Overlay resampled images (bottom-right to top-right)
                y_pos = img_h
                for res_img in resampled_imgs:
                    h, w = res_img.shape[:2]
                    y_start = max(0, y_pos - h)
                    x_start = max(0, img_w - w)
                    
                    # Ensure we don't go out of bounds
                    img[y_start:y_start+h, x_start:x_start+w] = res_img
                    y_pos = y_start

        # Store separate perception renders
        elif self._render_mode_perception == "separate":
            for module, cameras in self.perception.cameras_dict.items():
                for camera in cameras:
                    for img_arr in camera.render():
                        if img_arr is not None:
                            key = f"{module.modality}/{type(camera).__name__}"
                            self._render_stack_perception[key].append(img_arr)

        return img

    def _GUI_rendering_pygame(self):
        """Renders frame to Pygame window (human render mode)"""
        # Get rendered image and transpose for Pygame
        rgb_array = np.transpose(self._GUI_rendering(), (1, 0, 2))
        
        # Initialize Pygame window if needed
        if self._render_screen_size is None:
            self._render_screen_size = rgb_array.shape[:2]
        
        if self._render_window is None:
            pygame.init()
            pygame.display.init()
            self._render_window = pygame.display.set_mode(self._render_screen_size)
        
        if self._render_clock is None:
            self._render_clock = pygame.time.Clock()

        # Validate image size
        if self._render_screen_size != rgb_array.shape[:2]:
            raise ValueError(
                f"Render size mismatch: Expected {self._render_screen_size}, "
                f"got {rgb_array.shape[:2]}"
            )

        # Update Pygame window
        surf = pygame.surfarray.make_surface(rgb_array)
        self._render_window.blit(surf, (0, 0))
        pygame.event.pump()
        self._render_clock.tick(self.fps)
        pygame.display.flip()

    def get_state(self):
        """
        Gets full simulator state (for logging/evaluation).
        
        Returns:
            dict: Simulator state including MuJoCo data and component states
        """
        # Base MuJoCo state
        state = {
            "timestep": self._data.time,
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "qacc": self._data.qacc.copy(),
            "act_force": self._data.actuator_force.copy(),
            "act": self._data.act.copy(),
            "ctrl": self._data.ctrl.copy()
        }

        # Add component-specific state
        state.update(self.task.get_state(self._model, self._data))
        state.update(self.bm_model.get_state(self._model, self._data))
        state.update(self.perception.get_state(self._model, self._data))

        return state

    def close(self, **kwargs):
        """Cleans up simulator resources"""
        # Clean up components
        self.task.close(** kwargs)
        self.perception.close(**kwargs)
        self.bm_model.close(**kwargs)

        # Clean up Pygame
        if self._render_window is not None:
            pygame.display.quit()
            pygame.quit()
            self._render_window = None
            self._render_clock = None

    # Property getters (read-only)
    @property
    def fps(self):
        return self._GUI_camera._fps

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @property
    def run_parameters(self):
        # Exclude non-serializable objects from copy
        exclude = {"rendering_context"}
        run_params = {
            k: copy.deepcopy(v) 
            for k, v in self._run_parameters.items() 
            if k not in exclude
        }
        run_params["rendering_context"] = self._run_parameters["rendering_context"]
        return run_params

    @property
    def simulator_folder(self):
        return self._simulator_folder

    @property
    def render_mode(self):
        return self._render_mode

    # Ensure proper cleanup on object deletion
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
=======
from stable_baselines3 import PPO  #required to load a trained LLC policy in HRL approach

from .perception.base import Perception
from .utils.rendering import Camera, Context
from .utils.functions import output_path, parent_path, is_suitable_package_name, parse_yaml, write_yaml


class Simulator(gym.Env):
  """
  The Simulator class contains functionality to build a standalone Python package from a config file. The built package
   integrates a biomechanical model, a task model, and a perception model into one simulator that implements a gym.Env
   interface.
  """

  # May be useful for later, the three digit number suffix is of format X.Y.Z where X is a major version.
  version = "1.1.0"

  @classmethod
  def get_class(cls, *args):
    """ Returns a class from given strings. The last element in args should contain the class name. """
    # TODO check for incorrect module names etc
    modules = ".".join(args[:-1])
    if "." in args[-1]:
      splitted = args[-1].split(".")
      if modules == "":
        modules = ".".join(splitted[:-1])
      else:
        modules += "." + ".".join(splitted[:-1])
      cls_name = splitted[-1]
    else:
      cls_name = args[-1]
    module = cls.get_module(modules)
    return getattr(module, cls_name)

  @classmethod
  def get_module(cls, *args):
    """ Returns a module from given strings. """
    src = __name__.split(".")[0]
    modules = ".".join(args)
    return importlib.import_module(src + "." + modules)

  @classmethod
  def build(cls, config):
    """ Builds a simulator based on a config. The input 'config' may be a dict (parsed from YAML) or path to a YAML file

    Args:
      config:
        - A dict containing configuration information. See example configs in folder uitb/configs/
        - A path to a config file
    """

    # If config is a path to the config file, parse it first
    if isinstance(config, str):
      if not os.path.isfile(config):
        raise FileNotFoundError(f"Given config file {config} does not exist")
      config = parse_yaml(config)

    # Make sure required things are defined in config
    assert "simulation" in config, "Simulation specs (simulation) must be defined in config"
    assert "bm_model" in config["simulation"], "Biomechanical model (bm_model) must be defined in config"
    assert "task" in config["simulation"], "task (task) must be defined in config"

    assert "run_parameters" in config["simulation"], "Run parameters (run_parameters) must be defined in config"
    run_parameters = config["simulation"]["run_parameters"].copy()
    assert "action_sample_freq" in run_parameters, "Action sampling frequency (action_sample_freq) must be defined " \
                                                   "in run parameters"

    # Set simulator version
    config["version"] = cls.version

    # Save generated simulators to uitb/simulators
    if "simulator_folder" in config:
      simulator_folder = os.path.normpath(config["simulator_folder"])
    else:
      simulator_folder = os.path.join(output_path(), config["simulator_name"])

    # If 'package_name' is not defined use 'simulator_name'
    if "package_name" not in config:
      config["package_name"] = config["simulator_name"]
    if not is_suitable_package_name(config["package_name"]):
      raise NameError("Package name defined in the config file (either through 'package_name' or 'simulator_name') is "
                      "not a suitable Python package name. Use only lower-case letters and underscores instead of "
                      "spaces, and the name cannot start with a number.")

    # The name used in gym has a suffix -v0
    config["gym_name"] = "uitb:" + config["package_name"] + "-v0"

    # Create a simulator in the simulator folder
    cls._clone(simulator_folder, config["package_name"])

    # Load task class
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_cls.clone(simulator_folder, config["package_name"], app_executable=config["simulation"]["task"].get("kwargs", {}).get("unity_executable", None))
    simulation = task_cls.initialise(config["simulation"]["task"].get("kwargs", {}))

    # Set some compiler options
    # TODO: would make more sense to have a separate "environment" class / xml file that defines all these defaults,
    #  including e.g. cameras, lighting, etc., so that they could be easily changed. Task and biomechanical model would
    #  be integrated into that object
    compiler_defaults = {"inertiafromgeom": "auto", "balanceinertia": "true", "boundmass": "0.001",
                         "boundinertia": "0.001", "inertiagrouprange": "0 1"}
    compiler = simulation.find("compiler")
    if compiler is None:
      ET.SubElement(simulation, "compiler", compiler_defaults)
    else:
      compiler.attrib.update(compiler_defaults)

    # Load biomechanical model class
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_cls.clone(simulator_folder, config["package_name"])
    bm_cls.insert(simulation)

    # Add perception modules
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      module_cls.clone(simulator_folder, config["package_name"])
      module_cls.insert(simulation, **module_kwargs)

    # Clone also RL library files so the package will be completely standalone
    rl_cls = cls.get_class("rl", config["rl"]["algorithm"])
    rl_cls.clone(simulator_folder, config["package_name"])

    # TODO read the xml file directly from task.getroot() instead of writing it to a file first; need to input a dict
    #  of assets to mujoco.MjModel.from_xml_path
    simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")
    with open(simulation_file+".xml", 'w') as file:
      simulation.write(file, encoding='unicode')

    # Initialise the simulator
    model, _, _, _, _, _ = \
      cls._initialise(config, simulator_folder, {**run_parameters, "build": True})

    # Now that simulator has been initialised, everything should be set. Now we want to save the xml file again, but
    # mujoco only is able to save the latest loaded xml file (which is either the task or bm model xml files which are
    # are read in their __init__ functions), hence we need to read the file we've generated again before saving the
    # modified model
    mujoco.MjModel.from_xml_path(simulation_file+".xml")
    mujoco.mj_saveLastXML(simulation_file+".xml", model)

    # Save the modified model also as binary for faster loading
    mujoco.mj_saveModel(model, simulation_file+".mjcf", None)

    # Input built time into config
    config["built"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save config
    write_yaml(config, os.path.join(simulator_folder, "config.yaml"))

    return simulator_folder

  @classmethod
  def _clone(cls, simulator_folder, package_name):
    """ Create a folder for the simulator being built, and copy or create relevant files.

    Args:
       simulator_folder: Location of the simulator.
       package_name: Name of the simulator (which is a python package).
    """

    # Create the folder
    dst = os.path.join(simulator_folder, package_name)
    os.makedirs(dst, exist_ok=True)

    # Copy simulator
    src = pathlib.Path(inspect.getfile(cls))
    shutil.copyfile(src, os.path.join(dst, src.name))

    # Create __init__.py with env registration
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from .simulator import Simulator\n\n")
      file.write("from gymnasium.envs.registration import register\n")
      file.write("import pathlib\n\n")
      file.write("module_folder = pathlib.Path(__file__).parent\n")
      file.write("simulator_folder = module_folder.parent\n")
      file.write("kwargs = {'simulator_folder': simulator_folder}\n")
      file.write("register(id=f'{module_folder.stem}-v0', entry_point=f'{module_folder.stem}.simulator:Simulator', kwargs=kwargs)\n")

    # Copy utils
    shutil.copytree(os.path.join(parent_path(src), "utils"), os.path.join(simulator_folder, package_name, "utils"),
                    dirs_exist_ok=True)
    # Copy train
    shutil.copytree(os.path.join(parent_path(src), "train"), os.path.join(simulator_folder, package_name, "train"),
                    dirs_exist_ok=True)
    # Copy test
    shutil.copytree(os.path.join(parent_path(src), "test"), os.path.join(simulator_folder, package_name, "test"),
                    dirs_exist_ok=True)

  @classmethod
  def _initialise(cls, config, simulator_folder, run_parameters):
    """ Initialise a simulator -- i.e., create a MjModel, MjData, and initialise all necessary variables.

    Args:
        config: A config dict.
        simulator_folder: Location of the simulator.
        run_parameters: Important run time variables that may also be used to override parameters.
    """

    # Get task class and kwargs
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_kwargs = config["simulation"]["task"].get("kwargs", {})

    # Get bm class and kwargs
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_kwargs = config["simulation"]["bm_model"].get("kwargs", {})

    # Initialise perception modules
    perception_modules = {}
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      perception_modules[module_cls] = module_kwargs

    # Get simulation file
    simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")

    # Load the mujoco model; try first with the binary model (faster, contains some parameters that may be lost when
    # re-saving xml files like body mass). For some reason the binary model fails to load in some situations (like
    # when the simulator has been built on a different computer)
    # TODO loading from binary disabled, weird problems (like a body not found from model when loaded from binary, but
    #  found correctly when model loaded from xml)
    # try:
    #  model = mujoco.MjModel.from_binary_path(simulation_file + ".mjcf")
    # except: # TODO what was the exception type
    model = mujoco.MjModel.from_xml_path(simulation_file + ".xml")

    # Initialise MjData
    data = mujoco.MjData(model)

    # Add frame skip and dt to run parameters
    run_parameters["frame_skip"] = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
    run_parameters["dt"] = model.opt.timestep*run_parameters["frame_skip"]

    # Initialise a rendering context, required for e.g. some vision modules
    run_parameters["rendering_context"] = Context(model,
                                                  max_resolution=run_parameters.get("max_resolution", [1280, 960]))

    # Initialise callbacks
    callbacks = {}
    for cb in run_parameters.get("callbacks", []):
      callbacks[cb["name"]] = cls.get_class(cb["cls"])(cb["name"], **cb["kwargs"])

    # Now initialise the actual classes; model and data are input to the inits so that stuff can be modified if needed
    # (e.g. move target to a specific position wrt to a body part)
    task = task_cls(model, data, **{**task_kwargs, **callbacks, **run_parameters})
    bm_model = bm_cls(model, data, **{**bm_kwargs, **callbacks, **run_parameters})
    perception = Perception(model, data, bm_model, perception_modules, {**callbacks, **run_parameters})

    return model, data, task, bm_model, perception, callbacks

  @classmethod
  def get(cls, simulator_folder, render_mode="rgb_array", render_mode_perception="embed", render_show_depths=False, run_parameters=None, use_cloned=True):
    """ Returns a Simulator that is located in given folder.

    Args:
      simulator_folder: Location of the simulator.
      render_mode: Whether render() will return a single rgb array (render_mode="rgb_array"),
        a list of rgb arrays (render_mode="rgb_array_list";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/render_collection.py),
        or None while the frames in a separate PyGame window are updated directly when calling
        step() or reset() (render_mode="human";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/human_rendering.py)).
      render_mode_perception: Whether images of visual perception modules should be directly embedded into main camera view ("embed"), stored as separate videos ("separate"), or not used at all [which allows to watch vision in Unity Editor if debug mode is enabled/standalone app is disabled] (None)
      render_show_depths: Whether depth images of visual perception modules should be included in rendering.
      run_parameters: Can be used to override parameters during run time.
      use_cloned: Can be useful for debugging. Set to False to use original files instead of the ones that have been
        cloned/copied during building phase.
    """

    # Read config file
    config_file = os.path.join(simulator_folder, "config.yaml")
    try:
      config = parse_yaml(config_file)
    except:
      raise FileNotFoundError(f"Could not open file {config_file}")

    # Make sure the simulator has been built
    if "built" not in config:
      raise RuntimeError("Simulator has not been built")

    # Make sure simulator_folder is in path (used to import gen_cls_cloned)
    if simulator_folder not in sys.path:
      sys.path.insert(0, simulator_folder)

    # Get Simulator class
    gen_cls_cloned = getattr(importlib.import_module(config["package_name"]), "Simulator")
    if hasattr(gen_cls_cloned, "version"):
      _legacy_mode = False
      gen_cls_cloned_version = gen_cls_cloned.version.split("-v")[-1]
    else:
      _legacy_mode = True
      gen_cls_cloned_version = gen_cls_cloned.id.split("-v")[-1]  #deprecated
    if use_cloned:
      gen_cls = gen_cls_cloned
    else:
      gen_cls = cls
      gen_cls_version = gen_cls.version.split("-v")[-1]

      if gen_cls_version.split(".")[0] > gen_cls_cloned_version.split(".")[0]:
        raise RuntimeError(
          f"""Severe version mismatch. The simulator '{config["simulator_name"]}' has version {gen_cls_cloned_version}, while your uitb package has version {gen_cls_version}.\nTo run with version {gen_cls_cloned_version}, set 'use_cloned=True'.""")
      elif gen_cls_version.split(".")[1] > gen_cls_cloned_version.split(".")[1]:
        print(
          f"""WARNING: Version mismatch. The simulator '{config["simulator_name"]}' has version {gen_cls_cloned_version}, while your uitb package has version {gen_cls_version}.\nTo run with version {gen_cls_version}, set 'use_cloned=True'.""")

    if _legacy_mode:
      _simulator = gen_cls(simulator_folder, run_parameters=run_parameters)
    else:
      try:
        _simulator = gen_cls(simulator_folder, render_mode=render_mode, render_mode_perception=render_mode_perception, render_show_depths=render_show_depths,
                          run_parameters=run_parameters)
      except TypeError:
        _simulator = gen_cls(simulator_folder, render_mode=render_mode, render_show_depths=render_show_depths,
                          run_parameters=run_parameters)

    # Return Simulator object
    return _simulator

  def __init__(self, simulator_folder, render_mode="rgb_array", render_mode_perception="embed", render_show_depths=False, run_parameters=None):
    """ Initialise a new `Simulator`.

    Args:
      simulator_folder: Location of a simulator.
      render_mode: Whether render() will return a single rgb array (render_mode="rgb_array"),
        a list of rgb arrays (render_mode="rgb_array_list";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/render_collection.py),
        or None while the frames in a separate PyGame window are updated directly when calling
        step() or reset() (render_mode="human";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/human_rendering.py)).
      render_mode_perception: Whether images of visual perception modules should be directly embedded into main camera view ("embed"), stored as separate videos ("separate"), or not used at all [which allows to watch vision in Unity Editor if debug mode is enabled/standalone app is disabled] (None)
      render_show_depths: Whether depth images of visual perception modules should be included in rendering.
      run_parameters: Can be used to override parameters during run time.
    """

    # Make sure simulator exists
    if not os.path.exists(simulator_folder):
      raise FileNotFoundError(f"Simulator folder {simulator_folder} does not exists")
    self._simulator_folder = simulator_folder

    # Read config
    self._config = parse_yaml(os.path.join(self._simulator_folder, "config.yaml"))

    # Get run parameters: these parameters can be used to override parameters used during training
    self._run_parameters = self._config["simulation"]["run_parameters"].copy()
    self._run_parameters.update(run_parameters or {})

    # Initialise simulation
    self._model, self._data, self.task, self.bm_model, self.perception, self.callbacks = \
      self._initialise(self._config, self._simulator_folder, self._run_parameters)

    # Set action space TODO for now we assume all actuators have control signals between [-1, 1]
    self.action_space = self._initialise_action_space()

    # Set observation space
    self.observation_space = self._initialise_observation_space()

    # Collect some episode statistics
    self._episode_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    # Initialise viewer
    self._GUI_camera = Camera(self._run_parameters["rendering_context"], self._model, self._data, camera_id='for_testing',
                         dt=self._run_parameters["dt"])

    self._render_mode = render_mode
    self._render_mode_perception = render_mode_perception  #whether perception camera views should be directly embedded into camera view of camera_id ("embed"), stored in self._render_stack_perception ("separate"), or not used at all "separate"), or not used at all [which allows to watch vision in Unity Editor if debug mode is enabled/standalone app is disabled] (None)
    self._render_stack = []  #only used if render_mode == "rgb_array_list"
    self._render_stack_perception = defaultdict(list)  #only used if render_mode == "rgb_array_list" and self._render_mode_perception == "separate"
    self._render_stack_pop = True  #If True, clear the render stack after .render() is called.
    self._render_stack_clean_at_reset = True  #If True, clear the render stack when .reset() is called.
    self._render_show_depths = render_show_depths  #If True, depth images of visual perception modules are included in GUI rendering.
    self._render_screen_size = None  #only used if render_mode == "human"
    self._render_window = None  #only used if render_mode == "human"
    self._render_clock = None  #only used if render_mode == "human"

    if 'llc' in self.config:  #if HRL approach is used
        llc_simulator_folder = os.path.join(output_path(), self.config["llc"]["simulator_name"])
        if llc_simulator_folder not in sys.path:
            sys.path.insert(0, llc_simulator_folder)
        if not os.path.exists(llc_simulator_folder):
            raise FileNotFoundError(f"Simulator folder {llc_simulator_folder} does not exists")
        llccheckpoint_dir = os.path.join(llc_simulator_folder, 'checkpoints')
        # Load policy TODO should create a load method for uitb.rl.BaseRLModel
        print(f'Loading model: {os.path.join(llccheckpoint_dir, self.config["llc"]["checkpoint"])}\n')
        self.llc_model = PPO.load(os.path.join(llccheckpoint_dir, self.config["llc"]["checkpoint"]))
        self.action_space = self._initialise_HRL_action_space()
        self._max_steps = self.config["llc"]["llc_ratio"]
        self._dwell_threshold = int(0.5*self._max_steps)
        self._target_radius = 0.05
        self._independent_dofs = []
        self._independent_joints = []
        joints = self.config["llc"]["joints"]
        for joint in joints:
          joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint)
          if self._model.jnt_type[joint_id] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            raise NotImplementedError(f"Only 'hinge' and 'slide' joints are supported, joint "
                                  f"{joint} is of type {mujoco.mjtJoint(self._model.jnt_type[joint_id]).name}")
          self._independent_dofs.append(self._model.jnt_qposadr[joint_id])
          self._independent_joints.append(joint_id)
        self._jnt_range = self._model.jnt_range[self._independent_joints]


    #To normalize joint ranges for llc
  def _normalise_qpos(self, qpos):
    # Normalise to [0, 1]
    qpos = (qpos - self._jnt_range[:, 0]) / (self._jnt_range[:, 1] - self._jnt_range[:, 0])
    # Normalise to [-1, 1]
    qpos = (qpos - 0.5) * 2
    return qpos

  def _initialise_action_space(self):
    """ Initialise action space. """
    num_actuators = self.bm_model.nu + self.perception.nu
    actuator_limits = np.ones((num_actuators,2)) * np.array([-1.0, 1.0])
    return spaces.Box(low=np.float32(actuator_limits[:, 0]), high=np.float32(actuator_limits[:, 1]))

  def _initialise_HRL_action_space(self):
    bm_nu = self.bm_model.nu
    bm_jnt_range = np.ones((bm_nu,2)) * np.array([-1.0, 1.0])
    perception_nu = self.perception.nu
    perception_jnt_range = np.ones((perception_nu,2)) * np.array([-1.0, 1.0])
    jnt_range = np.concatenate((bm_jnt_range, perception_jnt_range), axis=0)
    action_space = gym.spaces.Box(low=jnt_range[:,0], high=jnt_range[:,1])
    return action_space

  def _initialise_observation_space(self):
    """ Initialise observation space. """
    observation = self.get_observation()
    obs_dict = dict()
    for module in self.perception.perception_modules:
      obs_dict[module.modality] = spaces.Box(dtype=np.float32, **module.get_observation_space_params())
    if "stateful_information" in observation:
      obs_dict["stateful_information"] = spaces.Box(dtype=np.float32,
                                                    **self.task.get_stateful_information_space_params())
    return spaces.Dict(obs_dict)

  def _get_qpos(self, model, data):
    qpos = data.qpos[self._independent_dofs].copy()
    qpos = self._normalise_qpos(qpos)
    return qpos

  def step(self, action):
    """ Step simulation forward with given actions.

    Args:
      action: Actions sampled from a policy. Limited to range [-1, 1].
    """
    if 'llc' in self.config:  #if HRL approach is used
        self.task._target_qpos = action # action to pass to LLC
        self._steps = 0 # Initialise loop control to 0
        #acc_reward = 0 #To be used when rewards are being accumulated in llc steps

        while self._steps < self._max_steps: # loop for llc controls based on llc_ratio

            llc_action, _states = self.llc_model.predict(self.get_llcobservation(action), deterministic=True) # Get BM action from LLC
            # Set control for the bm model
            self.bm_model.set_ctrl(self._model, self._data, llc_action)

            # Set control for perception modules (e.g. eye movements)
            self.perception.set_ctrl(self._model, self._data, action[self.bm_model.nu:])

            # Advance the simulation
            mujoco.mj_step(self._model, self._data, nstep=int(self._run_parameters["frame_skip"])) # Number of timesteps to skip for LLC

            # Update bm model (e.g. update constraints); updates also effort model
            self.bm_model.update(self._model, self._data)

            # Update perception modules
            self.perception.update(self._model, self._data)

            dist = np.abs(action - self._get_qpos(self._model, self._data))

            # Update environment        
            reward, terminated, truncated, info = self.task.update(self._model, self._data)

            # Add an effort cost to reward
            reward -= self.bm_model.get_effort_cost(self._model, self._data)

            #acc_reward += reward #To be used when rewards are being accumulated in llc steps

            # Get observation
            obs = self.get_observation()

            # Add frame to stack
            if self._render_mode == "rgb_array_list":
              self._render_stack.append(self._GUI_rendering())
            elif self._render_mode == "human":
              self._GUI_rendering_pygame()

            if truncated or terminated:
                break

            # Pointing
            if "target_spawned" in info:
                if info["target_spawned"] or info["target_hit"]:
                    break

            # Choice Reaction
            elif "new_button_generated" in info:
                if info["new_button_generated"] or info["target_hit"]:
                    break

            self._steps += 1
            if np.all(dist < self._target_radius):
                break

        return obs, reward, terminated, truncated, info

    else:
            # Set control for the bm model
        self.bm_model.set_ctrl(self._model, self._data, action[:self.bm_model.nu])

        # Set control for perception modules (e.g. eye movements)
        self.perception.set_ctrl(self._model, self._data, action[self.bm_model.nu:])

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, nstep=self._run_parameters["frame_skip"])

        # Update bm model (e.g. update constraints); updates also effort model
        self.bm_model.update(self._model, self._data)

        # Update perception modules
        self.perception.update(self._model, self._data)

        # Update environment
        reward, terminated, truncated, info = self.task.update(self._model, self._data)

        # Add an effort cost to reward
        effort_cost = self.bm_model.get_effort_cost(self._model, self._data)
        info["EffortCost"] = effort_cost
        reward -= effort_cost

        # Get observation
        obs = self.get_observation(info)

        # Add frame to stack
        if self._render_mode == "rgb_array_list":
          self._render_stack.append(self._GUI_rendering())
        elif self._render_mode == "human":
          self._GUI_rendering_pygame()

        return obs, reward, terminated, truncated, info


  def get_observation(self, info=None):
    """ Returns an observation from the perception model.

    Returns:
      A dict with observations from individual perception modules. May also contain stateful information from a task.
    """

    # Get observation from perception
    observation = self.perception.get_observation(self._model, self._data, info)

    # Add any stateful information that is required
    stateful_information = self.task.get_stateful_information(self._model, self._data)
    if stateful_information.size > 0:  #TODO: define stateful_information (and encoder) that can be used as default, if no stateful information is provided (zero-size arrays do not work with sb3 currently...)
      observation["stateful_information"] = stateful_information

    return observation

  def get_llcobservation(self,action):
    """ Returns an observation from the perception model.

    Returns:
      A dict with observations from individual perception modules. May also contain stateful information from a task.
    """

    # Get observation from perception
    observation = self.perception.get_observation(self._model, self._data)
    
    # Remove Vision for LLC
    observation.pop("vision")
    qpos = self._get_qpos(self._model, self._data)
    qpos_diff = action - qpos
    
    # Stateful Information for LLC policy
    stateful_information = qpos_diff
    if stateful_information is not None:
      observation["stateful_information"] = stateful_information
    
    return observation


  def reset(self, seed=None):
    """ Reset the simulator and return an observation. """

    super().reset(seed=seed)

    # Reset sim
    mujoco.mj_resetData(self._model, self._data)

    # Reset all models
    self.bm_model.reset(self._model, self._data)
    self.perception.reset(self._model, self._data)
    info = self.task.reset(self._model, self._data)

    # Do a forward so everything will be set
    mujoco.mj_forward(self._model, self._data)

    if self._render_mode == "rgb_array_list":
      if self._render_stack_clean_at_reset:
        self._render_stack = []
        self._render_stack_perception = defaultdict(list)
      self._render_stack.append(self._GUI_rendering())
    elif self._render_mode == "human":
      self._GUI_rendering_pygame()

    return self.get_observation(), info

  def render(self):
    if self._render_mode == "rgb_array_list":
      render_stack = self._render_stack
      if self._render_stack_pop:
        self._render_stack = []
      return render_stack
    elif self._render_mode == "rgb_array":
      return self._GUI_rendering()
    else:
      return None
    
  def get_render_stack_perception(self):
      render_stack_perception = self._render_stack_perception
      # if self._render_stack_pop:
      #   self._render_stack_perception = defaultdict(list)
      return render_stack_perception

  def _GUI_rendering(self):
    # Grab an image from the 'for_testing' camera and grab all GUI-prepared images from included visual perception modules, and display them 'picture-in-picture'

    # Grab images
    img, _ = self._GUI_camera.render()

    if self._render_mode_perception == "embed":
      # Embed perception camera images into main camera image
        
      # perception_camera_images = [rgb_or_depth_array for camera in self.perception.cameras
      #                             for rgb_or_depth_array in camera.render() if rgb_or_depth_array is not None]
      perception_camera_images = self.perception.get_renders()

      # TODO: add text annotations to perception camera images
      if len(perception_camera_images) > 0:
        _img_size = img.shape[:2]  #(height, width)


        # Vertical alignment of perception camera images, from bottom right to top right
        ## TODO: allow for different inset locations
        _desired_subwindow_height = np.round(_img_size[0] / len(perception_camera_images)).astype(int)
        _maximum_subwindow_width = np.round(0.2 * _img_size[1]).astype(int)

        perception_camera_images_resampled = []
        for ocular_img in perception_camera_images:
          # Convert 2D depth arrays to 3D heatmap arrays
          if ocular_img.ndim == 2:
            if self._render_show_depths:
              ocular_img = matplotlib.pyplot.imshow(ocular_img, cmap=matplotlib.pyplot.cm.jet, interpolation='bicubic').make_image('TkAgg', unsampled=True)[0][
              ..., :3]
              matplotlib.pyplot.close()  #delete image
            else:
              continue

          resample_factor = min(_desired_subwindow_height / ocular_img.shape[0], _maximum_subwindow_width / ocular_img.shape[1])

          resample_height = np.round(ocular_img.shape[0] * resample_factor).astype(int)
          resample_width = np.round(ocular_img.shape[1] * resample_factor).astype(int)
          resampled_img = np.zeros((resample_height, resample_width, ocular_img.shape[2]), dtype=np.uint8)
          for channel in range(ocular_img.shape[2]):
            resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)

          perception_camera_images_resampled.append(resampled_img)

        ocular_img_bottom = _img_size[0]
        for ocular_img_idx, ocular_img in enumerate(perception_camera_images_resampled):
          #print(f"Modify ({ocular_img_bottom - ocular_img.shape[0]}, { _img_size[1] - ocular_img.shape[1]})-({ocular_img_bottom}, {img.shape[1]}).")
          img[ocular_img_bottom - ocular_img.shape[0]:ocular_img_bottom, _img_size[1] - ocular_img.shape[1]:] = ocular_img
          ocular_img_bottom -= ocular_img.shape[0]
        # input((len(perception_camera_images_resampled), perception_camera_images_resampled[0].shape, img.shape))
    elif self._render_mode_perception == "separate":
      for module, camera_list in self.perception.cameras_dict.items():
        for camera in camera_list:
          for rgb_or_depth_array in camera.render():
            if rgb_or_depth_array is not None:
              self._render_stack_perception[f"{module.modality}/{type(camera).__name__}"].append(rgb_or_depth_array)

    return img

  def _GUI_rendering_pygame(self):
    rgb_array = np.transpose(self._GUI_rendering(), axes=(1, 0, 2))

    if self._render_screen_size is None:
      self._render_screen_size = rgb_array.shape[:2]

    assert self._render_screen_size == rgb_array.shape[
                                       :2], f"Expected an rgb array of shape {self._render_screen_size} from self._GUI_camera, but received an rgb array of shape {rgb_array.shape[:2]}. "

    if self._render_window is None:
      pygame.init()
      pygame.display.init()
      self._render_window = pygame.display.set_mode(self._render_screen_size)

    if self._render_clock is None:
      self._render_clock = pygame.time.Clock()

    surf = pygame.surfarray.make_surface(rgb_array)
    self._render_window.blit(surf, (0, 0))
    pygame.event.pump()
    self._render_clock.tick(self.fps)
    pygame.display.flip()

  def close(self):
    """ Close the rendering window (if self._render_mode == 'human')."""
    super().close()
    if self._render_window is not None:
      import pygame

      pygame.display.quit()
      pygame.quit()

  @property
  def fps(self):
    return self._GUI_camera._fps

  def callback(self, callback_name, num_timesteps):
    """ Update a callback -- may be useful during training, e.g. for curriculum learning. """
    self.callbacks[callback_name].update(num_timesteps)

  def update_callbacks(self, num_timesteps):
    """ Update all callbacks. """
    for callback_name in self.callbacks:
      self.callback(callback_name, num_timesteps)

  # def get_logdict_keys(self):
  #   return list(self.task._info["log_dict"].keys())

  # def get_logdict_value(self, key):
  #   return self.task._info["log_dict"].get(key)

  @property
  def config(self):
    """ Return config. """
    return copy.deepcopy(self._config)

  @property
  def run_parameters(self):
    """ Return run parameters. """
    # Context cannot be deep copied
    exclude = {"rendering_context"}
    run_params = {k: copy.deepcopy(self._run_parameters[k]) for k in self._run_parameters.keys() - exclude}
    run_params["rendering_context"] = self._run_parameters["rendering_context"]
    return run_params

  @property
  def simulator_folder(self):
    """ Return simulator folder. """
    return self._simulator_folder

  @property
  def render_mode(self):
    """ Return render mode. """
    return self._render_mode

  def get_state(self):
    """ Return a state of the simulator / individual components (biomechanical model, perception model, task).

    This function is used for logging/evaluation purposes, not for RL training.

    Returns:
      A dict with one float or numpy vector per keyword.
    """

    # Get time, qpos, qvel, qacc, act_force, act, ctrl of the current simulation
    state = {"timestep": self._data.time,
             "qpos": self._data.qpos.copy(),
             "qvel": self._data.qvel.copy(),
             "qacc": self._data.qacc.copy(),
             "act_force": self._data.actuator_force.copy(),
             "act": self._data.act.copy(),
             "ctrl": self._data.ctrl.copy()}

    # Add state from the task
    state.update(self.task.get_state(self._model, self._data))

    # Add state from the biomechanical model
    state.update(self.bm_model.get_state(self._model, self._data))

    # Add state from the perception model
    state.update(self.perception.get_state(self._model, self._data))

    return state

  def close(self, **kwargs):
    """ Perform any necessary clean up.

    This function is inherited from gym.Env. It should be automatically called when this object is garbage collected
     or the program exists, but that doesn't seem to be the case. This function will be called if this object has been
     initialised in the context manager fashion (i.e. using the 'with' statement). """
    self.task.close(**kwargs)
    self.perception.close(**kwargs)
    self.bm_model.close(**kwargs)
>>>>>>> origin/main
