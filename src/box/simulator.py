import sys
import os
import gymnasium as gym
from gymnasium import spaces
import pygame
import mujoco
import numpy as np
import scipy
import matplotlib
import importlib
import shutil
import inspect
import pathlib
from datetime import datetime
import copy
from collections import defaultdict
import xml.etree.ElementTree as ET

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
            config, simulator_folder, {** run_parameters, "build": True}
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
        common_kwargs = {** run_parameters, **callbacks}
        
        # Initialize core components
        task = task_cls(model, data,**{** task_kwargs, **common_kwargs})
        bm_model = bm_cls(model, data, **{** bm_kwargs, **common_kwargs})
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
                dtype=np.float32, **state_params
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
        self.bm_model.close(** kwargs)

        # Clean up Pygame
        if self._render_window is not None:
            pygame.display.quit()
            pygame.quit()
            self._render_window = None
            self._render_clock = None

    # Callback methods
    def callback(self, callback_name, num_timesteps):
        """ Update a specific callback """
        self.callbacks[callback_name].update(num_timesteps)

    def update_callbacks(self, num_timesteps):
        """ Update all registered callbacks """
        for callback_name in self.callbacks:
            self.callback(callback_name, num_timesteps)

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