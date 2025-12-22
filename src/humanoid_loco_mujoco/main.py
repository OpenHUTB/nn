from typing import List, Any, Union, Tuple, Dict
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import asdict

from mujoco import MjSpec, MjModel, MjData
from mujoco.mjx import Model, Data

import numpy as np
from loco_mujoco.task_factories import ImitationFactory,  DefaultDatasetConf
from loco_mujoco.environments import UnitreeG1
from loco_mujoco.core.initial_state_handler import InitialStateHandler
from loco_mujoco.core.observations import Observation, StatefulObservation
from loco_mujoco.core.control_functions.pd import PDControl, PDControlState
from loco_mujoco.core.terminal_state_handler import TerminalStateHandler

# ----- CUSTOM ENVIRONMENT -----
# custom environment class with fixed waist yaw joint
class CustomUnitreeG1(UnitreeG1):
    def __init__(self, spec=None, **kwargs):
        self._waist_yaw_joint_name = "waist_yaw_joint"
        self._waist_yaw_actuator_name = "waist_yaw"
        super().__init__(spec=spec or self.get_default_xml_file_path(), **kwargs)

    @staticmethod
    def fix_waist_yaw(spec: MjSpec):
        spec.joints = [j for j in spec.joints if j.name != "waist_yaw_joint"]

    def _get_observation_specification(self, spec: MjSpec) -> List[Observation]:
        obs = super()._get_observation_specification(spec)
        return [o for o in obs if o.xml_name != self._waist_yaw_joint_name]

    def _get_action_specification(self, spec: MjSpec) -> List[str]:
        actions = super()._get_action_specification(spec)
        return [a for a in actions if a != self._waist_yaw_actuator_name]

# ----- CUSTOM CONTROL FUNCTION -----
@struct.dataclass
class CustomControlFunctionState(PDControlState):
    moving_average: Union[np.ndarray, jnp.ndarray]

class CustomControlFunction(PDControl):
    def generate_action(self, env, action, model, data, carry, backend):
        orig_action, carry = super().generate_action(env, action, model, data, carry, backend)
        ma = 0.99 * carry.control_func_state.moving_average + 0.01 * orig_action
        state = carry.control_func_state.replace(moving_average=ma)
        carry = carry.replace(control_func_state=state)
        return ma, carry

    def init_state(self, env, key, model, data, backend):
        orig_state = super().init_state(env, key, model, data, backend)
        dim = env.info.action_space.shape[0]
        ma = backend.zeros_like(dim)
        return CustomControlFunctionState(moving_average=ma, **asdict(orig_state))

# ----- CUSTOM INITIAL STATE HANDLER -----
# custom initial state handler to set the initial height of the robot between 2.0 and 2.5
class CustomInitialStateHandler(InitialStateHandler):
    def reset(self, env: Any, model: Union[MjModel, Model],
              data: Union[MjData, Data], carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        if backend == np:
            data.qpos[2] = np.random.uniform(2.0, 2.5)
        else:
            key, subkey = jax.random.split(carry.key)
            z = jax.random.uniform(subkey, (1,), minval=2.0, maxval=2.5)
            data = data.replace(qpos=data.qpos.at[2].set(z))
            carry = carry.replace(key=key)
        return data, carry

# ----- CUSTOM TERMINAL STATE HANDLER -----
# custom terminal state handler to terminate the episode with a probability of 0.05
class CustomTerminalStateHandler(TerminalStateHandler):
    def reset(self, env, model, data, carry, backend):
        return data, carry

    def is_absorbing(self, env, obs, info, data, carry):
        return np.random.uniform() < 0.05, carry

    def mjx_is_absorbing(self, env, obs, info, data, carry):
        key, subkey = jax.random.split(carry.key)
        absorbing = jax.random.uniform(subkey) < 0.05
        return absorbing, carry.replace(key=key)



# ----- CUSTOM OBSERVATIONS -----
# custom observation class to observe the center of mass position of the pelvis
class CustomBodyCOMPos(Observation):
    dim = 3

    def __init__(self, name: str, xml_name: str):
        self.xml_name = xml_name
        super().__init__(name)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "xipos"


# custom observation class to observe the moving average of the center of mass position of the pelvis
@struct.dataclass
class CustomBodyCOMPosMovingAverageState:
    moving_average: Union[np.ndarray, jnp.ndarray]

class CustomBodyCOMPosMovingAverage(StatefulObservation):
    dim = 3

    def __init__(self, name: str, xml_name: str):
        self.xml_name = xml_name
        super().__init__(name)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend):
        return CustomBodyCOMPosMovingAverageState(moving_average=backend.zeros(self.dim))

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        obs_states = carry.observation_states
        obs_state = getattr(obs_states, self.name)
        xipos = backend.squeeze(data.xipos[self.data_type_ind])
        ma = 0.9 * obs_state.moving_average + 0.1 * xipos
        obs_states = obs_states.replace(**{self.name: obs_state.replace(moving_average=ma)})
        carry = carry.replace(observation_states=obs_states)
        return backend.ravel(ma), carry


# ----- REGISTRATION -----
CustomUnitreeG1.register()
CustomControlFunction.register()
CustomBodyCOMPos.register()
CustomBodyCOMPosMovingAverage.register()
CustomTerminalStateHandler.register()
CustomInitialStateHandler.register()

# ----- ENVIRONMENT SETUP -----
observation_spec = [
    CustomBodyCOMPos("pelvis_com", "pelvis"),
    CustomBodyCOMPosMovingAverage("pelvis_com_mov_avg", "pelvis"),
]


randomization_config = {
    # gravity
    "randomize_gravity": True,
    "gravity_range": [9.51, 10.11],

    # geom properties
    "randomize_geom_friction_tangential": True,
    "geom_friction_tangential_range": [0.8, 1.2],
    "randomize_geom_friction_torsional": True,
    "geom_friction_torsional_range": [0.003, 0.007],
    "randomize_geom_friction_rolling": True,
    "geom_friction_rolling_range": [0.00008, 0.00012],
    "randomize_geom_damping": True,
    "geom_damping_range": [72, 88],
    "randomize_geom_stiffness": True,
    "geom_stiffness_range": [900, 1100],

    # joint properties
    "randomize_joint_damping": True,
    "joint_damping_range": [0.3, 1.5],
    "randomize_joint_stiffness": True,
    "joint_stiffness_range": [0.9, 1.1],
    "randomize_joint_friction_loss": True,
    "joint_friction_loss_range": [0.0, 0.2],
    "randomize_joint_armature": True,
    "joint_armature_range": [0.08, 0.12],

    # base mass
    "randomize_base_mass": True,
    "base_mass_to_add_range": [-2.0, 2.0],

    # COM
    "randomize_com_displacement": True,
    "com_displacement_range": [-0.15, 0.15],

    # link mass
    "randomize_link_mass": True,
    "link_mass_multiplier_range": {
        "root_body": [0.5, 1.9],
        "other_bodies": [0.8, 1.2],
    },

    # PD Gains (if PDControl is used)
    "add_p_gains_noise": True,
    "add_d_gains_noise": True,
    "p_gains_noise_scale": 0.1,
    "d_gains_noise_scale": 0.1,

    # Observation Noise
    "add_joint_pos_noise": True,
    "joint_pos_noise_scale": 0.003,
    "add_joint_vel_noise": True,
    "joint_vel_noise_scale": 0.08,
    "add_gravity_noise": True,
    "gravity_noise_scale": 0.015,
    "add_free_joint_lin_vel_noise": True,
    "lin_vel_noise_scale": 0.1,
    "add_free_joint_ang_vel_noise": True,
    "ang_vel_noise_scale": 0.02,
}



# # example --> you can add as many datasets as you want in the lists!"squat",
env = ImitationFactory.make("UnitreeG1",
                            default_dataset_conf=DefaultDatasetConf([ "walk"]),
                            terrain_type="RoughTerrain", 
                            terrain_params=dict(random_min_height=-0.1,random_max_height=0.1,random_downsampled_scale=0.5),
                            domain_randomization_type="DefaultRandomizer",
                            domain_randomization_params=randomization_config,
                            observation_spec=observation_spec,
                            control_type="CustomControlFunction", 
                            control_params={"p_gain": 100, "d_gain": 1},
                            terminal_state_type="CustomTerminalStateHandler",
                            init_state_type="CustomInitialStateHandler",
                            n_substeps=20)

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)