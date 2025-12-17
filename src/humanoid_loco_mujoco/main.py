import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf
from loco_mujoco.core import ObservationType
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
observation_spec = [
    # prioritized observations
    ObservationType.FreeJointPosNoXY(obs_name="free_joint", xml_name="root", group="prioritized", allow_randomization=False),
    ObservationType.FreeJointVel(obs_name="free_joint_vel", xml_name="root", group="prioritized", allow_randomization=False),
    # 修正左髋关节名称（之前的KeyError问题）
    ObservationType.JointPos(obs_name="joint_pos", xml_name="hip_flexion_l", group="prioritized", allow_randomization=False),
    # 修正右髋关节名称
    ObservationType.JointVel(obs_name="joint_vel1", xml_name="hip_flexion_r", group="prioritized", allow_randomization=False),
    # 修正左膝关节名称
    ObservationType.JointVel(obs_name="joint_vel2", xml_name="knee_angle_l", group="prioritized", allow_randomization=False),
    # 将head替换为有效的身体名称（例如torso_link）
    ObservationType.BodyPos(obs_name="head_pos", xml_name="torso_link", group="prioritized", allow_randomization=False),
    # policy observations --> these will be noisy
    ObservationType.ProjectedGravityVector(obs_name="proj_grav", xml_name="root", group="policy", allow_randomization=True),
    ObservationType.FreeJointVel(obs_name="free_joint_vel_pi", xml_name="root", group="policy", allow_randomization=True),
    # 修正左髋关节名称
    ObservationType.JointPos(obs_name="joint_pos_pi", xml_name="hip_flexion_l", group="policy", allow_randomization=True),
    # 修正右髋关节名称
    ObservationType.JointVel(obs_name="joint_vel1_pi", xml_name="hip_flexion_r", group="policy", allow_randomization=True),
    # 修正左膝关节名称
    ObservationType.JointVel(obs_name="joint_vel2_pi", xml_name="knee_angle_l", group="policy", allow_randomization=True),
    # 将head替换为有效的身体名称（例如torso_link）
    ObservationType.BodyPos(obs_name="head_pos_pi", xml_name="torso_link", group="policy", allow_randomization=True),
    ObservationType.LastAction(obs_name="last_action_pi", group="policy", allow_randomization=True)
]

# # example --> you can add as many datasets as you want in the lists!
env = ImitationFactory.make("UnitreeH1",
                            default_dataset_conf=DefaultDatasetConf(["squat", "walk"]),
                            terrain_type="RoughTerrain", 
                            terrain_params=dict(random_min_height=-0.1,random_max_height=0.1,random_downsampled_scale=0.5),
                            domain_randomization_type="DefaultRandomizer",
                            domain_randomization_params=randomization_config,
                            observation_spec=observation_spec,
                            control_type="PDControl", control_params=dict(p_gain=100, d_gain=1),
                            n_substeps=20)

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
