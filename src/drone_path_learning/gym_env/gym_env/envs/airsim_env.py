import gym
from gym import spaces
import airsim
from configparser import NoOptionError
import keyboard

import torch as th
import numpy as np
import math
import cv2
import logging

from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim


from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

logger = logging.getLogger(__name__)


def apply_environment_overrides(cfg, dynamic_model):
    """Apply optional curriculum-style environment overrides from config."""
    if not cfg.has_section("environment"):
        return

    if cfg.has_option("environment", "start_position"):
        import ast

        raw_position = cfg.get("environment", "start_position")
        start_position = ast.literal_eval(raw_position)
        if not isinstance(start_position, (list, tuple)) or len(start_position) != 3:
            raise ValueError(
                "environment.start_position must be a list like [x, y, z]"
            )
        dynamic_model.start_position = [float(value) for value in start_position]

    float_overrides = {
        "start_random_angle": "start_random_angle",
        "goal_distance": "goal_distance",
        "goal_random_angle": "goal_random_angle",
    }
    for option_name, attr_name in float_overrides.items():
        if cfg.has_option("environment", option_name):
            setattr(dynamic_model, attr_name, cfg.getfloat("environment", option_name))


class AirsimGymEnv(gym.Env, QtCore.QThread):
    
    action_signal = pyqtSignal(int, np.ndarray)
    state_signal = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    lgmd_signal = pyqtSignal(float, float, np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={"float": "{: 4.2f}".format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        logger.info("Initialize airsim-gym-env")
        self.model = None
        self.dynamic_model = None
        self.data_path = None
        self.lgmd = None

    def set_config(self, cfg):
        self.cfg = cfg
        self.env_name = cfg.get("options", "env_name")
        self.dynamic_name = cfg.get("options", "dynamic_name")
        self.keyboard_debug = cfg.getboolean("options", "keyboard_debug")
        self.generate_q_map = cfg.getboolean("options", "generate_q_map")
        self.perception_type = cfg.get("options", "perception")

        if self.perception_type == "lgmd":
            self.lgmd = LGMD(
                type="origin",
                p_threshold=50,
                s_threshold=0,
                Ki=2,
                i_layer_size=3,
                activate_coeff=1,
                use_on_off=True,
            )
            self.split_out_last = np.array([0, 0, 0, 0, 0])

        logger.info(
            "Environment=%s Dynamics=%s Perception=%s",
            self.env_name,
            self.dynamic_name,
            self.perception_type,
        )

        if self.dynamic_name != "Multirotor":
            raise ValueError(
                f"Only 'Multirotor' is supported after cleanup, got: {self.dynamic_name}"
            )
        self.dynamic_model = MultirotorDynamicsAirsim(cfg)

        if self.env_name == "NH_center":
            start_position = [0, 0, 5]
            goal_distance = 30
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi * 2
            )
            self.work_space_x = [-10000, 10000]
            self.work_space_y = [-10000, 10000]
            self.work_space_z = [0.5, 500]
            self.max_episode_steps = 1000
        elif self.env_name == "NH_tree":
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance=90, random_angle=0)
            self.work_space_x = [
                start_position[0],
                start_position[0] + goal_distance + 10,
            ]
            self.work_space_y = [start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
            self.max_episode_steps = 400
        elif self.env_name == "City":
            start_position = [40, -30, 40]
            goal_position = [280, -200, 40]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 350]
            self.work_space_y = [-300, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 400
        elif self.env_name == "City_400":
            start_position = [0, 0, 50]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-220, 220]
            self.work_space_y = [-220, 220]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 800
        elif self.env_name == "Tree_200":
            start_position = [0, 0, 8]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 600
        elif self.env_name == "SimpleAvoid":
            start_position = [0, 0, 5]
            goal_distance = 50
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi * 2
            )
            self.work_space_x = [
                start_position[0] - goal_distance - 10,
                start_position[0] + goal_distance + 10,
            ]
            self.work_space_y = [
                start_position[1] - goal_distance - 10,
                start_position[1] + goal_distance + 10,
            ]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 400
        elif self.env_name == "Forest":
            start_position = [0, 0, 10]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 300
        elif self.env_name == "Trees":
            start_position = [0, 0, 5]
            goal_distance = 70
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi * 2
            )
            self.work_space_x = [
                start_position[0] - goal_distance - 10,
                start_position[0] + goal_distance + 10,
            ]
            self.work_space_y = [
                start_position[1] - goal_distance - 10,
                start_position[1] + goal_distance + 10,
            ]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 500
        else:
            raise ValueError(f"Invalid env_name: {self.env_name}")

        apply_environment_overrides(cfg, self.dynamic_model)

        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint("options", "cnn_feature_num")

        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        self.crash_distance = cfg.getfloat("environment", "crash_distance")
        self.accept_radius = cfg.getint("environment", "accept_radius")
        self.success_check_mode = "planar_with_altitude"
        self.success_altitude_tolerance = 5.0
        if cfg.has_option("environment", "success_check_mode"):
            self.success_check_mode = cfg.get("environment", "success_check_mode").strip().lower()
        if cfg.has_option("environment", "success_altitude_tolerance"):
            self.success_altitude_tolerance = cfg.getfloat(
                "environment", "success_altitude_tolerance"
            )
        self.success_altitude_tolerance = max(self.success_altitude_tolerance, 0.0)
        self.depth_collision_percentile = 5.0
        self.depth_collision_roi_top_ratio = 0.65
        if cfg.has_option("environment", "depth_collision_percentile"):
            self.depth_collision_percentile = cfg.getfloat(
                "environment", "depth_collision_percentile"
            )
        if cfg.has_option("environment", "depth_collision_roi_top_ratio"):
            self.depth_collision_roi_top_ratio = cfg.getfloat(
                "environment", "depth_collision_roi_top_ratio"
            )
        self.depth_collision_percentile = float(
            np.clip(self.depth_collision_percentile, 0.0, 100.0)
        )
        self.depth_collision_roi_top_ratio = float(
            np.clip(self.depth_collision_roi_top_ratio, 0.1, 1.0)
        )

        self.max_depth_meters = cfg.getint("environment", "max_depth_meters")
        self.screen_height = cfg.getint("environment", "screen_height")
        self.screen_width = cfg.getint("environment", "screen_width")

        self.trajectory_list = []

        if self.perception_type == "vector" or self.perception_type == "lgmd":
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(1, self.cnn_feature_length + self.state_feature_length),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.screen_height, self.screen_width, 2),
                dtype=np.uint8,
            )

        self.action_space = self.dynamic_model.action_space

        self.reward_type = None
        try:
            # Prefer [environment] to match current config schema, keep [options] for backward compatibility.
            if cfg.has_option("environment", "reward_type"):
                self.reward_type = cfg.get("environment", "reward_type")
            else:
                self.reward_type = cfg.get("options", "reward_type")
            logger.info("Reward type: %s", self.reward_type)
        except NoOptionError:
            self.reward_type = None

        # Allow explicit override from config while preserving per-env defaults.
        if cfg.has_option("environment", "max_episode_steps"):
            self.max_episode_steps = cfg.getint("environment", "max_episode_steps")

    def reset(self):
        if self.dynamic_model is None:
            raise RuntimeError(
                "AirsimGymEnv is not configured. Call env.set_config(cfg) before env.reset()."
            )

        self.dynamic_model.reset()

        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance

        self.trajectory_list = []

        obs = self.get_obs()

        return obs

    def step(self, action):
        if self.dynamic_model is None:
            raise RuntimeError(
                "AirsimGymEnv is not configured. Call env.set_config(cfg) before env.step()."
            )

        self.dynamic_model.set_action(action)

        position_ue4 = self.dynamic_model.get_position()
        self.trajectory_list.append(position_ue4)

        obs = self.get_obs()

        is_success = self.is_in_desired_pose()
        is_crash = self.is_crashed()
        is_not_in_workspace = self.is_not_inside_workspace()
        is_max_steps = self.step_num >= self.max_episode_steps

        done = is_success or is_crash or is_not_in_workspace or is_max_steps

        done_reason = "unknown"
        if is_success:
            done_reason = "success"
        elif is_not_in_workspace:
            done_reason = "out_of_workspace"
        elif is_crash:
            done_reason = "collision"
        elif is_max_steps:
            done_reason = "max_steps"

        info = {
            "is_success": is_success,
            "is_crash": is_crash,
            "is_not_in_workspace": is_not_in_workspace,
            "is_max_steps": is_max_steps,
            "done_reason": done_reason,
            "step_num": self.step_num,
        }
        if done:
            if is_crash:
                collision_info = self.client.simGetCollisionInfo()
                logger.info(
                    "Episode done | reason: COLLISION | physics_collision=%s depth_distance=%.2f crash_threshold=%.2f | "
                    "position=[%.2f, %.2f, %.2f] | step=%d",
                    collision_info.has_collided,
                    self.min_distance_to_obstacles,
                    self.crash_distance,
                    position_ue4[0],
                    position_ue4[1],
                    position_ue4[2],
                    self.step_num,
                )
            else:
                logger.info(
                    "Episode done | reason: %s | success=%s crash=%s workspace=%s max_steps=%s | "
                    "distance_3d=%.2f distance_2d=%.2f z_err=%.2f | step=%d",
                    done_reason,
                    is_success,
                    is_crash,
                    is_not_in_workspace,
                    is_max_steps,
                    self.get_distance_to_goal_3d(),
                    self.dynamic_model.get_distance_to_goal_2d(),
                    abs(
                        self.dynamic_model.get_position()[2]
                        - self.dynamic_model.goal_position[2]
                    ),
                    self.step_num,
                )

        if self.reward_type in ("reward_distance_based", "reward_default"):
            reward = self.compute_reward(done, action)
        elif self.reward_type == "reward_with_action":
            reward = self.compute_reward_with_action(done, action)
        elif self.reward_type == "reward_new":
            reward = self.compute_reward_multirotor_new(done, action)
        elif self.reward_type == "reward_lqr":
            reward = self.compute_reward_lqr(done, action)
        elif self.reward_type == "reward_final":
            reward = self.compute_reward_final(done, action)
        else:
            reward = self.compute_reward(done, action)

        self.cumulated_episode_reward += reward

        self.print_train_info_airsim(action, obs, reward, info)

        self.set_pyqt_signal_multirotor(action, reward)

        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={"float": "{: 0.3f}".format})
            print(
                "============================================================================="
            )
            print(
                "episode",
                self.episode_num,
                "step",
                self.step_num,
                "total step",
                self.total_step,
            )
            print("action", action_copy)
            print("state", state_copy)
            print("state_norm", self.dynamic_model.state_norm)
            print("reward {:.3f} {:.3f}".format(reward, self.cumulated_episode_reward))
            print("done", done)
            keyboard.wait("a")

        if self.generate_q_map and (
            self.cfg.get("options", "algo") == "TD3"
            or self.cfg.get("options", "algo") == "SAC"
        ):
            if self.model is not None:
                with th.no_grad():
                    obs_copy = obs.copy()
                    if self.perception_type != "vector":
                        obs_copy = obs_copy.swapaxes(0, 1)
                        obs_copy = obs_copy.swapaxes(0, 2)
                    device = getattr(self.model, "device", th.device("cpu"))
                    obs_tensor = (
                        th.from_numpy(obs_copy[tuple([None])]).float().to(device)
                    )
                    action_tensor = th.from_numpy(action[None]).float().to(device)
                    q_value_current = self.model.critic(obs_tensor, action_tensor)
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]

                    q_value = min(q_1, q_2)[0]

                    self.visual_log_q_value(q_value, action, reward)

        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    def get_obs(self):
        if self.perception_type == "vector":
            obs = self.get_obs_vector()
        elif self.perception_type == "lgmd":
            obs = self.get_obs_lgmd()
        else:
            obs = self.get_obs_image()

        return obs

    def _update_min_distance_to_obstacles(self, depth_image):
        if depth_image is None or depth_image.size == 0:
            self.min_distance_to_obstacles = 100
            return

        roi_height = max(
            1, int(depth_image.shape[0] * self.depth_collision_roi_top_ratio)
        )
        depth_roi = depth_image[:roi_height, :]
        depth_valid = depth_roi[(depth_roi > 0.1) & (depth_roi < 1000)]

        if len(depth_valid) > 1:
            percentile_depth = np.percentile(
                depth_valid, self.depth_collision_percentile
            )
            self.min_distance_to_obstacles = max(percentile_depth, 0.1)
        else:
            self.min_distance_to_obstacles = 100

    def get_obs_image(self):
        image = self.get_depth_image()  # 0-6550400.0 float 32
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))

        self._update_min_distance_to_obstacles(image)

        image_scaled = (
            np.clip(image_resize, 0, self.max_depth_meters)
            / self.max_depth_meters
            * 255
        )
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0 : self.state_feature_length] = state_feature

        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

    def get_depth_gray_image(self):
        responses = self.client.simGetImages(
            [
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ]
        )

        while responses[0].width == 0:
            logger.warning("Depth/Scene image acquisition failed, retrying...")
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                ]
            )

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width, responses[0].height
        )
        depth_meter = depth_img * 100

        img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[1].height, responses[1].width, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        return depth_meter, img_gray

    def get_depth_image(self):

        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
        )

        while responses[0].width == 0:
            logger.warning("Depth image acquisition failed, retrying...")
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
            )

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width, responses[0].height
        )

        depth_meter = depth_img * 100

        return depth_meter

    def get_obs_vector(self):

        image = self.get_depth_image()
        self._update_min_distance_to_obstacles(image)

        image_scaled = (
            np.clip(image, 0, self.max_depth_meters) / self.max_depth_meters * 255
        )
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        image_obs = image_uint8
        split_row = 1
        split_col = 5

        v_split_list = np.vsplit(image_obs, split_row)

        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())

        img_feature = np.array(split_final) / 255.0

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    def get_obs_lgmd(self):
        depth_meter, img_gray = self.get_depth_gray_image()
        self._update_min_distance_to_obstacles(depth_meter)

        self.lgmd.update(img_gray)

        split_col_num = 5
        s_layer = self.lgmd.s_layer
        s_layer_split = np.hsplit(s_layer, split_col_num)

        lgmd_out_list = []
        activate_coeff = 0.5
        for i in range(split_col_num):
            s_layer_activated_sum = abs(np.sum(s_layer_split[i]))
            Kf = -(s_layer_activated_sum * activate_coeff) / (192 * 64)  # 0 - 1
            a = np.exp(Kf)
            lgmd_out_norm = (1 / (1 + a) - 0.5) * 2
            lgmd_out_list.append(lgmd_out_norm)

        heatmapshow = None
        heatmapshow = cv2.normalize(
            s_layer,
            heatmapshow,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow("gray image", img_gray)
        cv2.imshow("depth image", np.clip(depth_meter, 0, 255) / 255)
        cv2.imshow("s-layer", heatmapshow)
        cv2.waitKey(1)

        split_final = np.array(lgmd_out_list)

        filter_coeff = 0.8
        split_final_filter = (
            filter_coeff * split_final + (1 - filter_coeff) * self.split_out_last
        )
        self.split_out_last = split_final_filter

        img_feature = np.array(split_final_filter)

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all


    def compute_reward(self, done, action):
        reward = 0.0
        reward_reach = 80.0
        reward_crash = -80.0
        reward_outside = -30.0
        reward_timeout = -10.0

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            delta_distance = self.previous_distance_from_des_point - distance_now
            goal_distance_base = max(self.dynamic_model.goal_distance, 1e-6)
            reward_distance = float(np.clip(delta_distance / goal_distance_base * 300.0, -2.0, 2.0))
            self.previous_distance_from_des_point = distance_now

            action_cost = 0.0
            yaw_speed_cost = 0.05 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            action_cost += yaw_speed_cost

            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.05 * ((abs(action[1]) / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = 0.03 * (
                    (
                        abs(self.dynamic_model.state_raw[1])
                        / self.dynamic_model.max_vertical_difference
                    )
                    ** 2
                )
                action_cost += v_z_cost + z_err_cost

            yaw_error = abs(self.dynamic_model.state_raw[2])
            yaw_error_cost = 0.05 * min(yaw_error / 90.0, 1.0)

            obs_cost = 0.0
            obs_punish_dist = self.crash_distance + 2.0
            if self.min_distance_to_obstacles < obs_punish_dist:
                obs_cost = 0.8 * np.clip(
                    (obs_punish_dist - self.min_distance_to_obstacles)
                    / max(obs_punish_dist - self.crash_distance, 1e-6),
                    0.0,
                    1.0,
                )

            direction_bonus = 0.2 * np.tanh(delta_distance * 2.0)
            reward = reward_distance + direction_bonus - action_cost - yaw_error_cost - obs_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            elif self.is_crashed():
                reward = reward_crash
            elif self.is_not_inside_workspace():
                reward = reward_outside
            elif self.step_num >= self.max_episode_steps:
                reward = reward_timeout

        return float(reward)

    def compute_reward_final(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if self.env_name == "NH_center":
            distance_reward_coef = 500
        else:
            distance_reward_coef = 50

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (
                distance_reward_coef
                * (self.previous_distance_from_des_point - distance_now)
                / self.dynamic_model.goal_distance
            )
            self.previous_distance_from_des_point = distance_now
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g) / 5, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip(
                    (self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1
                )
            else:
                punishment_obs = 0

            punishment_action = 0

            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                v_z_cost = (abs(action[1]) / self.dynamic_model.v_z_max) ** 2
                z_err_cost = (
                    abs(self.dynamic_model.state_raw[1])
                    / self.dynamic_model.max_vertical_difference
                ) ** 2
                punishment_action += v_z_cost + z_err_cost

            punishment_action += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = (
                reward_distance
                - 0.1 * punishment_pose
                - 0.2 * punishment_obs
                - 0.1 * punishment_action
                - 0.5 * yaw_error_cost
            )
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_multirotor_new(self, done, action):
        reward = 0
        reward_reach = 100
        reward_crash = -100
        reward_outside = 0

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (
                (self.previous_distance_from_des_point - distance_now)
                / self.dynamic_model.goal_distance
                * 5
            )
            self.previous_distance_from_des_point = distance_now

            state_cost = 0
            action_cost = 0
            obs_cost = 0

            yaw_error_deg = self.dynamic_model.state_raw[1]

            relative_yaw_cost = abs(yaw_error_deg / 180)
            action_cost = abs(action[1]) / self.dynamic_model.yaw_rate_max_rad

            obs_punish_dist = 5
            if self.min_distance_to_obstacles < obs_punish_dist:
                obs_cost = 1 - (
                    self.min_distance_to_obstacles - self.crash_distance
                ) / (obs_punish_dist - self.crash_distance)
                obs_cost = 0.5 * obs_cost**2
            reward = -(2 * relative_yaw_cost + 0.5 * action_cost)
        else:
            if self.is_in_desired_pose():
                reward = reward_reach * (1 - abs(self.dynamic_model.state_norm[1]))
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_with_action(self, done, action):
        reward = 0
        reward_reach = 50
        reward_crash = -50
        reward_outside = -10

        step_cost = 0.01

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (
                (self.previous_distance_from_des_point - distance_now)
                / self.dynamic_model.goal_distance
                * 10
            )
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            v_xy_cost = 0.02 * abs(action[0] - 5) / 4
            yaw_rate_cost = 0.02 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.02 * abs(action[1]) / self.dynamic_model.v_z_max
                action_cost += v_z_cost
            action_cost += v_xy_cost + yaw_rate_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.05 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_lqr(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = 0

        if not done:
            action_cost = 0
            yaw_speed_cost = 0.2 * (
                (action[-1] / self.dynamic_model.yaw_rate_max_rad) ** 2
            )

            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.1 * ((action[1] / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = 0.1 * (
                    (
                        self.dynamic_model.state_raw[1]
                        / self.dynamic_model.max_vertical_difference
                    )
                    ** 2
                )
                action_cost += v_z_cost + z_err_cost

            action_cost += yaw_speed_cost

            yaw_error_clip = min(max(-60, self.dynamic_model.state_raw[2]), 60) / 60
            yaw_error_cost = 1.0 * (yaw_error_clip**2)

            reward = -(action_cost + yaw_error_cost)

        else:
            if self.is_in_desired_pose():
                yaw_error_clip = min(max(-30, self.dynamic_model.state_raw[2]), 30) / 30
                reward = reward_reach * (1 - yaw_error_clip**2)
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def is_not_inside_workspace(self):
        is_not_inside = False
        current_position = self.dynamic_model.get_position()

        if (
            current_position[0] < self.work_space_x[0]
            or current_position[0] > self.work_space_x[1]
            or current_position[1] < self.work_space_y[0]
            or current_position[1] > self.work_space_y[1]
            or current_position[2] < self.work_space_z[0]
            or current_position[2] > self.work_space_z[1]
        ):
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        distance_2d = self.dynamic_model.get_distance_to_goal_2d()
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        z_error = abs(current_pose[2] - goal_pose[2])

        if self.success_check_mode == "3d":
            return self.get_distance_to_goal_3d() < self.accept_radius

        if self.success_check_mode == "2d":
            return distance_2d < self.accept_radius

        # default: planar success + altitude tolerance
        return (
            distance_2d < self.accept_radius
            and z_error < self.success_altitude_tolerance
        )

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()

        physics_collision = collision_info.has_collided

        depth_collision = self.min_distance_to_obstacles < self.crash_distance

        is_crashed = physics_collision or depth_collision

        return is_crashed

    def get_distance_to_goal_3d(self):
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(
            pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2)
        )

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))

        return dis

    def print_train_info_airsim(self, action, obs, reward, info):
        msg_train_info = "EP: {} Step: {} Total_step: {}".format(
            self.episode_num, self.step_num, self.total_step
        )

        self.client.simPrintLogMessage("Train: ", msg_train_info)
        self.client.simPrintLogMessage("Action: ", str(action))
        self.client.simPrintLogMessage(
            "reward: ",
            "{:4.4f} total: {:4.4f}".format(reward, self.cumulated_episode_reward),
        )
        self.client.simPrintLogMessage("Info: ", str(info))
        self.client.simPrintLogMessage(
            "Feature_norm: ", str(self.dynamic_model.state_norm)
        )
        self.client.simPrintLogMessage(
            "Feature_raw: ", str(self.dynamic_model.state_raw)
        )
        self.client.simPrintLogMessage(
            "Min_depth: ", str(self.min_distance_to_obstacles)
        )

    def set_pyqt_signal_multirotor(self, action, reward):
        step = int(self.total_step)

        state = self.dynamic_model.state_raw
        if self.dynamic_model.navigation_3d:
            action_output = np.asarray(action, dtype=np.float32)
            state_output = np.asarray(state, dtype=np.float32)
        else:
            action_output = np.asarray([action[0], 0, action[1]], dtype=np.float32)
            state_output = np.asarray(
                [state[0], 0, state[2], state[3], 0, state[5]], dtype=np.float32
            )

        self.action_signal.emit(step, action_output)
        self.state_signal.emit(step, state_output)

        self.attitude_signal.emit(
            step,
            np.asarray(self.dynamic_model.get_attitude()),
            np.asarray(self.dynamic_model.get_attitude_cmd()),
        )
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(
            np.asarray(self.dynamic_model.goal_position),
            np.asarray(self.dynamic_model.start_position),
            np.asarray(self.dynamic_model.get_position()),
            np.asarray(self.trajectory_list),
        )

    def visual_log_q_value(self, q_value, action, reward):
        
        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, "q_value_map"):
            self.q_value_map = np.full((9, map_size_x + 1, map_size_y + 1), np.nan)

        position = self.dynamic_model.get_position()
        pose_x = position[0]
        pose_y = position[1]

        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])

        if index_x in range(0, map_size_x) and index_y in range(0, map_size_y):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[-1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[-1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            logger.warning(
                "X/Y index is outside map range in visual_log_q_value: x=%s y=%s",
                index_x,
                index_y,
            )

        record_step = self.cfg.getint("options", "q_map_save_steps")
        if (self.total_step + 1) % record_step == 0:
            if self.data_path is not None:
                np.save(
                    self.data_path + "/q_value_map_{}".format(self.total_step + 1),
                    self.q_value_map,
                )
                self.q_value_map[5, :, :] = np.nan
                self.q_value_map[6, :, :] = np.nan
                self.q_value_map[7, :, :] = np.nan
                self.q_value_map[8, :, :] = np.nan

    def close(self):
        """Gym close hook: release simulator control cleanly."""
        if self.dynamic_model is not None and hasattr(self.dynamic_model, "close"):
            self.dynamic_model.close()

