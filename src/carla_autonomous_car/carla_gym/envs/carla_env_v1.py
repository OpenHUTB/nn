"""
2025.11.30 14:11- 增强版
优化版本 - 解决碰撞和换道问题
主要改进:
1. ✅ 增强安全机制（更严格的碰撞预防）
2. ✅ 优化换道激励（鼓励合理换道）
3. ✅ 重新平衡奖励（减少负值，增加正向激励）
4. ✅ 改进IDM和控制逻辑
5. ⭐ 换道过程塑形奖励（密集信号）
6. ⭐ 详细的换道统计追踪
"""


import gym
import time
import math
import itertools
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional  # ⭐ 添加这一行
from tools.modules import *
from config import cfg
from agents.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel

import sys
import os

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN']


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def constant_velocity_predict(distance, ego_speed, ahead_speed):
    """TTC估计"""
    rel_v = ego_speed - ahead_speed
    if rel_v <= 0:
        return float('inf')
    if distance <= 0:
        return 0.0
    return distance / rel_v


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0], ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    min_dist = 300
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx

    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)

        # 修复：分两步判断，避免过长的一行
        if temp_dist <= min_dist:
            body_frame_x = inertial_to_body_frame(
                ego_location,
                temp_wp[0],
                temp_wp[1],
                ego_state[2]
            )[0]

            if body_frame_x > 0.0:
                closest_wp_index = i
                min_dist = temp_dist

    return f_idx + closest_wp_index


# ============================================================
# ⭐ 新增：Baseline奖励系统（复现代码1的逻辑）
# ============================================================
class BaselineRewardSystem:
    """
    Baseline奖励系统 - 复现代码1的原始奖励逻辑

    特点：
    1. 简单的速度指数奖励
    2. 基于速度变化的换道奖励/惩罚
    3. 固定的碰撞和off-road惩罚
    4. 无课程学习、无milestone、无复杂塑形
    """

    def __init__(self, config):
        # 基础参数
        self.targetSpeed = config.get('target_speed', 30.0)
        self.maxSpeed = config.get('max_speed', 40.0)
        self.lane_width = config.get('lane_width', 3.5)
        self.track_length = config.get('track_length', 1000.0)

        # RL参数（来自代码1）
        self.w_speed = config.get('w_speed', 10)  # 默认值从cfg.RL.W_SPEED
        self.w_r_speed = config.get('w_r_speed', 10)  # 默认值从cfg.RL.W_R_SPEED

        self.min_speed_gain = config.get('min_speed_gain', 0.05)
        self.min_speed_loss = config.get('min_speed_loss', -0.05)
        self.lane_change_reward = config.get('lane_change_reward', 1.2)
        self.lane_change_penalty = config.get('lane_change_penalty', 0.3)

        # 惩罚参数（固定值）
        self.collision_penalty = config.get('collision_penalty', -100)
        self.off_the_road_penalty = config.get('off_the_road_penalty', -50)

        # 历史状态
        self.last_speed = 0.0
        self.init_speed = 0.0  # 路径开始时的速度

        # ⭐ 添加这段代码
        self.weights = {
            'speed': 1.0,
            'safety': 0.0,
            'lane_keeping': 0.0,
            'lane_change': 1.0,
            'comfort': 0.0,
            'efficiency': 0.0,
            'progress': 0.0,
            'lane_change_shaping': 0.0,
        }

        # 统计
        self.episode_stats = {
            'total_reward': 0.0,
            'speed_violations': 0,
            'successful_lane_changes': 0,
            'attempted_lane_changes': 0,
        }

        self.training_step = 0

        print("✅ BaselineRewardSystem initialized (Original reward from Code1)")

    def reset(self):
        """重置episode"""
        self.last_speed = 0.0
        self.init_speed = 0.0

        self.episode_stats = {
            'total_reward': 0.0,
            'speed_violations': 0,
            'successful_lane_changes': 0,
            'attempted_lane_changes': 0,
        }

    def compute_total_reward(self, state_info):
        """
        计算总奖励 - 复现代码1的逻辑

        Args:
            state_info: 包含当前状态的字典

        Returns:
            reward: 标量奖励值
            reward_details: 详细信息字典
        """
        self.training_step += 1

        # 提取状态信息
        current_speed = state_info['current_speed']
        init_speed = state_info.get('init_speed', self.init_speed)
        is_lane_changing = state_info.get('is_lane_changing', False)
        collision = state_info.get('collision', False)
        off_road = state_info.get('off_road', False)

        # 保存init_speed用于下一步
        if init_speed != 0:
            self.init_speed = init_speed

        # ============================================================
        # 1. 碰撞惩罚
        # ============================================================
        if collision:
            components = {
                'speed': 0.0,
                'lane_change': 0.0,
            }

            return self.collision_penalty, {
                'total': self.collision_penalty,
                'base': self.collision_penalty,
                'components': components,
                'type': 'collision',
                'training_step': self.training_step,
                'collision_penalty': self.collision_penalty,
                'off_road_penalty': self.off_the_road_penalty,
                'episode_stats': self.episode_stats.copy(),
            }

        # ============================================================
        # 2. Off-road惩罚
        # ============================================================
        if off_road:
            components = {
                'speed': 0.0,
                'lane_change': 0.0,
            }

            return self.off_the_road_penalty, {
                'total': self.off_the_road_penalty,
                'base': self.off_the_road_penalty,
                'components': components,
                'type': 'off_road',
                'training_step': self.training_step,
                'collision_penalty': self.collision_penalty,
                'off_road_penalty': self.off_the_road_penalty,
                'episode_stats': self.episode_stats.copy(),
            }

        # ============================================================
        # 3. 速度奖励（指数形式）
        # ============================================================
        e_speed = abs(self.targetSpeed - current_speed)
        r_speed = self.w_r_speed * np.exp(-e_speed ** 2 / self.maxSpeed * self.w_speed)

        # 统计速度违规
        if current_speed > self.maxSpeed * 1.1:
            self.episode_stats['speed_violations'] += 1

        # ============================================================
        # 4. 换道奖励/惩罚
        # ============================================================
        r_lane_change = 0.0

        if is_lane_changing:
            # 计算速度变化百分比
            if init_speed != 0:
                spd_change_percentage = (current_speed - init_speed) / init_speed
            else:
                spd_change_percentage = -1

            # 根据速度变化判断换道质量
            if spd_change_percentage < self.min_speed_gain:
                # 换道后速度下降太多 -> 惩罚
                r_lane_change = -1 * r_speed * self.lane_change_penalty
            else:
                # 换道后速度保持/提升 -> 奖励
                r_speed *= self.lane_change_reward
                self.episode_stats['successful_lane_changes'] += 1

        # ============================================================
        # 5. 总奖励
        # ============================================================
        positives = r_speed
        negatives = r_lane_change
        total_reward = positives + negatives

        # 更新统计
        self.episode_stats['total_reward'] += total_reward
        self.last_speed = current_speed

        # 构建详细信息
        components = {
            'speed': r_speed,
            'lane_change': r_lane_change,
        }

        reward_details = {
            'total': total_reward,
            'base': total_reward,
            'components': components,
            'type': 'normal',
            'training_step': self.training_step,
            'collision_penalty': self.collision_penalty,
            'off_road_penalty': self.off_the_road_penalty,
            'episode_stats': self.episode_stats.copy(),
            # 为了兼容，添加空的字段
            'shaping': 0.0,
            'milestone': 0.0,
            'milestone_details': [],
            'improvement': 0.0,
            'weights': {'speed': 1.0, 'lane_change': 1.0},
            'stage': 'baseline',
        }

        return total_reward, reward_details

    def get_summary(self):
        """获取摘要"""
        return {
            'training_step': self.training_step,
            'collision_penalty': self.collision_penalty,
            'off_road_penalty': self.off_the_road_penalty,
            'current_weights': {'speed': 1.0, 'lane_change': 1.0},
            'episode_stats': self.episode_stats
        }


class MilestoneTracker:
    """里程碑追踪器 - 保持原有实现"""
    def __init__(self):
        self.milestones = {
            'speed_stable_5s': {'achieved': False, 'reward': 0.8, 'duration': 0.0},
            'speed_stable_10s': {'achieved': False, 'reward': 1.2, 'duration': 0.0},
            'collision_free_50m': {'achieved': False, 'reward': 0.5, 'distance': 0.0},
            'collision_free_100m': {'achieved': False, 'reward': 0.8, 'distance': 0.0},
            'collision_free_200m': {'achieved': False, 'reward': 1.5, 'distance': 0.0},
            'safe_lane_change': {'achieved': False, 'reward': 1.5, 'count': 0},
            'progress_25': {'achieved': False, 'reward': 0.5},
            'progress_50': {'achieved': False, 'reward': 0.8},
            'progress_75': {'achieved': False, 'reward': 1.0},
        }
        self.speed_stable_start = None
        self.last_collision_s = 0.0
        self.last_lane_change_success = False

    def reset(self):
        for milestone in self.milestones.values():
            milestone['achieved'] = False
            if 'duration' in milestone:
                milestone['duration'] = 0.0
            if 'distance' in milestone:
                milestone['distance'] = 0.0
            if 'count' in milestone:
                milestone['count'] = 0
        self.speed_stable_start = None
        self.last_collision_s = 0.0

    def check_milestones(self, state_info):
        rewards = []
        current_speed = state_info['current_speed']
        target_speed = state_info['target_speed']
        distance_traveled = state_info['distance_traveled']
        track_length = state_info.get('track_length', 1000.0)
        progress = distance_traveled / track_length
        dt = state_info.get('dt', 0.05)

        # 速度稳定性
        speed_error = abs(current_speed - target_speed)
        if speed_error < target_speed * 0.15:
            if self.speed_stable_start is None:
                self.speed_stable_start = 0.0
            self.speed_stable_start += dt

            if self.speed_stable_start >= 5.0 and not self.milestones['speed_stable_5s']['achieved']:
                self.milestones['speed_stable_5s']['achieved'] = True
                rewards.append(('speed_stable_5s', self.milestones['speed_stable_5s']['reward']))

            if self.speed_stable_start >= 10.0 and not self.milestones['speed_stable_10s']['achieved']:
                self.milestones['speed_stable_10s']['achieved'] = True
                rewards.append(('speed_stable_10s', self.milestones['speed_stable_10s']['reward']))
        else:
            self.speed_stable_start = None

        # 无碰撞行驶距离
        collision_free_distance = distance_traveled - self.last_collision_s
        if collision_free_distance >= 50 and not self.milestones['collision_free_50m']['achieved']:
            self.milestones['collision_free_50m']['achieved'] = True
            rewards.append(('collision_free_50m', self.milestones['collision_free_50m']['reward']))

        if collision_free_distance >= 100 and not self.milestones['collision_free_100m']['achieved']:
            self.milestones['collision_free_100m']['achieved'] = True
            rewards.append(('collision_free_100m', self.milestones['collision_free_100m']['reward']))

        if collision_free_distance >= 200 and not self.milestones['collision_free_200m']['achieved']:
            self.milestones['collision_free_200m']['achieved'] = True
            rewards.append(('collision_free_200m', self.milestones['collision_free_200m']['reward']))

        # 安全换道
        if state_info.get('lane_change_just_completed', False) and state_info.get('lane_change_safe', False):
            if not self.last_lane_change_success:
                self.milestones['safe_lane_change']['count'] += 1
                rewards.append(('safe_lane_change', self.milestones['safe_lane_change']['reward']))
                self.last_lane_change_success = True
        else:
            self.last_lane_change_success = False

        # 进度里程碑
        if progress >= 0.25 and not self.milestones['progress_25']['achieved']:
            self.milestones['progress_25']['achieved'] = True
            rewards.append(('progress_25', self.milestones['progress_25']['reward']))

        if progress >= 0.50 and not self.milestones['progress_50']['achieved']:
            self.milestones['progress_50']['achieved'] = True
            rewards.append(('progress_50', self.milestones['progress_50']['reward']))

        if progress >= 0.75 and not self.milestones['progress_75']['achieved']:
            self.milestones['progress_75']['achieved'] = True
            rewards.append(('progress_75', self.milestones['progress_75']['reward']))

        return rewards

    def record_collision(self, distance_traveled):
        self.last_collision_s = distance_traveled


# ============================================================
# ⭐ 新增：换道统计追踪器
# ============================================================
class LaneChangeTracker:
    """
    详细的换道统计追踪器

    追踪指标：
    1. 换道尝试/成功/失败次数
    2. 换道持续时间
    3. 换道距离
    4. 换道时的安全指标（TTC、距离）
    5. 换道原因分析
    """

    def __init__(self):
        # 当前换道状态
        self.is_lane_changing = False
        self.lane_change_start_time = None
        self.lane_change_start_s = None
        self.lane_change_start_d = None
        self.lane_change_target_d = None

        # 累计统计
        self.total_attempts = 0  # 换道尝试次数
        self.successful_changes = 0  # 成功次数
        self.failed_changes = 0  # 失败次数（碰撞/取消）
        self.abandoned_changes = 0  # 中途放弃次数

        # 换道详细数据（每次换道）
        self.lane_change_history = []

        # 当前Episode统计
        self.episode_attempts = 0
        self.episode_successes = 0
        self.episode_failures = 0

        # 换道原因统计
        self.reason_stats = {
            'avoid_slow_vehicle': 0,  # 避让慢车
            'overtake': 0,  # 超车
            'exploration': 0,  # 探索
            'unknown': 0  # 未知原因
        }

    def start_lane_change(self, current_time, current_s, current_d, target_d,
                         vehicle_ahead_info=None, reason='unknown'):
        """开始换道"""
        if self.is_lane_changing:
            # 已经在换道中，不重复记录
            return

        self.is_lane_changing = True
        self.lane_change_start_time = current_time
        self.lane_change_start_s = current_s
        self.lane_change_start_d = current_d
        self.lane_change_target_d = target_d

        self.total_attempts += 1
        self.episode_attempts += 1
        self.reason_stats[reason] += 1

        # 记录初始状态
        self.current_lane_change_data = {
            'start_time': current_time,
            'start_s': current_s,
            'start_d': current_d,
            'target_d': target_d,
            'reason': reason,
            'initial_vehicle_ahead': vehicle_ahead_info.copy() if vehicle_ahead_info else None,
            'min_ttc': float('inf'),
            'min_distance': float('inf'),
            'trajectory': [],  # 记录换道过程中的位置
        }

    def update_lane_change(self, current_s, current_d, vehicle_ahead_info=None):
        """更新换道进度"""
        if not self.is_lane_changing:
            return

        # 记录轨迹点
        self.current_lane_change_data['trajectory'].append({
            's': current_s,
            'd': current_d,
        })

        # 更新安全指标
        if vehicle_ahead_info and vehicle_ahead_info.get('exists', False):
            ttc = vehicle_ahead_info.get('ttc', float('inf'))
            distance = vehicle_ahead_info.get('distance', float('inf'))

            if ttc < self.current_lane_change_data['min_ttc']:
                self.current_lane_change_data['min_ttc'] = ttc
            if distance < self.current_lane_change_data['min_distance']:
                self.current_lane_change_data['min_distance'] = distance

    def complete_lane_change(self, current_time, current_s, current_d, success=True,
                            collision=False):
        """完成换道"""
        if not self.is_lane_changing:
            return None

        self.is_lane_changing = False

        # 计算换道统计
        duration = current_time - self.lane_change_start_time
        distance_traveled = current_s - self.lane_change_start_s
        lateral_displacement = abs(current_d - self.lane_change_start_d)

        # 完成记录
        self.current_lane_change_data.update({
            'end_time': current_time,
            'end_s': current_s,
            'end_d': current_d,
            'duration': duration,
            'distance_traveled': distance_traveled,
            'lateral_displacement': lateral_displacement,
            'success': success,
            'collision': collision,
        })

        # 更新统计
        if success and not collision:
            self.successful_changes += 1
            self.episode_successes += 1
        else:
            self.failed_changes += 1
            self.episode_failures += 1

        # 保存到历史
        self.lane_change_history.append(self.current_lane_change_data.copy())

        return self.current_lane_change_data

    def abandon_lane_change(self, current_time, reason='cancelled'):
        """放弃换道"""
        if not self.is_lane_changing:
            return

        self.is_lane_changing = False
        self.abandoned_changes += 1

        # 记录放弃信息
        if hasattr(self, 'current_lane_change_data'):
            self.current_lane_change_data['abandoned'] = True
            self.current_lane_change_data['abandon_reason'] = reason
            self.lane_change_history.append(self.current_lane_change_data.copy())

    def get_episode_stats(self):
        """获取当前Episode统计"""
        success_rate = (self.episode_successes / self.episode_attempts
                       if self.episode_attempts > 0 else 0.0)

        return {
            'attempts': self.episode_attempts,
            'successes': self.episode_successes,
            'failures': self.episode_failures,
            'success_rate': success_rate,
        }

    def get_overall_stats(self):
        """获取总体统计"""
        success_rate = (self.successful_changes / self.total_attempts
                       if self.total_attempts > 0 else 0.0)

        # 计算平均换道时间
        successful_lane_changes = [lc for lc in self.lane_change_history
                                  if lc.get('success', False)]
        avg_duration = (np.mean([lc['duration'] for lc in successful_lane_changes])
                       if successful_lane_changes else 0.0)
        avg_distance = (np.mean([lc['distance_traveled'] for lc in successful_lane_changes])
                       if successful_lane_changes else 0.0)

        return {
            'total_attempts': self.total_attempts,
            'successful_changes': self.successful_changes,
            'failed_changes': self.failed_changes,
            'abandoned_changes': self.abandoned_changes,
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'avg_distance': avg_distance,
            'reason_stats': self.reason_stats.copy(),
        }

    def reset_episode(self):
        """重置Episode统计"""
        self.episode_attempts = 0
        self.episode_successes = 0
        self.episode_failures = 0
        self.is_lane_changing = False

    def print_summary(self):
        """打印统计摘要"""
        overall = self.get_overall_stats()
        print('\\n' + '='*80)
        print('LANE CHANGE STATISTICS SUMMARY')
        print('='*80)
        print(f"Total Attempts:     {overall['total_attempts']}")
        print(f"Successful:         {overall['successful_changes']} ({overall['success_rate']*100:.1f}%)")
        print(f"Failed:             {overall['failed_changes']}")
        print(f"Abandoned:          {overall['abandoned_changes']}")
        print(f"\\nAverage Duration:   {overall['avg_duration']:.2f}s")
        print(f"Average Distance:   {overall['avg_distance']:.2f}m")
        print(f"\\nReasons:")
        for reason, count in overall['reason_stats'].items():
            print(f"  {reason:20s}: {count}")
        print('='*80 + '\\n')



class ImprovedRewardSystem:
    """
    增强版奖励系统 - 提升学习效率，减少碰撞

    ⭐ 新增功能:
    1. 换道过程塑形奖励（密集信号）
    2. 与 LaneChangeTracker 集成
    """

    def __init__(self, config):
        # 基础参数
        self.targetSpeed = config.get('target_speed', 30.0)
        self.maxSpeed = config.get('max_speed', 40.0)
        self.lane_width = config.get('lane_width', 3.5)
        self.track_length = config.get('track_length', 1000.0)

        # ⭐ 新增：换道塑形参数
        self.use_lane_change_shaping = True
        self.lane_change_shaping_weight = 0.25
        self.last_d = None
        self.last_action = None

        # 课程学习配置
        self.curriculum_stages = {
            'stage1': {  # 0-20000步：学习换道避障
                'threshold': 20000,
                'weights': {
                    'speed': 0.10,
                    'safety': 0.30,  # ⭐ 降低安全权重，给换道让路
                    'lane_keeping': 0.10,
                    'comfort': 0.05,
                    'efficiency': 0.15,  # ⭐ 提高效率权重
                    'progress': 0.15,
                    'lane_change_shaping': 0.15,  # ⭐ 新增换道塑形
                }
            },
            'stage2': {  # 20000-50000步
                'threshold': 50000,
                'weights': {
                    'speed': 0.15,
                    'safety': 0.38,
                    'lane_keeping': 0.12,
                    'comfort': 0.08,
                    'efficiency': 0.12,
                    'progress': 0.15,
                    'lane_change_shaping': 0.00,  # 后期不需要
                }
            },
            'stage3': {  # 50000步以上
                'threshold': float('inf'),
                'weights': {
                    'speed':       0.20,
                    'safety':      0.30,
                    'lane_keeping':0.12,
                    'comfort':     0.10,
                    'efficiency':  0.15,
                    'progress':    0.13,
                    'lane_change_shaping': 0.00,
                }
            }
        }

        self.weights = self.curriculum_stages['stage3']['weights']

        # 安全参数
        self.safe_time_gap = 1.5
        self.min_safe_distance = 6.0
        self.critical_distance = 4.0
        self.emergency_distance = 2.5
        self.comfortable_distance = 25.0

        # 舒适性参数
        self.max_comfortable_acc = 3.0
        self.max_comfortable_jerk = 2.5

        # 惩罚参数
        self.training_step = 0
        self.collision_penalty_min = -1.0
        self.collision_penalty_max = -5.0
        self.off_road_penalty_min = -0.8
        self.off_road_penalty_max = -3.0
        self.penalty_warmup_steps = 50000

        self.collision_penalty = self.collision_penalty_min
        self.off_road_penalty = self.off_road_penalty_min

        # 奖励塑形参数
        self.use_potential_shaping = True
        self.use_milestone_rewards = True
        self.use_improvement_rewards = True
        self.potential_weight = 0.12
        self.milestone_weight = 0.20
        self.improvement_weight = 0.08

        # 存活奖励
        self.use_survival_bonus = True
        self.survival_bonus = 1.0

        # 危险预警系统
        self.use_danger_warning = True
        self.danger_warning_weight = 0.10
        self.consecutive_danger_steps = 0
        self.max_danger_steps_before_heavy_penalty = 8

        # 历史状态
        self.last_potential = 0.0
        self.last_s = 0.0
        self.last_speed = 0.0
        self.last_lateral_error = 0.0
        self.last_distance_to_front = float('inf')
        self.last_acc = 0.0
        self.last_ttc = float('inf')
        self.speed_history = deque(maxlen=10)
        self.acc_history = deque(maxlen=10)
        self.distance_history = deque(maxlen=20)

        # 换道和被堵追踪
        self.stuck_behind_slow_vehicle_steps = 0
        self.last_lane_d = 0.0
        self.consecutive_safe_steps = 0

        self.milestone_tracker = MilestoneTracker()

        self.episode_stats = {
            'total_reward': 0.0,
            'speed_violations': 0,
            'safety_violations': 0,
            'comfort_violations': 0,
            'successful_lane_changes': 0,
            'attempted_lane_changes': 0,  # ⭐ 新增
            'stuck_episodes': 0,
            'danger_warnings': 0,
            'near_misses': 0,
            'lane_change_shaping_reward': 0.0,  # ⭐ 新增
        }

    def reset(self):
        self.last_potential = 0.0
        self.last_s = 0.0
        self.last_speed = 0.0
        # ⭐ 修复：横向误差初始化为无穷大，让第一步容易获得改善
        self.last_lateral_error = float('inf')
        self.last_distance_to_front = float('inf')
        self.last_acc = 0.0
        self.last_ttc = float('inf')
        self.speed_history.clear()
        self.acc_history.clear()
        self.distance_history.clear()
        self.stuck_behind_slow_vehicle_steps = 0
        self.last_lane_d = 0.0
        self.consecutive_safe_steps = 0
        self.consecutive_danger_steps = 0
        self.milestone_tracker.reset()

        # ⭐ 重置换道追踪
        self.last_d = None
        self.last_action = None

        self.episode_stats = {
            'total_reward': 0.0,
            'speed_violations': 0,
            'safety_violations': 0,
            'comfort_violations': 0,
            'successful_lane_changes': 0,
            'attempted_lane_changes': 0,
            'stuck_episodes': 0,
            'danger_warnings': 0,
            'near_misses': 0,
            'lane_change_shaping_reward': 0.0,
        }

    def update_curriculum_weights(self):
        for stage_name, stage_config in self.curriculum_stages.items():
            if self.training_step < stage_config['threshold']:
                self.weights = stage_config['weights']
                return stage_name
        return 'stage3'

    def update_penalties(self):
        progress = min(1.0, self.training_step / self.penalty_warmup_steps)
        smooth_progress = progress ** 0.7

        self.collision_penalty = (
            self.collision_penalty_min +
            smooth_progress * (self.collision_penalty_max - self.collision_penalty_min)
        )

        self.off_road_penalty = (
            self.off_road_penalty_min +
            smooth_progress * (self.off_road_penalty_max - self.off_road_penalty_min)
        )

    # ============================================================
    # ⭐ 新增：换道过程塑形奖励
    # ============================================================
    def compute_lane_change_shaping_reward(self, current_d, target_d, action,
                                          vehicle_ahead_info, current_speed):
        """
        换道过程塑形奖励 - 提供密集的学习信号

        奖励结构：
        1. 尝试奖励：采取横向动作 (+0.1)
        2. 进度奖励：朝目标车道移动 (+0.3)
        3. 方向奖励：动作方向正确 (+0.2)
        4. 必要性奖励：前方有慢车时换道 (+0.2)
        5. 完成奖励：到达目标车道 (+0.5)

        总计：最高 +1.3
        """
        if self.last_d is None:
            self.last_d = current_d
            self.last_action = action
            return 0.0

        reward = 0.0

        # 计算横向位移
        d_change = current_d - self.last_d
        d_to_target = target_d - current_d
        d_from_start = current_d - self.last_d

        # 1. 尝试奖励：检测到横向动作
        if abs(action) > 0.1:  # 有明显的横向意图
            reward += 0.1
            self.episode_stats['attempted_lane_changes'] += 1

        # 2. 进度奖励：正在接近目标车道
        if abs(d_to_target) < abs(target_d - self.last_d):
            # 距离目标更近了
            progress = abs(self.last_d - current_d)
            reward += 0.3 * min(1.0, progress / 0.5)  # 每接近0.5米给0.3

        # 3. 方向奖励：动作方向与目标一致
        if d_to_target != 0:
            action_direction = np.sign(action)
            target_direction = np.sign(d_to_target)
            if action_direction == target_direction:
                reward += 0.2 * abs(action)  # 动作越大，奖励越多
            else:
                # 方向错误，轻微惩罚
                reward -= 0.1

        # 4. 必要性奖励：前方有慢车，且正在换道
        if vehicle_ahead_info.get('exists', False):
            distance = vehicle_ahead_info.get('distance', float('inf'))
            ahead_speed = vehicle_ahead_info.get('speed', 0)

            # 前方车慢，且距离较近
            if ahead_speed < current_speed * 0.8 and distance < 20:
                if abs(d_change) > 0.05:  # 正在换道
                    reward += 0.2

        # 5. 完成奖励：成功到达目标车道附近
        if abs(d_to_target) < 0.3:  # 距离目标车道很近了
            reward += 0.5

        # 6. 持续性奖励：连续朝目标方向移动
        if self.last_action is not None:
            if np.sign(action) == np.sign(self.last_action) and abs(action) > 0.1:
                reward += 0.1  # 鼓励连续动作

        # 7. 平滑性惩罚：动作变化太剧烈
        if self.last_action is not None:
            action_change = abs(action - self.last_action)
            if action_change > 0.5:
                reward -= 0.15  # 惩罚突然的大幅度转向

        # 更新历史
        self.last_d = current_d
        self.last_action = action

        # 累计统计
        self.episode_stats['lane_change_shaping_reward'] += reward

        return reward

    def compute_speed_reward(self, current_speed):
        """速度奖励 - 分段函数，确保正常速度时为正"""
        speed_ratio = current_speed / self.targetSpeed

        if 0.85 <= speed_ratio <= 1.10:
            reward = 1.0 - 2.0 * abs(speed_ratio - 1.0)
        elif 0.70 <= speed_ratio < 0.85:
            reward = 0.5 + (speed_ratio - 0.70) / 0.15 * 0.2
        elif 1.10 < speed_ratio <= 1.20:
            reward = 0.7 - (speed_ratio - 1.10) / 0.10 * 0.3
        elif 0.50 <= speed_ratio < 0.70:
            reward = 0.2 + (speed_ratio - 0.50) / 0.20 * 0.3
        elif speed_ratio > 1.20:
            reward = max(-0.3, 0.4 - (speed_ratio - 1.20) * 2.0)
            if speed_ratio > self.maxSpeed / self.targetSpeed * 1.1:
                self.episode_stats['speed_violations'] += 1
        elif 0.30 <= speed_ratio < 0.50:
            reward = 0.1 * (speed_ratio / 0.30)
        else:
            reward = -0.05
            self.episode_stats['speed_violations'] += 1

        return reward

    def compute_ttc(self, distance, ego_speed, ahead_speed):
        """计算碰撞时间(Time To Collision)"""
        relative_speed = ego_speed - ahead_speed
        if relative_speed <= 0.1:
            return float('inf')
        if distance <= 0:
            return 0.0
        return distance / relative_speed

    def compute_safety_reward(self, ego_speed, vehicle_ahead_info):
        """安全奖励 - 核心改进"""
        if not vehicle_ahead_info.get('exists', False):
            self.consecutive_danger_steps = 0
            return 0.9

        distance = vehicle_ahead_info.get('distance', float('inf'))
        ahead_speed = vehicle_ahead_info.get('speed', 0)

        self.distance_history.append(distance)
        ttc = self.compute_ttc(distance, ego_speed, ahead_speed)

        desired_distance = max(
            self.min_safe_distance,
            ego_speed * self.safe_time_gap
        )

        # 距离奖励
        if distance >= self.comfortable_distance:
            distance_reward = 0.9
            self.consecutive_danger_steps = 0
        elif distance >= desired_distance * 1.3:
            ratio = (distance - desired_distance * 1.3) / (self.comfortable_distance - desired_distance * 1.3)
            distance_reward = 0.6 + 0.3 * ratio
            self.consecutive_danger_steps = 0
        elif distance >= desired_distance:
            ratio = (distance - desired_distance) / (desired_distance * 0.3)
            distance_reward = 0.3 + 0.3 * ratio
            self.consecutive_danger_steps = 0
        elif distance >= self.critical_distance:
            ratio = (distance - self.critical_distance) / (desired_distance - self.critical_distance)
            distance_reward = 0.1 + 0.2 * ratio
            self.consecutive_danger_steps += 1
            self.episode_stats['safety_violations'] += 1
        elif distance >= self.emergency_distance:
            ratio = (distance - self.emergency_distance) / (self.critical_distance - self.emergency_distance)
            distance_reward = 0.05 + 0.05 * ratio
            self.consecutive_danger_steps += 1
            self.episode_stats['danger_warnings'] += 1
        else:
            distance_reward = 0.01
            self.consecutive_danger_steps += 1
            self.episode_stats['near_misses'] += 1

        # TTC奖励
        ttc_reward = 0.5
        if ttc < float('inf'):
            if ttc >= 4.0:
                ttc_reward = 0.8
            elif ttc >= 2.5:
                ttc_reward = 0.5
            elif ttc >= 1.5:
                ttc_reward = 0.3
                self.episode_stats['safety_violations'] += 1
            elif ttc >= 0.8:
                ttc_reward = 0.15
                self.episode_stats['danger_warnings'] += 1
            else:
                ttc_reward = 0.05
                self.episode_stats['near_misses'] += 1

        # 距离变化奖励
        distance_change_reward = 0.0
        if len(self.distance_history) >= 2:
            distance_change = distance - self.distance_history[-2]
            if distance_change > 0.5:
                distance_change_reward = min(0.1, distance_change * 0.05)
            elif distance_change < -0.5:
                distance_change_reward = max(-0.1, distance_change * 0.05)

        # 连续危险惩罚
        consecutive_danger_penalty = 0.0
        if self.consecutive_danger_steps > self.max_danger_steps_before_heavy_penalty:
            consecutive_danger_penalty = -0.2 * min(5,
                                                    self.consecutive_danger_steps - self.max_danger_steps_before_heavy_penalty)

        safety_reward = (
                0.5 * distance_reward +
                0.35 * ttc_reward +
                0.1 * distance_change_reward +
                0.05 * consecutive_danger_penalty
        )

        self.last_ttc = ttc
        self.last_distance_to_front = distance

        return np.clip(safety_reward, 0.0, 1.0)

    def compute_danger_warning(self, ego_speed, vehicle_ahead_info):
        """危险预警奖励 - 在碰撞前给出强烈信号"""
        if not vehicle_ahead_info.get('exists', False):
            return 0.15

        distance = vehicle_ahead_info.get('distance', float('inf'))
        ahead_speed = vehicle_ahead_info.get('speed', 0)
        ttc = self.compute_ttc(distance, ego_speed, ahead_speed)

        danger_level = 0.0

        if distance < self.emergency_distance:
            danger_level = max(danger_level, 1.0)
        elif distance < self.critical_distance:
            danger_level = max(danger_level, 0.7)
        elif distance < self.min_safe_distance:
            danger_level = max(danger_level, 0.4)

        if ttc < 1.0:
            danger_level = max(danger_level, 1.0)
        elif ttc < 2.0:
            danger_level = max(danger_level, 0.8)
        elif ttc < 3.0:
            danger_level = max(danger_level, 0.5)

        if danger_level > 0.7:
            warning_reward = -0.5 * danger_level
        elif danger_level > 0.3:
            warning_reward = -0.3 * danger_level
        elif danger_level > 0:
            warning_reward = -0.1 * danger_level
        else:
            warning_reward = 0.15

        return warning_reward

    def compute_lane_keeping_reward(self, lateral_error, is_lane_changing):
        """车道保持奖励"""
        if is_lane_changing:
            return 0.3

        max_tolerable_error = self.lane_width * 0.4

        if lateral_error <= max_tolerable_error * 0.3:
            reward = 0.9
        elif lateral_error <= max_tolerable_error * 0.6:
            ratio = (lateral_error - max_tolerable_error * 0.3) / (max_tolerable_error * 0.3)
            reward = 0.7 + 0.2 * (1 - ratio)
        elif lateral_error <= max_tolerable_error:
            ratio = (lateral_error - max_tolerable_error * 0.6) / (max_tolerable_error * 0.4)
            reward = 0.4 + 0.3 * (1 - ratio)
        elif lateral_error <= max_tolerable_error * 1.5:
            ratio = (lateral_error - max_tolerable_error) / (max_tolerable_error * 0.5)
            reward = 0.1 + 0.3 * (1 - ratio)
        else:
            reward = 0.0

        return reward

    def compute_comfort_reward(self, acceleration, jerk):
        """舒适性奖励"""
        acc_ratio = min(abs(acceleration) / self.max_comfortable_acc, 2.0)
        jerk_ratio = min(abs(jerk) / self.max_comfortable_jerk, 2.0)

        if acc_ratio <= 0.5:
            acc_comfort = 1.0
        elif acc_ratio <= 1.0:
            acc_comfort = 0.6 + 0.4 * (1 - (acc_ratio - 0.5) / 0.5)
        elif acc_ratio <= 1.5:
            acc_comfort = 0.3 + 0.3 * (1 - (acc_ratio - 1.0) / 0.5)
        else:
            acc_comfort = max(0.0, 0.3 - 0.3 * (acc_ratio - 1.5) / 0.5)
            self.episode_stats['comfort_violations'] += 1

        if jerk_ratio <= 0.5:
            jerk_comfort = 1.0
        elif jerk_ratio <= 1.0:
            jerk_comfort = 0.6 + 0.4 * (1 - (jerk_ratio - 0.5) / 0.5)
        elif jerk_ratio <= 1.5:
            jerk_comfort = 0.3 + 0.3 * (1 - (jerk_ratio - 1.0) / 0.5)
        else:
            jerk_comfort = max(0.0, 0.3 - 0.3 * (jerk_ratio - 1.5) / 0.5)

        comfort_reward = 0.6 * acc_comfort + 0.4 * jerk_comfort

        return comfort_reward

    def compute_efficiency_reward(self, current_speed, vehicle_ahead_info, is_lane_changing):
        """效率奖励"""
        speed_ratio = current_speed / self.targetSpeed

        if speed_ratio >= 0.9:
            base_efficiency = 0.8
        elif speed_ratio >= 0.7:
            base_efficiency = 0.4 + 0.4 * (speed_ratio - 0.7) / 0.2
        elif speed_ratio >= 0.5:
            base_efficiency = 0.2 + 0.2 * (speed_ratio - 0.5) / 0.2
        else:
            base_efficiency = 0.1

        stuck_penalty = 0.0
        if vehicle_ahead_info.get('exists', False):
            distance = vehicle_ahead_info.get('distance', float('inf'))
            ahead_speed = vehicle_ahead_info.get('speed', 0)

            if ahead_speed < current_speed * 0.85 and 10 < distance < 30:
                self.stuck_behind_slow_vehicle_steps += 1
                if self.stuck_behind_slow_vehicle_steps > 30:
                    stuck_penalty = -0.15
                    self.episode_stats['stuck_episodes'] += 1
            else:
                self.stuck_behind_slow_vehicle_steps = 0
        else:
            self.stuck_behind_slow_vehicle_steps = 0

        lane_change_bonus = 0.0
        if is_lane_changing:
            lane_change_bonus = 0.2

        efficiency_reward = base_efficiency + stuck_penalty + lane_change_bonus
        return np.clip(efficiency_reward, -0.3, 1.0)

    def compute_progress_reward(self, distance_traveled):
        """进度奖励 - 增强，让智能体学会前进"""
        distance_increment = distance_traveled - self.last_s

        if distance_increment > 0:
            progress_reward = min(1.0, distance_increment / 2.0)
        elif distance_increment > -1:
            progress_reward = 0.0
        else:
            progress_reward = -0.1

        self.last_s = distance_traveled
        return progress_reward

    def compute_potential(self, state_info):
        """潜力函数"""
        progress = state_info['distance_traveled'] / self.track_length
        progress_potential = progress

        speed_error = abs(state_info['current_speed'] - self.targetSpeed)
        speed_potential = max(0, 1 - (speed_error / self.targetSpeed))

        if state_info['vehicle_ahead']['exists']:
            distance = state_info['vehicle_ahead']['distance']
            ahead_speed = state_info['vehicle_ahead'].get('speed', 0)
            ego_speed = state_info['current_speed']

            desired_distance = max(
                self.min_safe_distance,
                ego_speed * self.safe_time_gap
            )
            distance_potential = min(1.0, distance / (desired_distance * 1.5))

            ttc = self.compute_ttc(distance, ego_speed, ahead_speed)
            ttc_potential = min(1.0, ttc / 5.0) if ttc < float('inf') else 1.0

            safety_potential = 0.6 * distance_potential + 0.4 * ttc_potential
        else:
            safety_potential = 1.0

        lateral_error = abs(state_info.get('ego_d', 0) - state_info.get('target_d', 0))
        lane_potential = max(0, 1 - lateral_error / self.lane_width)

        potential = (
                0.30 * progress_potential +
                0.25 * speed_potential +
                0.35 * safety_potential +
                0.10 * lane_potential
        )

        return potential

    def compute_improvement_rewards(self, state_info):
        """改善奖励"""
        improvement_reward = 0.0

        current_speed_error = abs(state_info['current_speed'] - self.targetSpeed)
        last_speed_error = abs(self.last_speed - self.targetSpeed)
        if current_speed_error < last_speed_error:
            improvement_reward += 0.12

        if state_info['vehicle_ahead']['exists']:
            current_distance = state_info['vehicle_ahead']['distance']
            if current_distance > self.last_distance_to_front + 0.5:
                improvement_reward += 0.20

        current_lateral_error = abs(state_info.get('ego_d', 0) - state_info.get('target_d', 0))
        if current_lateral_error < self.last_lateral_error:
            improvement_reward += 0.08
        self.last_lateral_error = current_lateral_error

        if state_info['vehicle_ahead']['exists']:
            ego_speed = state_info['current_speed']
            ahead_speed = state_info['vehicle_ahead'].get('speed', 0)
            distance = state_info['vehicle_ahead']['distance']
            current_ttc = self.compute_ttc(distance, ego_speed, ahead_speed)

            if current_ttc > self.last_ttc + 0.5:
                improvement_reward += 0.15

        return improvement_reward

    def compute_total_reward(self, state_info):
        """计算总奖励 - ⭐ 集成换道塑形"""
        self.training_step += 1
        self.update_penalties()
        current_stage = self.update_curriculum_weights()

        # ⭐ 修复：终止状态也计算各项奖励，便于调试和分析
        # 先计算所有基础奖励（无论是否终止）
        speed_reward = self.compute_speed_reward(state_info['current_speed'])

        safety_reward = self.compute_safety_reward(
            state_info['current_speed'],
            state_info.get('vehicle_ahead', {'exists': False, 'distance': float('inf'), 'speed': 0})
        )

        lateral_error = abs(state_info.get('ego_d', 0) - state_info.get('target_d', 0))
        lane_keeping_reward = self.compute_lane_keeping_reward(
            lateral_error,
            state_info.get('is_lane_changing', False)
        )

        current_acc = state_info.get('current_acc', 0)
        dt = state_info.get('dt', 0.05)
        jerk = abs(current_acc - self.last_acc) / dt if dt > 0 else 0
        self.last_acc = current_acc
        comfort_reward = self.compute_comfort_reward(current_acc, jerk)

        efficiency_reward = self.compute_efficiency_reward(
            state_info['current_speed'],
            state_info.get('vehicle_ahead', {'exists': False}),
            state_info.get('is_lane_changing', False)
        )

        progress_reward = self.compute_progress_reward(
            state_info.get('distance_traveled', 0)
        )

        # 换道塑形奖励
        lane_change_shaping_reward = 0.0
        if self.use_lane_change_shaping and self.weights.get('lane_change_shaping', 0) > 0:
            lane_change_shaping_reward = self.compute_lane_change_shaping_reward(
                state_info.get('ego_d', 0),
                state_info.get('target_d', 0),
                state_info.get('action', 0.0),
                state_info.get('vehicle_ahead', {'exists': False}),
                state_info['current_speed']
            )

        # 危险预警
        danger_warning_reward = 0.0
        if self.use_danger_warning:
            danger_warning_reward = self.compute_danger_warning(
                state_info['current_speed'],
                state_info.get('vehicle_ahead', {'exists': False})
            )

        # 基础奖励（加权和）
        base_reward = (
                self.weights.get('speed', 0) * speed_reward +
                self.weights.get('safety', 0) * safety_reward +
                self.weights.get('lane_keeping', 0) * lane_keeping_reward +
                self.weights.get('comfort', 0) * comfort_reward +
                self.weights.get('efficiency', 0) * efficiency_reward +
                self.weights.get('progress', 0) * progress_reward +
                self.weights.get('lane_change_shaping', 0) * lane_change_shaping_reward
        )

        # 构建组件字典（所有情况都要包含）
        components = {
            'speed': speed_reward,
            'safety': safety_reward,
            'danger_warning': danger_warning_reward,
            'lane_keeping': lane_keeping_reward,
            'lane_change_shaping': lane_change_shaping_reward,
            'comfort': comfort_reward,
            'efficiency': efficiency_reward,
            'progress': progress_reward,
            'survival': 0.0,  # 稍后更新
        }

        # ⭐ 终止状态检查 - 现在包含完整的奖励细节
        if state_info.get('collision', False):
            self.milestone_tracker.record_collision(state_info['distance_traveled'])

            extra_penalty = 0.0
            if self.consecutive_danger_steps > 5:
                extra_penalty = -0.5

            total_penalty = self.collision_penalty + extra_penalty

            # ⭐ 返回完整的奖励结构
            return total_penalty, {
                'total': total_penalty,
                'base': base_reward,  # ⭐ 添加base
                'components': components,  # ⭐ 添加components
                'shaping': 0.0,
                'milestone': 0.0,
                'improvement': 0.0,
                'weights': self.weights.copy(),
                'type': 'collision',
                'stage': current_stage,
                'training_step': self.training_step,
                'collision_penalty': self.collision_penalty,
                'off_road_penalty': self.off_road_penalty,
                'consecutive_danger_steps': self.consecutive_danger_steps,
                'episode_stats': self.episode_stats.copy(),
                'milestone_details': []
            }

        if state_info.get('off_road', False):
            # ⭐ 返回完整的奖励结构
            return self.off_road_penalty, {
                'total': self.off_road_penalty,
                'base': base_reward,  # ⭐ 添加base
                'components': components,  # ⭐ 添加components
                'shaping': 0.0,
                'milestone': 0.0,
                'improvement': 0.0,
                'weights': self.weights.copy(),
                'type': 'off_road',
                'stage': current_stage,
                'training_step': self.training_step,
                'collision_penalty': self.collision_penalty,
                'off_road_penalty': self.off_road_penalty,
                'episode_stats': self.episode_stats.copy(),
                'milestone_details': []
            }

        # 正常情况继续计算总奖励
        total_reward = base_reward

        # 危险预警
        if self.use_danger_warning:
            total_reward += self.danger_warning_weight * danger_warning_reward

        # 存活奖励
        survival_reward = 0.0
        if self.use_survival_bonus:
            survival_reward = self.survival_bonus
            total_reward += survival_reward
            components['survival'] = survival_reward  # 更新组件

        # 潜力塑形
        shaping_reward = 0.0
        if self.use_potential_shaping:
            current_potential = self.compute_potential(state_info)
            shaping_reward = current_potential - self.last_potential
            total_reward += self.potential_weight * shaping_reward
            self.last_potential = current_potential

        # 里程碑奖励
        milestone_reward = 0.0
        milestone_details = []
        if self.use_milestone_rewards:
            state_info['target_speed'] = self.targetSpeed
            state_info['track_length'] = self.track_length
            state_info['dt'] = dt
            milestones = self.milestone_tracker.check_milestones(state_info)
            for milestone_name, milestone_value in milestones:
                milestone_reward += milestone_value
                milestone_details.append(milestone_name)
            total_reward += self.milestone_weight * milestone_reward

        # 改善奖励
        improvement_reward = 0.0
        if self.use_improvement_rewards:
            improvement_reward = self.compute_improvement_rewards(state_info)
            total_reward += self.improvement_weight * improvement_reward

        # 更新历史
        self.last_speed = state_info['current_speed']
        if state_info['vehicle_ahead']['exists']:
            self.last_distance_to_front = state_info['vehicle_ahead']['distance']
            ego_speed = state_info['current_speed']
            ahead_speed = state_info['vehicle_ahead'].get('speed', 0)
            self.last_ttc = self.compute_ttc(self.last_distance_to_front, ego_speed, ahead_speed)

        total_reward = np.clip(total_reward, -1.0, 4.0)

        self.episode_stats['total_reward'] += total_reward

        reward_details = {
            'total': total_reward,
            'base': base_reward,
            'components': components,
            'shaping': shaping_reward,
            'milestone': milestone_reward,
            'milestone_details': milestone_details,
            'improvement': improvement_reward,
            'weights': self.weights.copy(),
            'stage': current_stage,
            'training_step': self.training_step,
            'collision_penalty': self.collision_penalty,
            'off_road_penalty': self.off_road_penalty,
            'consecutive_danger_steps': self.consecutive_danger_steps,
            'episode_stats': self.episode_stats.copy()
        }

        return total_reward, reward_details

    def get_summary(self):
        return {
            'training_step': self.training_step,
            'collision_penalty': self.collision_penalty,
            'off_road_penalty': self.off_road_penalty,
            'current_weights': self.weights,
            'episode_stats': self.episode_stats
        }


# ============================================================
# ⭐ 新增：Lyapunov理论奖励系统
# ============================================================

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from theory.lyapunov_shaping import (
        LyapunovPotentialShaping,
        LaneChangeDynamicsParams
    )

    LYAPUNOV_AVAILABLE = True
except ImportError:
    LYAPUNOV_AVAILABLE = False
    print("⚠️  Warning: Lyapunov theory module not available")


class LyapunovRewardSystem(ImprovedRewardSystem):
    """
    基于Lyapunov理论的奖励系统

    继承ImprovedRewardSystem，添加理论驱动的势能塑形

    特点：
    1. 保留所有原有奖励组件
    2. 添加Lyapunov侧向势能塑形
    3. 添加安全障碍势能塑形
    4. 理论保证：策略不变性、加速收敛
    """

    def __init__(self, config):
        # 调用父类初始化
        super().__init__(config)

        # 检查Lyapunov模块是否可用
        if not LYAPUNOV_AVAILABLE:
            raise ImportError("Lyapunov theory module not available. Please create theory/lyapunov_shaping.py")

        # 创建Lyapunov塑形参数
        lyapunov_params = LaneChangeDynamicsParams(
            omega_n=config.get('LYAPUNOV_OMEGA_N', 2.0),
            zeta=config.get('LYAPUNOV_ZETA', 0.7),
            ttc_min=config.get('LYAPUNOV_TTC_MIN', 2.5),
            d_min=config.get('LYAPUNOV_D_MIN', 5.0),
            d_comfortable=config.get('LYAPUNOV_D_COMFORTABLE', 20.0),
            alpha_lateral=config.get('LYAPUNOV_ALPHA_LATERAL', 0.5),
            alpha_longitudinal=config.get('LYAPUNOV_ALPHA_LONGITUDINAL', 0.3),
            sigma_safety=config.get('LYAPUNOV_SIGMA_SAFETY', 5.0),
            lane_width=config.get('lane_width', 3.5),
        )

        # 初始化Lyapunov塑形器
        self.lyapunov = LyapunovPotentialShaping(lyapunov_params)

        # Lyapunov塑形权重
        self.lyapunov_weight = config.get('LYAPUNOV_WEIGHT', 0.30)

        # 是否禁用原有的简单换道塑形（避免冲突）
        self.disable_simple_lc_shaping = config.get('DISABLE_SIMPLE_LC_SHAPING', True)
        if self.disable_simple_lc_shaping:
            self.use_lane_change_shaping = False
            print("⚠️  Simple lane change shaping disabled (using Lyapunov instead)")

        # 历史状态（用于Lyapunov塑形）
        self.lyapunov_last_state = None

        print(f"✅ LyapunovRewardSystem initialized (weight: {self.lyapunov_weight:.2f})")

    def reset(self):
        """重置（覆盖父类方法）"""
        super().reset()
        self.lyapunov_last_state = None
        self.lyapunov.reset_episode()

    def _build_lyapunov_state(self, state_info) -> Dict:
        """
        从环境状态构建Lyapunov状态

        Args:
            state_info: 环境提供的状态字典

        Returns:
            lyapunov_state: Lyapunov塑形所需的状态
        """
        vehicle_ahead = state_info.get('vehicle_ahead', {})

        # 估算横向速度（如果没有直接提供）
        # 注意：你的环境可能需要添加d_dot的计算
        d_dot = 0.0  # 默认值，需要根据实际情况修改
        if self.lyapunov_last_state is not None:
            dt = state_info.get('dt', 0.05)
            if dt > 0:
                d_dot = (state_info['ego_d'] - self.lyapunov_last_state['d']) / dt

        lyapunov_state = {
            'd': state_info.get('ego_d', 0.0),
            'd_dot': d_dot,
            'd_target': state_info.get('target_d', 0.0),
            'delta_s': vehicle_ahead.get('distance', float('inf')),
            'v_ego': state_info.get('current_speed', 0.0),
            'v_lead': vehicle_ahead.get('speed', 0.0),
            'has_lead': vehicle_ahead.get('exists', False),
        }

        return lyapunov_state

    def compute_total_reward(self, state_info):
        """
        计算总奖励（覆盖父类方法）

        在原有奖励基础上添加Lyapunov塑形
        """
        # 1. 调用父类方法获取原有奖励
        base_reward, reward_details = super().compute_total_reward(state_info)

        # 2. 如果是终止状态，直接返回（不计算Lyapunov塑形）
        if state_info.get('collision', False) or state_info.get('off_road', False):
            return base_reward, reward_details

        # 3. 构建Lyapunov状态
        current_lyapunov_state = self._build_lyapunov_state(state_info)

        # 4. 计算Lyapunov塑形奖励
        lyapunov_shaping = 0.0
        lyapunov_details = {}

        if self.lyapunov_last_state is not None:
            lyapunov_shaping, lyapunov_details = self.lyapunov.compute_shaping_reward(
                self.lyapunov_last_state,
                current_lyapunov_state,
                gamma=0.99
            )

        # 5. 更新历史状态
        self.lyapunov_last_state = current_lyapunov_state.copy()

        # 6. 添加Lyapunov塑形到总奖励
        total_reward = base_reward + self.lyapunov_weight * lyapunov_shaping

        # 7. 更新reward_details
        reward_details['total'] = total_reward
        reward_details['lyapunov_shaping'] = lyapunov_shaping
        reward_details['lyapunov_weight'] = self.lyapunov_weight
        reward_details['lyapunov_details'] = lyapunov_details

        # 8. 更新components
        if 'components' not in reward_details:
            reward_details['components'] = {}
        reward_details['components']['lyapunov_shaping'] = lyapunov_shaping

        return total_reward, reward_details

    def get_summary(self):
        """获取摘要（扩展父类方法）"""
        summary = super().get_summary()

        # 添加Lyapunov统计
        lyapunov_stats = self.lyapunov.get_stats()
        summary['lyapunov_stats'] = lyapunov_stats
        summary['lyapunov_weight'] = self.lyapunov_weight

        return summary

class CarlaGymEnv(gym.Env):
    def __init__(self, reward_system_type='improved'):
        self.__version__ = "11.0.3-LYAPUNOV"  # ⭐ 新版本号

        # simulation
        self.verbosity = 0
        self.auto_render = False
        self.n_step = 0
        try:
            self.global_route = np.load('road_maps/global_route_town04.npy')
        except IOError:
            self.global_route = None

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

        # 观测模式配置
        self.use_local_obs = getattr(cfg.GYM_ENV, 'USE_LOCAL_OBS', True)
        self.sensor_range = float(getattr(cfg.GYM_ENV, 'SENSOR_RANGE', 50.0))
        self.use_coop_perception = getattr(cfg.GYM_ENV, 'USE_COOP_PERCEPTION', False)
        self.coop_range = float(getattr(cfg.GYM_ENV, 'COOP_RANGE', 150.0))
        self.use_prediction = getattr(cfg.GYM_ENV, 'USE_PREDICTION', False)
        self.prediction_horizon = float(getattr(cfg.GYM_ENV, 'PREDICTION_HORIZON', 3.0))
        self.prediction_dt = float(getattr(cfg.GYM_ENV, 'PREDICTION_DT', 0.5))

        # 默认启用安全层
        self.use_safety_layer = getattr(cfg.GYM_ENV, 'USE_SAFETY_LAYER', True)
        self.min_safe_dist = float(getattr(cfg.GYM_ENV, 'MIN_SAFE_DIST', 10.0))
        self.min_ttc = float(getattr(cfg.GYM_ENV, 'MIN_TTC', 3.0))

        # frenet
        self.f_idx = 0
        self.init_s = None
        self.max_s = int(cfg.CARLA.MAX_S)
        self.track_length = int(cfg.GYM_ENV.TRACK_LENGTH)
        self.look_back = int(cfg.GYM_ENV.LOOK_BACK)
        self.time_step = int(cfg.GYM_ENV.TIME_STEP)
        self.loop_break = int(cfg.GYM_ENV.LOOP_BREAK)
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.lanechange = False
        self.is_first_path = True
        self.lane_change_start_time = None
        self.lane_change_init_d = None

        # ⭐ 新增：换道追踪器
        self.lane_change_tracker = LaneChangeTracker()

        # ⭐ 修改：初始化奖励系统（支持选择）
        reward_config = {
            'target_speed': self.targetSpeed,
            'max_speed': self.maxSpeed,
            'lane_width': self.LANE_WIDTH,
            'track_length': self.track_length,

            # Baseline参数（代码1的参数）
            'w_speed': int(getattr(cfg.RL, 'W_SPEED', 10)),
            'w_r_speed': int(getattr(cfg.RL, 'W_R_SPEED', 10)),
            'min_speed_gain': float(getattr(cfg.RL, 'MIN_SPEED_GAIN', 0.05)),
            'min_speed_loss': float(getattr(cfg.RL, 'MIN_SPEED_LOSS', -0.05)),
            'lane_change_reward': float(getattr(cfg.RL, 'LANE_CHANGE_REWARD', 1.2)),
            'lane_change_penalty': float(getattr(cfg.RL, 'LANE_CHANGE_PENALTY', 0.3)),
            'collision_penalty': int(getattr(cfg.RL, 'COLLISION', -100)),
            'off_the_road_penalty': int(getattr(cfg.RL, 'OFF_THE_ROAD', -50)),

            # ⭐ 添加Lyapunov参数（从cfg读取）
            'LYAPUNOV_OMEGA_N': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_OMEGA_N', 2.0)),
            'LYAPUNOV_ZETA': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_ZETA', 0.7)),
            'LYAPUNOV_TTC_MIN': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_TTC_MIN', 2.5)),
            'LYAPUNOV_D_MIN': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_D_MIN', 5.0)),
            'LYAPUNOV_D_COMFORTABLE': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_D_COMFORTABLE', 20.0)),
            'LYAPUNOV_ALPHA_LATERAL': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_ALPHA_LATERAL', 0.5)),
            'LYAPUNOV_ALPHA_LONGITUDINAL': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_ALPHA_LONGITUDINAL', 0.3)),
            'LYAPUNOV_SIGMA_SAFETY': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_SIGMA_SAFETY', 5.0)),
            'LYAPUNOV_WEIGHT': float(getattr(cfg.GYM_ENV, 'LYAPUNOV_WEIGHT', 0.30)),
            'DISABLE_SIMPLE_LC_SHAPING': getattr(cfg.GYM_ENV, 'DISABLE_SIMPLE_LC_SHAPING', True),
        }

        # ⭐ 根据参数选择奖励系统
        if reward_system_type == 'baseline':
            self.reward_system = BaselineRewardSystem(reward_config)
            print("✅ Using BaselineRewardSystem (Code1 Baseline)")
        elif reward_system_type == 'lyapunov':
            if LYAPUNOV_AVAILABLE:
                self.reward_system = LyapunovRewardSystem(reward_config)
                print("✅ Using LyapunovRewardSystem")
            else:
                print("⚠️  Lyapunov module not available, falling back to ImprovedRewardSystem")
                self.reward_system = ImprovedRewardSystem(reward_config)
        else:  # 'improved' 或其他
            self.reward_system = ImprovedRewardSystem(reward_config)
            print("✅ Using ImprovedRewardSystem")

        # observation and action space
        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.low_state = np.array([[-1 for _ in range(self.look_back)] for _ in range(16)])
            self.high_state = np.array([[1 for _ in range(self.look_back)] for _ in range(16)])
        else:
            self.low_state = np.array(
                [[-1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])
            self.high_state = np.array(
                [[1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.time_step + 1, 9),
                                                dtype=np.float32)
        action_low = np.array([-1])
        action_high = np.array([1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.state = np.zeros_like(self.observation_space.sample())

        # instances
        self.ego = None
        self.ego_los_sensor = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None
        self.acceleration_ = 0
        self.eps_rew = 0
        self.no_leading_steps = 0
        self.episode_steps = 0

        self.actor_enumerated_dict = {}
        self.actor_enumeration = []
        self.side_window = 5

        self.motionPlanner = None
        self.vehicleController = None
        self.IDM = None

        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

        # 换道追踪
        self.was_lane_changing = False
        self.lane_change_start_distance = 0.0

        # ⭐ 新增：保存当前action用于奖励计算
        self.current_action = 0.0

    def seed(self, seed=None):
        """设置随机种子以保证实验可重复性"""
        self.np_random = np.random.RandomState(seed)
        # 设置其他库的种子
        import random
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def get_vehicle_distance(self, vehicle):
        if vehicle is None:
            return float('inf')

        ego_loc = self.ego.get_location()
        vehicle_loc = vehicle.get_location()

        distance = math.sqrt(
            (ego_loc.x - vehicle_loc.x) ** 2 +
            (ego_loc.y - vehicle_loc.y) ** 2
        )

        return distance

    def get_vehicle_ahead(self, ego_s, ego_d, ego_init_d, ego_target_d):
        """获取前方车辆"""
        distance = self.effective_distance_from_vehicle_ahead

        others_s = []
        others_d = []
        actor_indices = []

        max_range = None
        if self.use_local_obs:
            max_range = self.coop_range if self.use_coop_perception else self.sensor_range

        for idx, actor in enumerate(self.traffic_module.actors_batch):
            act_s_hist, act_d = actor['Frenet State']
            act_s = act_s_hist[0] if isinstance(act_s_hist, (list, deque, np.ndarray)) else act_s_hist

            ds = act_s - ego_s
            if max_range is not None and abs(ds) > max_range:
                continue

            others_s.append(act_s)
            others_d.append(act_d)
            actor_indices.append(idx)

        if len(others_s) == 0:
            return None

        others_s = np.array(others_s)
        others_d = np.array(others_d)

        init_lane_d_idx = np.where((abs(others_d - ego_d) < 1.75) *
                                   (abs(others_d - ego_init_d) < 1))[0]

        init_lane_strict_d_idx = np.where((abs(others_d - ego_d) < 0.4) *
                                          (abs(others_d - ego_init_d) < 1))[0]

        target_lane_d_idx = np.where((abs(others_d - ego_d) < 3.3) *
                                     (abs(others_d - ego_target_d) < 1))[0]

        if len(init_lane_d_idx) and len(target_lane_d_idx) == 0:
            return None
        else:
            def sorted_idx(d_idx):
                lane_s = others_s[d_idx]
                s_idx = np.concatenate(
                    (d_idx.reshape(-1, 1), (lane_s - ego_s).reshape(-1, 1)),
                    axis=1
                )
                return s_idx[s_idx[:, 1].argsort()]

            sorted_init_s_idx = sorted_idx(init_lane_d_idx)
            sorted_init_strict_s_idx = sorted_idx(init_lane_strict_d_idx) if len(init_lane_strict_d_idx) else np.empty(
                (0, 2))
            sorted_target_s_idx = sorted_idx(target_lane_d_idx) if len(target_lane_d_idx) else np.empty((0, 2))

            vehicle_ahead_local_idx = None

            if len(sorted_init_s_idx) and any(sorted_init_s_idx[:, 1][sorted_init_s_idx[:, 1] <= 10] > 0):
                vehicle_ahead_local_idx = int(sorted_init_s_idx[:, 0][sorted_init_s_idx[:, 1] > 0][0])
            elif len(sorted_init_strict_s_idx) and any(
                    sorted_init_strict_s_idx[:, 1][sorted_init_strict_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_local_idx = int(sorted_init_strict_s_idx[:, 0][sorted_init_strict_s_idx[:, 1] > 0][0])
            elif len(sorted_target_s_idx) and any(sorted_target_s_idx[:, 1][sorted_target_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_local_idx = int(sorted_target_s_idx[:, 0][sorted_target_s_idx[:, 1] > 0][0])
            else:
                return None

            global_idx = actor_indices[vehicle_ahead_local_idx]
            return self.traffic_module.actors_batch[global_idx]['Actor']

    def enumerate_actors(self):
        """枚举周围车辆"""
        self.actor_enumeration = []
        ego_s = self.actor_enumerated_dict['EGO']['S'][-1]
        ego_d = self.actor_enumerated_dict['EGO']['D'][-1]

        others_s = []
        others_d = []
        others_id = []

        max_range = None
        if self.use_local_obs:
            max_range = self.coop_range if self.use_coop_perception else self.sensor_range

        for actor in self.traffic_module.actors_batch:
            act_s_hist, act_d = actor['Frenet State']
            act_s = act_s_hist[-1] if hasattr(act_s_hist, '__len__') else act_s_hist

            ds = act_s - ego_s

            if max_range is not None and abs(ds) > max_range:
                continue

            others_s.append(act_s)
            others_d.append(act_d)
            others_id.append(actor['Actor'].id)

        if len(others_s) == 0:
            others_s = np.array([])
            others_d = np.array([])
            others_id = np.array([])
        else:
            others_s = np.array(others_s)
            others_d = np.array(others_d)
            others_id = np.array(others_id)

        def append_actor(x_lane_d_idx, actor_names=None):
            x_lane_s = others_s[x_lane_d_idx]
            x_lane_id = others_id[x_lane_d_idx]
            s_idx = np.concatenate(
                (x_lane_d_idx.reshape(-1, 1), (x_lane_s - ego_s).reshape(-1, 1), x_lane_id.reshape(-1, 1)),
                axis=1
            )
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < self.side_window][0])]
                if (any(abs(sorted_s_idx[:, 1][abs(sorted_s_idx[:, 1]) <= self.side_window]) >= -self.side_window))
                else -1
            )

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > self.side_window][0])]
                if (any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > self.side_window))
                else -1
            )

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < -self.side_window][-1])]
                if (any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < -self.side_window))
                else -1
            )

        # ego lane
        same_lane_d_idx = np.where(abs(others_d - ego_d) < 1)[0]
        if len(same_lane_d_idx) == 0:
            self.actor_enumeration.append(-2)
            self.actor_enumeration.append(-2)
        else:
            same_lane_s = others_s[same_lane_d_idx]
            same_lane_id = others_id[same_lane_d_idx]
            same_s_idx = np.concatenate(
                (same_lane_d_idx.reshape(-1, 1), (same_lane_s - ego_s).reshape(-1, 1), same_lane_id.reshape(-1, 1)),
                axis=1
            )
            sorted_same_s_idx = same_s_idx[same_s_idx[:, 1].argsort()]
            self.actor_enumeration.append(
                others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] > 0][0])]
                if (any(sorted_same_s_idx[:, 1] > 0)) else -1
            )
            self.actor_enumeration.append(
                others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] < 0][-1])]
                if (any(sorted_same_s_idx[:, 1] < 0)) else -1
            )

        # left lane
        left_lane_d_idx = np.where(((others_d - ego_d) < -3) * ((others_d - ego_d) > -4))[0]
        if ego_d < -1.75:
            self.actor_enumeration += [-2, -2, -2]
        elif len(left_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]
        else:
            append_actor(left_lane_d_idx)

        # two left lane
        lleft_lane_d_idx = np.where(((others_d - ego_d) < -6.5) * ((others_d - ego_d) > -7.5))[0]
        if ego_d < 1.75:
            self.actor_enumeration += [-2, -2, -2]
        elif len(lleft_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]
        else:
            append_actor(lleft_lane_d_idx)

        # right lane
        right_lane_d_idx = np.where(((others_d - ego_d) > 3) * ((others_d - ego_d) < 4))[0]
        if ego_d > 5.25:
            self.actor_enumeration += [-2, -2, -2]
        elif len(right_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]
        else:
            append_actor(right_lane_d_idx)

        # two right lane
        rright_lane_d_idx = np.where(((others_d - ego_d) > 6.5) * ((others_d - ego_d) < 7.5))[0]
        if ego_d > 1.75:
            self.actor_enumeration += [-2, -2, -2]
        elif len(rright_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]
        else:
            append_actor(rright_lane_d_idx)

        # Fill enumerated actor values
        actor_id_s_d = {}
        norm_s = []
        for actor in self.traffic_module.actors_batch:
            actor_id_s_d[actor['Actor'].id] = actor['Frenet State']

        for i, actor_id in enumerate(self.actor_enumeration):
            if actor_id >= 0:
                actor_norm_s = []
                act_s_hist, act_d = actor_id_s_d[actor_id]
                for act_s, ego_s_hist in zip(list(act_s_hist)[-self.look_back:],
                                             self.actor_enumerated_dict['EGO']['S'][-self.look_back:]):
                    actor_norm_s.append((act_s - ego_s_hist) / self.max_s)
                norm_s.append(actor_norm_s)
            else:
                norm_s.append(actor_id)

        emp_ln_max = 0.03
        emp_ln_min = -0.03
        no_ln_down = -0.03
        no_ln_up = 0.004
        no_ln = 0.001

        if norm_s[0] not in (-1, -2):
            self.actor_enumerated_dict['LEADING'] = {'S': norm_s[0]}
        else:
            self.actor_enumerated_dict['LEADING'] = {'S': [emp_ln_max]}

        if norm_s[1] not in (-1, -2):
            self.actor_enumerated_dict['FOLLOWING'] = {'S': norm_s[1]}
        else:
            self.actor_enumerated_dict['FOLLOWING'] = {'S': [emp_ln_min]}

        if norm_s[2] not in (-1, -2):
            self.actor_enumerated_dict['LEFT'] = {'S': norm_s[2]}
        else:
            self.actor_enumerated_dict['LEFT'] = {'S': [emp_ln_min] if norm_s[2] == -1 else [no_ln]}

        if norm_s[3] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_UP'] = {'S': norm_s[3]}
        else:
            self.actor_enumerated_dict['LEFT_UP'] = {'S': [emp_ln_max] if norm_s[3] == -1 else [no_ln_up]}

        if norm_s[4] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': norm_s[4]}
        else:
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[4] == -1 else [no_ln_down]}

        if norm_s[5] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT'] = {'S': norm_s[5]}
        else:
            self.actor_enumerated_dict['LLEFT'] = {'S': [emp_ln_min] if norm_s[5] == -1 else [no_ln]}

        if norm_s[6] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': norm_s[6]}
        else:
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': [emp_ln_max] if norm_s[6] == -1 else [no_ln_up]}

        if norm_s[7] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': norm_s[7]}
        else:
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[7] == -1 else [no_ln_down]}

        if norm_s[8] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT'] = {'S': norm_s[8]}
        else:
            self.actor_enumerated_dict['RIGHT'] = {'S': [emp_ln_min] if norm_s[8] == -1 else [no_ln]}

        if norm_s[9] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': norm_s[9]}
        else:
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': [emp_ln_max] if norm_s[9] == -1 else [no_ln_up]}

        if norm_s[10] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': norm_s[10]}
        else:
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[10] == -1 else [no_ln_down]}

        if norm_s[11] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT'] = {'S': norm_s[11]}
        else:
            self.actor_enumerated_dict['RRIGHT'] = {'S': [emp_ln_min] if norm_s[11] == -1 else [no_ln]}

        if norm_s[12] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': norm_s[12]}
        else:
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': [emp_ln_max] if norm_s[12] == -1 else [no_ln_up]}

        if norm_s[13] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': norm_s[13]}
        else:
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[13] == -1 else [no_ln_down]}

    def fix_representation(self):
        self.enumerate_actors()

        self.actor_enumerated_dict['EGO']['SPEED'].extend(self.actor_enumerated_dict['EGO']['SPEED'][-1]
                                                     for _ in range(self.look_back - len(self.actor_enumerated_dict['EGO']['NORM_D'])))

        for act_values in self.actor_enumerated_dict.values():
            act_values['S'].extend(act_values['S'][-1] for _ in range(self.look_back - len(act_values['S'])))

        _range = np.arange(-self.look_back, -1, int(np.ceil(self.look_back / self.time_step)), dtype=int)
        _range = np.append(_range, -1)

        lstm_obs = np.concatenate((np.array(self.actor_enumerated_dict['EGO']['SPEED'])[_range],
                                   np.array(self.actor_enumerated_dict['LEADING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['FOLLOWING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_DOWN']['S'])[_range]),
                                  axis=0)

        return lstm_obs.reshape(self.observation_space.shape[1], -1).transpose()

    def step(self, action=None):
        self.n_step += 1
        self.episode_steps += 1

        self.actor_enumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}

        if action is not None:
            if isinstance(action, np.ndarray):
                action = float(action[0])
            action = np.clip(action, -1.0, 1.0)
            self.current_action = action  # ⭐ 保存用于奖励计算

        if self.verbosity >= 1:
            print('ACTION'.ljust(15), '{:+8.6f}'.format(float(action) if action is not None else 0.0))

        if self.is_first_path:
            self.is_first_path = False
            action = 0.0

        # Motion Planner
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        init_speed = speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x,
                     self.ego.get_location().y,
                     speed, acc, psi, temp, self.max_s]

        # 规划路径
        fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(
            ego_state, self.f_idx, df_n=action, Tf=5, Vf_n=-1
        )
        wps_to_go = len(fpath.t) - 3
        self.f_idx = 1

        collision = False
        track_finished = False
        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]

        # 安全层
        if self.use_safety_layer and self.lanechange:
            ego_s_tmp = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]
            ego_d_tmp = fpath.d[self.f_idx]

            vehicle_ahead_target = self.get_vehicle_ahead(ego_s_tmp, ego_d_tmp, ego_init_d, ego_target_d)
            if vehicle_ahead_target is not None:
                dist = self.get_vehicle_distance(vehicle_ahead_target)
                ahead_speed = get_speed(vehicle_ahead_target)
                ego_speed_now = get_speed(self.ego)

                ttc = constant_velocity_predict(dist, ego_speed_now, ahead_speed)

                unsafe = False
                if dist < self.min_safe_dist:
                    unsafe = True
                if ttc < self.min_ttc:
                    unsafe = True

                if unsafe:
                    if self.verbosity >= 1:
                        print(f'[SAFETY] Lane change blocked: dist={dist:.1f}m, TTC={ttc:.1f}s')

                    # ⭐ 记录放弃换道
                    self.lane_change_tracker.abandon_lane_change(
                        time.time(),
                        reason='safety_blocked'
                    )

                    # 取消换道
                    self.lanechange = False
                    fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(
                        ego_state, self.f_idx, df_n=0.0, Tf=5, Vf_n=-1
                    )
                    wps_to_go = len(fpath.t) - 3
                    self.f_idx = 1
                    ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]

        loop_counter = 0

        # ⭐ 新增：检测换道开始
        if self.lanechange and not self.was_lane_changing:
            self.lane_change_start_time = time.time()
            ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]
            ego_d = fpath.d[self.f_idx]

            vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
            vehicle_ahead_info = None
            if vehicle_ahead:
                vehicle_ahead_info = {
                    'exists': True,
                    'distance': self.get_vehicle_distance(vehicle_ahead),
                    'speed': get_speed(vehicle_ahead),
                }

            # 判断换道原因
            reason = 'exploration'
            if vehicle_ahead_info and vehicle_ahead_info['exists']:
                if vehicle_ahead_info['speed'] < init_speed * 0.8:
                    reason = 'avoid_slow_vehicle'
                elif vehicle_ahead_info['distance'] < 20:
                    reason = 'overtake'

            self.lane_change_tracker.start_lane_change(
                time.time(), ego_s, ego_d, ego_target_d,
                vehicle_ahead_info, reason
            )

            self.lane_change_start_distance = ego_s
            self.lane_change_init_d = ego_init_d

        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]
        ego_d = fpath.d[self.f_idx]
        vehicle_ahead_before = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
        vehicle_ahead_before_info = {
            'exists': vehicle_ahead_before is not None,
            'distance': self.get_vehicle_distance(vehicle_ahead_before) if vehicle_ahead_before else float('inf'),
            'speed': get_speed(vehicle_ahead_before) if vehicle_ahead_before else 0.0
        }

        # ⭐ 计算TTC并添加到info
        if vehicle_ahead_before_info['exists']:
            vehicle_ahead_before_info['ttc'] = constant_velocity_predict(
                vehicle_ahead_before_info['distance'],
                init_speed,
                vehicle_ahead_before_info['speed']
            )

        # 控制循环
        while self.f_idx < wps_to_go and (
                elapsed_time(path_start_time) < self.motionPlanner.D_T * 1.5
                or loop_counter < self.loop_break
                or self.lanechange
        ):

            loop_counter += 1
            ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                         math.radians(self.ego.get_transform().rotation.yaw),
                         0, 0, temp, self.max_s]

            self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]
            ego_d = fpath.d[self.f_idx]
            vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)

            # ⭐ 更新换道进度
            if self.lanechange and self.lane_change_tracker.is_lane_changing:
                vehicle_ahead_info_for_tracker = None
                if vehicle_ahead:
                    distance = self.get_vehicle_distance(vehicle_ahead)
                    ahead_speed = get_speed(vehicle_ahead)
                    ttc = constant_velocity_predict(distance, get_speed(self.ego), ahead_speed)
                    vehicle_ahead_info_for_tracker = {
                        'exists': True,
                        'distance': distance,
                        'speed': ahead_speed,
                        'ttc': ttc
                    }

                self.lane_change_tracker.update_lane_change(
                    ego_s, ego_d, vehicle_ahead_info_for_tracker
                )

            cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)

            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)
            self.ego.apply_control(control)

            # Draw Waypoints
            if self.world_module.args.play_mode != 0:
                for i in range(len(fpath.t)):
                    self.world_module.points_to_draw['path wp {}'.format(i)] = [
                        carla.Location(x=fpath.x[i], y=fpath.y[i]),
                        'COLOR_ALUMINIUM_0']
                self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
                self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
                self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

            # Update Carla
            self.module_manager.tick()
            if self.auto_render:
                self.render()

            collision_hist = self.world_module.get_collision_history()

            self.actor_enumerated_dict['EGO']['S'].append(ego_s)
            self.actor_enumerated_dict['EGO']['D'].append(ego_d)
            self.actor_enumerated_dict['EGO']['NORM_S'].append((ego_s - self.init_s) / self.track_length)
            self.actor_enumerated_dict['EGO']['NORM_D'].append(
                round((ego_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2)
            )
            last_speed_loop = get_speed(self.ego)
            self.actor_enumerated_dict['EGO']['SPEED'].append(last_speed_loop / self.maxSpeed)

            if any(collision_hist):
                collision = True
                break

            distance_traveled = ego_s - self.init_s
            if distance_traveled < -5:
                distance_traveled = self.max_s + distance_traveled
            if distance_traveled >= self.track_length:
                track_finished = True
                break

        # ⭐ 判断换道完成
        lane_change_just_completed = False
        lane_change_safe = False
        lane_change_data = None

        if self.was_lane_changing and not self.lanechange:
            lane_change_just_completed = True
            success = not collision
            lane_change_safe = success

            # 完成换道记录
            lane_change_data = self.lane_change_tracker.complete_lane_change(
                time.time(), ego_s, ego_d,
                success=success, collision=collision
            )

            if success:
                self.episode_stats['successful_lane_changes'] += 1

        self.was_lane_changing = self.lanechange

        # RL Observation
        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.state = self.fix_representation()
            if self.verbosity == 2:
                print(3 * '---EPS UPDATE---')
                print(TENSOR_ROW_NAMES[0].ljust(15), '{:+8.6f}'.format(self.state[-1][0]))
                for idx in range(1, self.state.shape[1]):
                    print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))

            leading_val = float(self.state[-1][1])
            if abs(leading_val - 0.03) < 1e-6:
                self.no_leading_steps += 1

        if self.verbosity == 3:
            print(self.state)

        # RL Reward
        last_speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        current_acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)

        vehicle_ahead_after = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
        vehicle_ahead_after_info = {
            'exists': vehicle_ahead_after is not None,
            'distance': self.get_vehicle_distance(vehicle_ahead_after) if vehicle_ahead_after else float('inf'),
            'speed': get_speed(vehicle_ahead_after) if vehicle_ahead_after else 0.0
        }

        # ⭐ 计算TTC
        if vehicle_ahead_after_info['exists']:
            vehicle_ahead_after_info['ttc'] = constant_velocity_predict(
                vehicle_ahead_after_info['distance'],
                last_speed,
                vehicle_ahead_after_info['speed']
            )

        distance_traveled = ego_s - self.init_s
        if distance_traveled < -5:
            distance_traveled = self.max_s + distance_traveled

        time_elapsed = time.time() - path_start_time

        state_info = {
            'current_speed': last_speed,
            'current_acc': current_acc,
            'ego_d': ego_d,
            'target_d': ego_target_d,
            'action': self.current_action,  # ⭐ 添加action
            'vehicle_ahead': vehicle_ahead_after_info,
            'is_lane_changing': self.lanechange,
            'lane_change_just_completed': lane_change_just_completed,
            'lane_change_safe': lane_change_safe,
            'init_speed': init_speed,
            'vehicle_ahead_before': vehicle_ahead_before_info,
            'distance_traveled': distance_traveled,
            'time_elapsed': time_elapsed,
            'dt': self.dt,
            'collision': collision,
            'off_road': off_the_road
        }

        reward, reward_details = self.reward_system.compute_total_reward(state_info)

        # 构造 info
        info = {}
        info['training_step'] = reward_details.get('training_step', 0)
        info['rew_total'] = reward_details.get('total', reward)
        info['rew_base'] = reward_details.get('base', 0.0)

        comps = reward_details.get('components', {})
        info['rew_speed'] = comps.get('speed', 0.0)
        info['rew_safety'] = comps.get('safety', 0.0)
        info['rew_lane_keep'] = comps.get('lane_keeping', 0.0)
        info['rew_lane_change_shaping'] = comps.get('lane_change_shaping', 0.0)  # ⭐ 新增
        info['rew_comfort'] = comps.get('comfort', 0.0)
        info['rew_efficiency'] = comps.get('efficiency', 0.0)
        info['rew_progress'] = comps.get('progress', 0.0)

        info['rew_shaping'] = reward_details.get('shaping', 0.0)
        info['rew_milestone'] = reward_details.get('milestone', 0.0)
        info['rew_improvement'] = reward_details.get('improvement', 0.0)
        milestone_details = reward_details.get('milestone_details', []) or []
        info['milestone_count'] = len(milestone_details)

        stage_name = reward_details.get('stage', 'stage1')
        if stage_name == 'stage1':
            info['stage_id'] = 1
        elif stage_name == 'stage2':
            info['stage_id'] = 2
        elif stage_name == 'stage3':
            info['stage_id'] = 3
        else:
            info['stage_id'] = 0


        info['collision_penalty'] = reward_details.get('collision_penalty', 0.0)
        info['off_road_penalty'] = reward_details.get('off_road_penalty', 0.0)

        ep_stats = reward_details.get('episode_stats', {})
        info['ep_total_rew'] = ep_stats.get('total_reward', 0.0)
        info['ep_speed_viols'] = ep_stats.get('speed_violations', 0)
        info['ep_safety_viols'] = ep_stats.get('safety_violations', 0)
        info['ep_comfort_viols'] = ep_stats.get('comfort_violations', 0)
        info['ep_succ_lane_changes'] = ep_stats.get('successful_lane_changes', 0)
        info['ep_attempted_lane_changes'] = ep_stats.get('attempted_lane_changes', 0)  # ⭐ 新增

        info['ego_speed'] = last_speed
        info['ego_acc'] = current_acc
        info['lead_exists'] = vehicle_ahead_after_info['exists']
        info['lead_distance'] = vehicle_ahead_after_info['distance']
        info['lead_speed'] = vehicle_ahead_after_info['speed']
        info['distance_traveled'] = distance_traveled
        info['lanechange'] = bool(self.lanechange)
        info['lane_change_just_completed'] = lane_change_just_completed
        info['lane_change_safe'] = lane_change_safe
        info['collision'] = bool(collision)
        info['off_road'] = bool(off_the_road)
        info['track_finished'] = bool(track_finished)

        # ⭐ 新增：换道统计信息
        lc_episode_stats = self.lane_change_tracker.get_episode_stats()
        info['lc_attempts'] = lc_episode_stats['attempts']
        info['lc_successes'] = lc_episode_stats['successes']
        info['lc_success_rate'] = lc_episode_stats['success_rate']

        if lane_change_data:
            info['lc_duration'] = lane_change_data.get('duration', 0.0)
            info['lc_distance'] = lane_change_data.get('distance_traveled', 0.0)
            info['lc_min_ttc'] = lane_change_data.get('min_ttc', float('inf'))
            info['lc_min_distance'] = lane_change_data.get('min_distance', float('inf'))

        info['no_leading_steps'] = int(self.no_leading_steps)
        info['episode_steps'] = int(self.episode_steps)
        if self.episode_steps > 0:
            info['no_leading_ratio'] = float(self.no_leading_steps / self.episode_steps)
        else:
            info['no_leading_ratio'] = 0.0

        info['reward_details'] = reward_details

        # 打印
        if self.verbosity >= 1:
            print('=' * 80)
            print(
                f"REWARD: {reward:+8.4f} | Stage: {reward_details.get('stage', 'N/A')} | "
                f"Step: {reward_details.get('training_step', 0)}"
            )

            # ⭐ 打印换道塑形奖励
            if abs(comps.get('lane_change_shaping', 0)) > 0.01:
                print(f"  Lane Change Shaping: {comps.get('lane_change_shaping', 0):+8.4f}")

            if collision or off_the_road or track_finished:
                print('\\n' + '=' * 80)
                print('EPISODE TERMINATION:')
                if collision:
                    print(f"  ⚠️  COLLISION (Penalty: {reward_details.get('collision_penalty', 0):.2f})")
                if off_the_road:
                    print(f"  ⚠️  OFF ROAD (Penalty: {reward_details.get('off_road_penalty', 0):.2f})")
                if track_finished:
                    print(f"  ✅ TRACK FINISHED")
                print(f"  Speed: {last_speed:.2f} m/s")
                print(f"  Distance: {distance_traveled:.2f} m")

                # ⭐ 打印换道统计
                print(f"\\n  Lane Changes:")
                print(f"    Attempts:  {lc_episode_stats['attempts']}")
                print(f"    Successes: {lc_episode_stats['successes']}")
                print(f"    Rate:      {lc_episode_stats['success_rate']*100:.1f}%")
                print('=' * 80)

        if self.verbosity >= 2:
            print('\\nREWARD BREAKDOWN:')
            components = reward_details.get('components', {})
            print(f"  Base Reward:      {reward_details.get('base', 0):+8.4f}")
            print(
                f"    - Speed:        {components.get('speed', 0):+8.4f} "
                f"(w={self.reward_system.weights['speed']:.2f})")
            print(
                f"    - Safety:       {components.get('safety', 0):+8.4f} "
                f"(w={self.reward_system.weights['safety']:.2f})")
            print(
                f"    - Lane Keep:    {components.get('lane_keeping', 0):+8.4f} "
                f"(w={self.reward_system.weights['lane_keeping']:.2f})")

            # ⭐ 显示换道塑形
            if self.reward_system.weights.get('lane_change_shaping', 0) > 0:
                print(
                    f"    - LC Shaping:   {components.get('lane_change_shaping', 0):+8.4f} "
                    f"(w={self.reward_system.weights['lane_change_shaping']:.2f})")

            print(
                f"    - Comfort:      {components.get('comfort', 0):+8.4f} "
                f"(w={self.reward_system.weights['comfort']:.2f})")
            print(
                f"    - Efficiency:   {components.get('efficiency', 0):+8.4f} "
                f"(w={self.reward_system.weights['efficiency']:.2f})")
            print(
                f"    - Progress:     {components.get('progress', 0):+8.4f} "
                f"(w={self.reward_system.weights['progress']:.2f})")

            if abs(reward_details.get('shaping', 0)) > 0.001:
                print(f"  Potential Shaping: {reward_details.get('shaping', 0):+8.4f}")

            if abs(reward_details.get('improvement', 0)) > 0.001:
                print(f"  Improvement:      {reward_details.get('improvement', 0):+8.4f}")

            milestone_reward = reward_details.get('milestone', 0)
            if abs(milestone_reward) > 0.001:
                print(f"  Milestone:        {milestone_reward:+8.4f}")
                milestone_details = reward_details.get('milestone_details', [])
                if milestone_details:
                    print(f"    Achieved: {', '.join(milestone_details)}")

            print(f"\\n  Collision Penalty: {reward_details.get('collision_penalty', 0):+8.2f}")
            print(f"  Off Road Penalty:  {reward_details.get('off_road_penalty', 0):+8.2f}")

            stats = reward_details.get('episode_stats', {})
            print(f"\\nEPISODE STATS:")
            print(f"  Total Reward:     {stats.get('total_reward', 0):+8.4f}")
            print(f"  Speed Violations: {stats.get('speed_violations', 0)}")
            print(f"  Safety Violations:{stats.get('safety_violations', 0)}")
            print(f"  Comfort Violations:{stats.get('comfort_violations', 0)}")
            print(f"  Lane Changes:     {stats.get('successful_lane_changes', 0)}")
            print(f"  LC Attempts:      {stats.get('attempted_lane_changes', 0)}")
            print('=' * 80)

        # Episode Termination
        done = False

        if collision:
            done = True
            self.eps_rew += reward
            if self.verbosity >= 1:
                print('⚠️  Episode terminated due to COLLISION')
            return self.state, reward, done, info

        elif track_finished:
            done = True
            self.eps_rew += reward
            if self.verbosity >= 1:
                print('✅ Episode completed - TRACK FINISHED')
                print(f'Total Episode Reward: {self.eps_rew:+8.4f}')
            return self.state, reward, done, info

        elif off_the_road:
            # ⭐ 关键修改：baseline系统不终止episode（复现代码1行为）
            if isinstance(self.reward_system, BaselineRewardSystem):
                done = False  # 代码1的行为：off_road不终止
                if self.verbosity >= 1:
                    print('⚠️  Off the road (continuing episode - baseline mode)')
            else:
                done = True  # 其他系统：off_road终止
                if self.verbosity >= 1:
                    print('⚠️  Episode terminated - OFF THE ROAD')

            self.eps_rew += reward
            return self.state, reward, done, info



    def reset(self):
        self.no_leading_steps = 0
        self.episode_steps = 0

        self.vehicleController.reset()
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        init_d = self.world_module.init_d
        self.traffic_module.reset(self.init_s, init_d)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0

        self.n_step = 0

        if self.eps_rew != 0 and self.verbosity >= 1:
            print('\\n' + '=' * 80)
            print(f'EPISODE SUMMARY:')
            print(f'  Total Reward: {self.eps_rew:+8.4f}')
            summary = self.reward_system.get_summary()
            stats = summary.get('episode_stats', {})
            print(f'  Speed Violations: {stats.get("speed_violations", 0)}')
            print(f'  Safety Violations: {stats.get("safety_violations", 0)}')
            print(f'  Comfort Violations: {stats.get("comfort_violations", 0)}')
            print(f'  Successful Lane Changes: {stats.get("successful_lane_changes", 0)}')
            print(f'  Attempted Lane Changes: {stats.get("attempted_lane_changes", 0)}')
            print(f'  Training Step: {summary.get("training_step", 0)}')

            # ⭐ 打印换道统计摘要
            lc_stats = self.lane_change_tracker.get_episode_stats()
            if lc_stats['attempts'] > 0:
                print(f'\\n  Lane Change Stats:')
                print(f'    Attempts:  {lc_stats["attempts"]}')
                print(f'    Successes: {lc_stats["successes"]}')
                print(f'    Success Rate: {lc_stats["success_rate"]*100:.1f}%')

            print('=' * 80 + '\\n')

        self.eps_rew = 0
        self.is_first_path = True
        self.was_lane_changing = False
        self.lane_change_start_time = None
        self.lane_change_start_distance = 0.0

        # ⭐ 重置换道追踪器
        self.lane_change_tracker.reset_episode()

        self.reward_system.reset()

        init_norm_d = round((init_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2)
        ego_s_list = [self.init_s for _ in range(self.look_back)]
        ego_d_list = [init_d for _ in range(self.look_back)]

        self.actor_enumerated_dict['EGO'] = {'NORM_S': [0], 'NORM_D': [init_norm_d],
                                             'S': ego_s_list, 'D': ego_d_list, 'SPEED': [0]}

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.state = self.fix_representation()
            if self.verbosity == 2:
                print(3 * '---RESET---')
                print(TENSOR_ROW_NAMES[0].ljust(15), '{:+8.6f}'.format(self.state[-1][0]))
                for idx in range(1, self.state.shape[1]):
                    print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))

        self.ego.set_simulate_physics(enabled=False)
        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)

        return self.state

    def begin_modules(self, args):
        self.verbosity = 2

        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)

        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            np.save('road_maps/global_route_town04', self.global_route)

        self.motionPlanner = MotionPlanner()
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()

        self.ego = self.world_module.hero_actor
        self.ego_los_sensor = self.world_module.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.IDM = IntelligentDriverModel(self.ego)

        self.module_manager.tick()
        self.init_transform = self.ego.get_transform()

        if self.verbosity >= 1:
            print('\\n' + '=' * 80)
            print('⭐ ENHANCED CARLA ENVIRONMENT v11.0.2-SHAPING')
            print('='* 80)

            # ⭐ 显示奖励系统类型
            reward_system_name = self.reward_system.__class__.__name__
            print(f'Reward System: {reward_system_name}')

            summary = self.reward_system.get_summary()
            print(f'Collision Penalty (Initial): {summary["collision_penalty"]:+8.2f}')
            print(f'Off Road Penalty (Initial):  {summary["off_road_penalty"]:+8.2f}')

            # ⭐ 如果是Lyapunov系统，显示额外信息
            if 'lyapunov_weight' in summary:
                print(f'Lyapunov Weight: {summary["lyapunov_weight"]:.2f}')

            print(f'\\nCurrent Weights:')
            for key, value in summary['current_weights'].items():
                print(f'  {key:25s}: {value:.3f}')
            print('\\n✅ Key Improvements:')
            print('  1. Enhanced safety mechanisms')
            print('  2. Lane change incentives')
            print('  3. Reduced negative rewards')
            print('  4. Improved IDM parameters')
            print('  5. Better exploration rewards')
            print('  6. ⭐ Lane change shaping rewards (dense signals)')
            print('  7. ⭐ Detailed lane change statistics tracking')
            print('=' * 80 + '\\n')

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')

        # ⭐ 打印最终换道统计
        if self.verbosity >= 1:
            self.lane_change_tracker.print_summary()

        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()