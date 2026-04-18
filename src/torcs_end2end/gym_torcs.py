import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time


class TorcsEnv:
    terminal_judge_start = 100  # 如果经过100个时间步仍无进展，则终止回合
    termination_limit_progress = 5  # [km/h]，如果车辆运行速度低于此限制，回合终止
    default_speed = 50  # 默认速度

    initial_reset = True  # 初始重置标记

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision  # 是否启用视觉观测
        self.throttle = throttle  # 是否手动控制油门（False则自动控制）
        self.gear_change = gear_change  # 是否手动控制换挡（False则自动换挡）

        self.initial_run = True  # 初始运行标记

        ##print("launch torcs")
        # 终止已运行的TORCS进程
        os.system('pkill torcs')
        time.sleep(0.5)
        # 根据是否启用视觉模式启动TORCS
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')  # 带视觉模式启动，无燃油、无损伤、无圈速限制
        else:
            os.system('torcs -nofuel -nolaptime &')  # 无燃油、无圈速限制启动
        time.sleep(0.5)
        # 执行自动启动脚本（连接客户端）
        os.system('sh autostart.sh')
        time.sleep(0.5)

        # 定义动作空间
        if throttle is False:
            # 仅转向控制，动作空间为[-1,1]的一维向量
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            # 转向+油门/刹车控制，动作空间为[-1,1]的二维向量
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        # 定义观测空间
        if vision is False:
            # 无视觉模式下的观测空间范围
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            # 有视觉模式下的观测空间范围（增加视觉数据维度）
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        # print("Step")
        # 将智能体动作转换为TORCS实际的动作格式
        client = self.client

        this_action = self.agent_to_torcs(u)

        # 应用动作到TORCS客户端
        action_torcs = client.R.d

        # 转向控制（范围[-1, 1]）
        action_torcs['steer'] = this_action['steer']

        # 由Snakeoil实现的简单自动油门控制
        if self.throttle is False:
            target_speed = self.default_speed  # 设置目标速度
            # 根据当前速度和转向调整油门
            if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
                client.R.d['accel'] += .01  # 加速
            else:
                client.R.d['accel'] -= .01  # 减速

            # 限制最大油门值
            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            # 低速时增加油门
            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1 / (client.S.d['speedX'] + .1)

            # 牵引力控制系统
            if ((client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) -
                    (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2  # 车轮打滑时降低油门
        else:
            # 手动油门/刹车控制
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        # 由Snakeoil实现的自动换挡
        if self.gear_change is True:
            # 手动换挡控制
            action_torcs['gear'] = this_action['gear']
        else:
            # 自动换挡逻辑（根据速度换挡）
            action_torcs['gear'] = 1  # 默认1挡
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2  # 50km/h以上换2挡
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3  # 80km/h以上换3挡
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4  # 110km/h以上换4挡
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5  # 140km/h以上换5挡
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6  # 170km/h以上换6挡

        # 保存上一帧的完整观测数据，用于奖励计算
        obs_pre = copy.deepcopy(client.S.d)

        # 一步动力学更新 #################################
        # 将智能体动作应用到TORCS
        client.respond_to_server()
        # 获取TORCS的响应
        client.get_servers_input()

        # 获取当前TORCS的完整观测数据
        obs = client.S.d

        # 从TORCS原始观测向量构建标准化的观测数据
        self.observation = self.make_observaton(obs)

        # 奖励函数设置 #######################################
        # 方向相关的正向奖励
        track = np.array(obs['track'])  # 赛道边界距离
        trackPos = np.array(obs['trackPos'])  # 赛道位置（偏离中心的距离）
        sp = np.array(obs['speedX'])  # X方向速度（前进速度）
        damage = np.array(obs['damage'])  # 车辆损伤值
        rpm = np.array(obs['rpm'])  # 发动机转速

        # 进度计算：综合考虑前进速度、方向和赛道位置
        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        reward = progress  # 基础奖励为进度值

        # 碰撞检测（损伤值增加说明发生碰撞）
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1  # 碰撞惩罚

        # 惩罚方向盘大幅度动作（鼓励平稳驾驶）
        reward -= abs(u[0]) * 0.3

        # 终止条件判断 #########################
        episode_terminate = False
        # 车辆驶出赛道则终止回合
        if (abs(track.any()) > 1 or abs(trackPos) > 1):
            reward = -200  # 驶出赛道严重惩罚
            episode_terminate = True
            client.R.d['meta'] = True  # 设置重置标记

        # 长时间无进展则终止回合
        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                print("无进度推进")
                episode_terminate = True
                client.R.d['meta'] = True  # 设置重置标记

        # 车辆倒车（角度余弦值小于0）则终止回合
        if np.cos(obs['angle']) < 0:
            episode_terminate = True
            client.R.d['meta'] = True  # 设置重置标记

        # 发送重置信号
        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1  # 时间步计数+1

        # 返回观测、奖励、终止标记和额外信息
        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        # print("Reset")

        self.time_step = 0  # 重置时间步计数

        if self.initial_reset is not True:
            # 设置重置标记并发送到服务器
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## 临时解决方案：每次回合重启TORCS会存在内存泄漏问题！
            if relaunch is True:
                self.reset_torcs()  # 重启TORCS
                print("### TORCS 已重启 ###")

        # 如果你在环境中使用多个赛道，请在此处修改
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # 在vtorcs中打开新的UDP连接
        self.client.MAX_STEPS = np.inf  # 设置客户端最大步数为无穷大

        client = self.client
        client.get_servers_input()  # 从TORCS获取初始输入

        obs = client.S.d  # 从TORCS获取当前完整观测数据
        self.observation = self.make_observaton(obs)  # 构建标准化观测数据

        self.last_u = None  # 重置上一动作

        self.initial_reset = False  # 初始重置标记置False
        return self.get_obs()  # 返回初始观测

    def end(self):
        # 终止TORCS进程
        os.system('pkill torcs')

    def get_obs(self):
        # 获取当前标准化的观测数据
        return self.observation

    def reset_torcs(self):
        # print("relaunch torcs")
        # 终止现有TORCS进程
        os.system('pkill torcs')
        time.sleep(0.5)
        # 重新启动TORCS
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        # 执行自动启动脚本
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        # 将智能体输出的动作转换为TORCS可识别的动作格式
        torcs_action = {'steer': u[0]}  # 转向动作

        if self.throttle is True:  # 启用油门控制时
            torcs_action.update({'accel': u[1]})  # 油门动作
            torcs_action.update({'brake': u[2]})  # 刹车动作

        if self.gear_change is True:  # 启用换挡控制时
            torcs_action.update({'gear': int(u[3])})  # 换挡动作

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        # 将视觉观测向量转换为RGB图像格式
        image_vec = obs_image_vec
        # 分离RGB通道
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)  # 图像尺寸64x64
        # 重塑为二维数组
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        # 组合为RGB图像（uint8格式）
        return np.array([r, g, b], dtype=np.uint8)

    def make_observaton(self, raw_obs):
        # 将TORCS原始观测数据标准化为智能体可用的格式
        if self.vision is False:
            # 无视觉模式下的观测字段
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)  # 定义观测数据结构
            # 标准化各观测值到合理范围
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32) / 300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32) / 300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / 300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32) / 3.1416,  # 角度归一化到[-1,1]
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32) / 10000,  # 转速归一化
                               track=np.array(raw_obs['track'], dtype=np.float32) / 200.,  # 赛道距离归一化
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32) / 1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            # 有视觉模式下的观测字段（增加图像字段）
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)  # 定义观测数据结构

            # 从观测中提取RGB图像
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            # 标准化各观测值（视觉模式）
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32) / 1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)  # 加入RGB图像数据
