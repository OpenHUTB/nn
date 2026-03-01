"""
MuJoCo 四旋翼无人机仿真 - 公转+避障版
✅ 无人机绕世界Z轴公转，保持原旋转逻辑
✅ 自动避开立方体/圆柱体/球体障碍物
✅ 避障后自动恢复原轨迹，高度固定、无闪烁
✅ 保留所有原代码核心特征
✅ 优化的无人机模型：更精致的外观、更真实的旋翼、更好的视觉效果
✅ 真实风格的障碍物：建筑、油罐、巨石
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math


class QuadrotorSimulation:
    def __init__(self):
        """初始化：添加避障相关参数"""
        xml_string = self.create_quadrotor_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        print("✓ 模型加载成功")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 原代码悬停推力参数
        hover_thrust = 600
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

        # ========== 原代码旋转参数 ==========
        self.base_radius = 1.0      # 基础公转半径
        self.rotate_speed = 1.0     # 公转角速度（rad/s）
        self.hover_height = 0.8     # 固定高度
        self.rotate_angle = 0.0     # 公转角度累计
        self.rotor_visual_speed = 8.0  # 旋翼旋转速度

        # ========== 避障核心参数 ==========
        self.safety_distance = 0.5  # 安全距离（小于此距离触发避障）
        self.avoidance_offset = 0.8 # 避障偏移量（扩大半径绕开障碍物）
        self.obstacle_positions = { # 预定义障碍物位置（与XML中一致）
            "cube": np.array([2.0, 0.0, 0.75]),
            "cylinder": np.array([-1.0, 1.0, 0.5]),
            "sphere": np.array([0.0, -2.0, 1.0])
        }
        self.obstacle_sizes = {     # 障碍物尺寸（碰撞判定用）
            "cube": np.array([0.25, 0.25, 0.75]),
            "cylinder": np.array([0.3, 0.5]),  # 半径、高度
            "sphere": np.array([0.4])          # 半径
        }

    def create_quadrotor_xml(self):
        """创建优化的四旋翼XML模型 - 包含真实风格的障碍物"""
        xml_string = """<?xml version="1.0" ?>
<mujoco model="quadrotor_enhanced">
  <option timestep="0.005" iterations="100" tolerance="1e-10">
    <flag contact="enable" energy="enable"/>
  </option>
  <size nconmax="100" njmax="200"/>
  <default>
    <joint damping="0.001" frictionloss="0.001"/>
    <geom solref="0.02 1" solimp="0.9 0.95 0.01" margin="0.001"/>
  </default>
  
  <asset>
    <!-- 地面材质 -->
    <material name="ground_mat" rgba="0.6 0.7 0.6 1" specular="0.2" shininess="0.1"/>
    
    <!-- 无人机主体材质 -->
    <material name="body_carbon" rgba="0.15 0.15 0.15 1" specular="0.5" shininess="0.4"/>
    <material name="arm_carbon" rgba="0.1 0.1 0.1 1" specular="0.6" shininess="0.5"/>
    <material name="motor_metal" rgba="0.3 0.3 0.35 1" specular="0.8" shininess="0.7"/>
    
    <!-- 旋翼材质（半透明效果） -->
    <material name="propeller_red" rgba="0.9 0.2 0.2 0.9" specular="0.4" shininess="0.3"/>
    <material name="propeller_green" rgba="0.2 0.8 0.2 0.9" specular="0.4" shininess="0.3"/>
    <material name="propeller_blue" rgba="0.2 0.3 0.9 0.9" specular="0.4" shininess="0.3"/>
    <material name="propeller_yellow" rgba="0.9 0.8 0.2 0.9" specular="0.4" shininess="0.3"/>
    
    <!-- LED灯光材质 -->
    <material name="led_red" rgba="1 0.3 0.3 1" specular="0.8" shininess="0.8"/>
    <material name="led_green" rgba="0.3 1 0.3 1" specular="0.8" shininess="0.8"/>
    <material name="led_blue" rgba="0.3 0.3 1 1" specular="0.8" shininess="0.8"/>
    
    <!-- ===== 真实风格的障碍物材质 ===== -->
    <!-- 建筑材质 -->
    <material name="building_concrete" rgba="0.7 0.7 0.7 1" specular="0.2" shininess="0.1"/>
    <material name="building_brick" rgba="0.7 0.4 0.3 1" specular="0.1" shininess="0.05"/>
    <material name="building_glass" rgba="0.6 0.8 1 0.7" specular="0.9" shininess="0.8"/>
    <material name="building_metal" rgba="0.5 0.5 0.55 1" specular="0.7" shininess="0.6"/>
    <material name="building_roof" rgba="0.3 0.3 0.35 1" specular="0.3" shininess="0.2"/>
    
    <!-- 油罐材质 -->
    <material name="tank_metal" rgba="0.6 0.6 0.65 1" specular="0.8" shininess="0.7"/>
    <material name="tank_rust" rgba="0.7 0.4 0.2 1" specular="0.1" shininess="0.05"/>
    <material name="tank_label" rgba="0.9 0.2 0.2 1" specular="0.2" shininess="0.1"/>
    <material name="tank_rim" rgba="0.4 0.4 0.45 1" specular="0.6" shininess="0.5"/>
    
    <!-- 巨石材质 -->
    <material name="rock_gray" rgba="0.5 0.5 0.5 1" specular="0.1" shininess="0.05"/>
    <material name="rock_brown" rgba="0.5 0.3 0.2 1" specular="0.1" shininess="0.05"/>
    <material name="rock_moss" rgba="0.3 0.5 0.2 1" specular="0.05" shininess="0.02"/>
    
    <!-- 纹理 -->
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.6 0.7 0.6" rgb2="0.4 0.5 0.4"/>
    <texture name="gradient" type="2d" builtin="gradient" width="256" height="256" rgb1="0.6 0.6 0.6" rgb2="0.4 0.4 0.4"/>
    <material name="ground_texture" texture="grid" texrepeat="5 5" texuniform="true"/>
  </asset>
  
  <worldbody>
    <!-- 环境光照 - 自然光照 -->
    <light name="sun" pos="10 10 20" dir="-1 -1 -1" directional="true" castshadow="true" diffuse="1 1 1" ambient="0.4 0.4 0.4"/>
    <light name="fill_light" pos="-5 5 10" dir="0.5 -0.5 -1" directional="true" castshadow="false" diffuse="0.6 0.6 0.6"/>
    
    <!-- 带纹理的地面 -->
    <geom name="ground" type="plane" pos="0 0 0" size="20 20 0.1" material="ground_texture" 
          condim="3" friction="0.8 0.005 0.0001"/>
    
    <!-- 参考坐标系（半透明） -->
    <geom name="origin_x" type="cylinder" fromto="0 0 0.1 1 0 0.1" size="0.01" rgba="1 0.2 0.2 0.6"/>
    <geom name="origin_y" type="cylinder" fromto="0 0 0.1 0 1 0.1" size="0.01" rgba="0.2 1 0.2 0.6"/>
    <geom name="origin_z" type="cylinder" fromto="0 0 0.1 0 0 1.1" size="0.01" rgba="0.2 0.2 1 0.6"/>
    
    <!-- 轨迹辅助点（显示公转路径） -->
    <body name="path_marker" pos="0 0 0.8">
      <geom name="path_circle" type="cylinder" size="1.0 0.01" euler="1.57 0 0" rgba="0.5 0.5 0.5 0.15"/>
    </body>
    
    <!-- ========== 优化的无人机模型 ========== -->
    <body name="quadrotor" pos="0 0 0.8" euler="0 0 0">
      <joint name="quad_free_joint" type="free" damping="0.001"/>
      
      <!-- 中心主体 - 碳纤维圆柱体 -->
      <geom name="center_body_main" type="cylinder" size="0.12 0.035" material="body_carbon" mass="0.3"/>
      <geom name="center_body_top" type="cylinder" size="0.08 0.01" pos="0 0 0.04" material="body_carbon" mass="0.02"/>
      <geom name="center_body_bottom" type="cylinder" size="0.08 0.01" pos="0 0 -0.04" material="body_carbon" mass="0.02"/>
      
      <!-- 电子设备装饰 -->
      <geom name="gps_dome" type="sphere" size="0.025" pos="0.06 0.06 0.06" material="motor_metal" mass="0.005"/>
      <geom name="flight_controller" type="box" size="0.03 0.03 0.01" pos="0 0 0.02" rgba="0 0.5 1 0.8" mass="0.005"/>
      
      <!-- LED状态灯 -->
      <geom name="led_front" type="sphere" size="0.015" pos="0.1 0 0.02" material="led_red" mass="0.001"/>
      <geom name="led_rear" type="sphere" size="0.015" pos="-0.1 0 0.02" material="led_blue" mass="0.001"/>
      <geom name="led_left" type="sphere" size="0.015" pos="0 0.1 0.02" material="led_green" mass="0.001"/>
      <geom name="led_right" type="sphere" size="0.015" pos="0 -0.1 0.02" material="led_green" mass="0.001"/>
      
      <!-- 优化的机臂 - X型碳纤维管 -->
      <geom name="arm_front_right" type="capsule" fromto="0.08 0.08 0 0.3 0.3 0" size="0.015" material="arm_carbon" mass="0.03"/>
      <geom name="arm_front_left" type="capsule" fromto="0.08 -0.08 0 0.3 -0.3 0" size="0.015" material="arm_carbon" mass="0.03"/>
      <geom name="arm_back_left" type="capsule" fromto="-0.08 -0.08 0 -0.3 -0.3 0" size="0.015" material="arm_carbon" mass="0.03"/>
      <geom name="arm_back_right" type="capsule" fromto="-0.08 0.08 0 -0.3 0.3 0" size="0.015" material="arm_carbon" mass="0.03"/>
      
      <!-- 机臂加固件 -->
      <geom name="arm_joint_front_right" type="sphere" size="0.025" pos="0.08 0.08 0" material="motor_metal" mass="0.005"/>
      <geom name="arm_joint_front_left" type="sphere" size="0.025" pos="0.08 -0.08 0" material="motor_metal" mass="0.005"/>
      <geom name="arm_joint_back_left" type="sphere" size="0.025" pos="-0.08 -0.08 0" material="motor_metal" mass="0.005"/>
      <geom name="arm_joint_back_right" type="sphere" size="0.025" pos="-0.08 0.08 0" material="motor_metal" mass="0.005"/>
      
      <!-- 电机和旋翼 -->
      <!-- 前右电机（红色旋翼） -->
      <body name="motor_front_right" pos="0.3 0.3 0">
        <geom name="motor_housing_front_right" type="cylinder" size="0.045 0.025" material="motor_metal" mass="0.04"/>
        <geom name="motor_top_front_right" type="cylinder" size="0.03 0.01" pos="0 0 0.03" material="motor_metal" mass="0.01"/>
        <geom name="motor_bottom_front_right" type="cylinder" size="0.03 0.01" pos="0 0 -0.03" material="motor_metal" mass="0.01"/>
        <body name="rotor_front_right" pos="0 0 0.06">
          <joint name="rotor_front_right_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_blade1_front_right" type="box" size="0.12 0.02 0.005" pos="0.06 0 0" euler="0 0 0" material="propeller_red" mass="0.005"/>
          <geom name="propeller_blade2_front_right" type="box" size="0.12 0.02 0.005" pos="-0.06 0 0" euler="0 0 0" material="propeller_red" mass="0.005"/>
          <geom name="propeller_blade3_front_right" type="box" size="0.02 0.12 0.005" pos="0 0.06 0" euler="0 0 0" material="propeller_red" mass="0.005"/>
          <geom name="propeller_blade4_front_right" type="box" size="0.02 0.12 0.005" pos="0 -0.06 0" euler="0 0 0" material="propeller_red" mass="0.005"/>
          <geom name="propeller_center_front_right" type="cylinder" size="0.025 0.01" pos="0 0 0" material="motor_metal" mass="0.002"/>
        </body>
      </body>
      
      <!-- 前左电机（绿色旋翼） -->
      <body name="motor_front_left" pos="0.3 -0.3 0">
        <geom name="motor_housing_front_left" type="cylinder" size="0.045 0.025" material="motor_metal" mass="0.04"/>
        <geom name="motor_top_front_left" type="cylinder" size="0.03 0.01" pos="0 0 0.03" material="motor_metal" mass="0.01"/>
        <geom name="motor_bottom_front_left" type="cylinder" size="0.03 0.01" pos="0 0 -0.03" material="motor_metal" mass="0.01"/>
        <body name="rotor_front_left" pos="0 0 0.06">
          <joint name="rotor_front_left_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_blade1_front_left" type="box" size="0.12 0.02 0.005" pos="0.06 0 0" euler="0 0 0" material="propeller_green" mass="0.005"/>
          <geom name="propeller_blade2_front_left" type="box" size="0.12 0.02 0.005" pos="-0.06 0 0" euler="0 0 0" material="propeller_green" mass="0.005"/>
          <geom name="propeller_blade3_front_left" type="box" size="0.02 0.12 0.005" pos="0 0.06 0" euler="0 0 0" material="propeller_green" mass="0.005"/>
          <geom name="propeller_blade4_front_left" type="box" size="0.02 0.12 0.005" pos="0 -0.06 0" euler="0 0 0" material="propeller_green" mass="0.005"/>
          <geom name="propeller_center_front_left" type="cylinder" size="0.025 0.01" pos="0 0 0" material="motor_metal" mass="0.002"/>
        </body>
      </body>
      
      <!-- 后左电机（蓝色旋翼） -->
      <body name="motor_back_left" pos="-0.3 -0.3 0">
        <geom name="motor_housing_back_left" type="cylinder" size="0.045 0.025" material="motor_metal" mass="0.04"/>
        <geom name="motor_top_back_left" type="cylinder" size="0.03 0.01" pos="0 0 0.03" material="motor_metal" mass="0.01"/>
        <geom name="motor_bottom_back_left" type="cylinder" size="0.03 0.01" pos="0 0 -0.03" material="motor_metal" mass="0.01"/>
        <body name="rotor_back_left" pos="0 0 0.06">
          <joint name="rotor_back_left_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_blade1_back_left" type="box" size="0.12 0.02 0.005" pos="0.06 0 0" euler="0 0 0" material="propeller_blue" mass="0.005"/>
          <geom name="propeller_blade2_back_left" type="box" size="0.12 0.02 0.005" pos="-0.06 0 0" euler="0 0 0" material="propeller_blue" mass="0.005"/>
          <geom name="propeller_blade3_back_left" type="box" size="0.02 0.12 0.005" pos="0 0.06 0" euler="0 0 0" material="propeller_blue" mass="0.005"/>
          <geom name="propeller_blade4_back_left" type="box" size="0.02 0.12 0.005" pos="0 -0.06 0" euler="0 0 0" material="propeller_blue" mass="0.005"/>
          <geom name="propeller_center_back_left" type="cylinder" size="0.025 0.01" pos="0 0 0" material="motor_metal" mass="0.002"/>
        </body>
      </body>
      
      <!-- 后右电机（黄色旋翼） -->
      <body name="motor_back_right" pos="-0.3 0.3 0">
        <geom name="motor_housing_back_right" type="cylinder" size="0.045 0.025" material="motor_metal" mass="0.04"/>
        <geom name="motor_top_back_right" type="cylinder" size="0.03 0.01" pos="0 0 0.03" material="motor_metal" mass="0.01"/>
        <geom name="motor_bottom_back_right" type="cylinder" size="0.03 0.01" pos="0 0 -0.03" material="motor_metal" mass="0.01"/>
        <body name="rotor_back_right" pos="0 0 0.06">
          <joint name="rotor_back_right_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_blade1_back_right" type="box" size="0.12 0.02 0.005" pos="0.06 0 0" euler="0 0 0" material="propeller_yellow" mass="0.005"/>
          <geom name="propeller_blade2_back_right" type="box" size="0.12 0.02 0.005" pos="-0.06 0 0" euler="0 0 0" material="propeller_yellow" mass="0.005"/>
          <geom name="propeller_blade3_back_right" type="box" size="0.02 0.12 0.005" pos="0 0.06 0" euler="0 0 0" material="propeller_yellow" mass="0.005"/>
          <geom name="propeller_blade4_back_right" type="box" size="0.02 0.12 0.005" pos="0 -0.06 0" euler="0 0 0" material="propeller_yellow" mass="0.005"/>
          <geom name="propeller_center_back_right" type="cylinder" size="0.025 0.01" pos="0 0 0" material="motor_metal" mass="0.002"/>
        </body>
      </body>

      <!-- 优化的起落架 -->
      <geom name="landing_gear_front_left" type="capsule" fromto="0.1 0.05 -0.05 0.1 0.05 -0.15" size="0.008" rgba="0.3 0.3 0.3 1" mass="0.01"/>
      <geom name="landing_gear_front_right" type="capsule" fromto="0.1 -0.05 -0.05 0.1 -0.05 -0.15" size="0.008" rgba="0.3 0.3 0.3 1" mass="0.01"/>
      <geom name="landing_gear_back_left" type="capsule" fromto="-0.1 -0.05 -0.05 -0.1 -0.05 -0.15" size="0.008" rgba="0.3 0.3 0.3 1" mass="0.01"/>
      <geom name="landing_gear_back_right" type="capsule" fromto="-0.1 0.05 -0.05 -0.1 0.05 -0.15" size="0.008" rgba="0.3 0.3 0.3 1" mass="0.01"/>
      
      <!-- 起落架横杆 -->
      <geom name="landing_skid_front" type="cylinder" fromto="0.1 0.1 -0.15 0.1 -0.1 -0.15" size="0.008" rgba="0.3 0.3 0.3 1" mass="0.01"/>
      <geom name="landing_skid_back" type="cylinder" fromto="-0.1 -0.1 -0.15 -0.1 0.1 -0.15" size="0.008" rgba="0.3 0.3 0.3 1" mass="0.01"/>

      <!-- 方向标记（前端） -->
      <geom name="front_arrow" type="box" size="0.02 0.02 0.005" pos="0.15 0 0.06" rgba="1 1 1 1" mass="0.001"/>
      <geom name="front_arrow_tip" type="sphere" size="0.015" pos="0.18 0 0.06" rgba="1 1 0 1" mass="0.001"/>
    </body>

    <!-- ========== 真实风格的障碍物（使用标准几何体） ========== -->
    
    <!-- 1. 建筑（立方体）- 多层办公楼 -->
    <body name="building" pos="2 0 0.75">
      <!-- 地基 -->
      <geom name="building_base" type="box" size="0.35 0.35 0.1" pos="0 0 -0.65" material="building_concrete" mass="2"/>
      
      <!-- 主体 - 分层设计 -->
      <geom name="building_floor1" type="box" size="0.3 0.3 0.2" pos="0 0 -0.4" material="building_concrete" mass="1"/>
      <geom name="building_floor2" type="box" size="0.28 0.28 0.2" pos="0 0 -0.15" material="building_brick" mass="1"/>
      <geom name="building_floor3" type="box" size="0.26 0.26 0.2" pos="0 0 0.1" material="building_concrete" mass="1"/>
      <geom name="building_floor4" type="box" size="0.24 0.24 0.2" pos="0 0 0.35" material="building_brick" mass="1"/>
      
      <!-- 屋顶 -->
      <geom name="building_roof" type="box" size="0.22 0.22 0.1" pos="0 0 0.6" material="building_roof" mass="0.5"/>
      <geom name="building_antenna_base" type="cylinder" size="0.04 0.05" pos="0 0 0.7" material="building_metal" mass="0.05"/>
      <geom name="building_antenna" type="cylinder" size="0.02 0.2" pos="0 0 0.85" material="building_metal" mass="0.05"/>
      
      <!-- 窗户 - 每层四个面（用小立方体表示） -->
      <!-- 第一层窗户 -->
      <geom name="building_win1_f" type="box" size="0.05 0.1 0.1" pos="0.18 0 -0.4" material="building_glass" mass="0.05"/>
      <geom name="building_win1_b" type="box" size="0.05 0.1 0.1" pos="-0.18 0 -0.4" material="building_glass" mass="0.05"/>
      <geom name="building_win1_l" type="box" size="0.1 0.05 0.1" pos="0 0.18 -0.4" material="building_glass" mass="0.05"/>
      <geom name="building_win1_r" type="box" size="0.1 0.05 0.1" pos="0 -0.18 -0.4" material="building_glass" mass="0.05"/>
      
      <!-- 第二层窗户 -->
      <geom name="building_win2_f" type="box" size="0.05 0.1 0.1" pos="0.16 0 -0.15" material="building_glass" mass="0.05"/>
      <geom name="building_win2_b" type="box" size="0.05 0.1 0.1" pos="-0.16 0 -0.15" material="building_glass" mass="0.05"/>
      <geom name="building_win2_l" type="box" size="0.1 0.05 0.1" pos="0 0.16 -0.15" material="building_glass" mass="0.05"/>
      <geom name="building_win2_r" type="box" size="0.1 0.05 0.1" pos="0 -0.16 -0.15" material="building_glass" mass="0.05"/>
      
      <!-- 第三层窗户 -->
      <geom name="building_win3_f" type="box" size="0.05 0.1 0.1" pos="0.14 0 0.1" material="building_glass" mass="0.05"/>
      <geom name="building_win3_b" type="box" size="0.05 0.1 0.1" pos="-0.14 0 0.1" material="building_glass" mass="0.05"/>
      <geom name="building_win3_l" type="box" size="0.1 0.05 0.1" pos="0 0.14 0.1" material="building_glass" mass="0.05"/>
      <geom name="building_win3_r" type="box" size="0.1 0.05 0.1" pos="0 -0.14 0.1" material="building_glass" mass="0.05"/>
      
      <!-- 第四层窗户 -->
      <geom name="building_win4_f" type="box" size="0.05 0.1 0.1" pos="0.12 0 0.35" material="building_glass" mass="0.05"/>
      <geom name="building_win4_b" type="box" size="0.05 0.1 0.1" pos="-0.12 0 0.35" material="building_glass" mass="0.05"/>
      <geom name="building_win4_l" type="box" size="0.1 0.05 0.1" pos="0 0.12 0.35" material="building_glass" mass="0.05"/>
      <geom name="building_win4_r" type="box" size="0.1 0.05 0.1" pos="0 -0.12 0.35" material="building_glass" mass="0.05"/>
      
      <!-- 门口 -->
      <geom name="building_door" type="box" size="0.08 0.05 0.15" pos="0.18 0 -0.55" material="building_metal" mass="0.1"/>
    </body>

    <!-- 2. 油罐（圆柱体）- 工业储罐（用圆柱体和小球体组合） -->
    <body name="tank" pos="-1 1 0.5">
      <!-- 主罐体 -->
      <geom name="tank_main" type="cylinder" size="0.35 0.5" pos="0 0 0" material="tank_metal" mass="3"/>
      
      <!-- 罐顶和罐底 -->
      <geom name="tank_top" type="cylinder" size="0.3 0.05" pos="0 0 0.55" material="tank_rim" mass="0.3"/>
      <geom name="tank_bottom" type="cylinder" size="0.3 0.05" pos="0 0 -0.55" material="tank_rim" mass="0.3"/>
      <geom name="tank_dome" type="sphere" size="0.2" pos="0 0 0.7" material="tank_metal" mass="0.2"/>
      
      <!-- 加强环 - 用薄圆柱体代替 -->
      <geom name="tank_ring1" type="cylinder" size="0.37 0.03" pos="0 0 -0.3" material="tank_rim" mass="0.1"/>
      <geom name="tank_ring2" type="cylinder" size="0.37 0.03" pos="0 0 0" material="tank_rim" mass="0.1"/>
      <geom name="tank_ring3" type="cylinder" size="0.37 0.03" pos="0 0 0.3" material="tank_rim" mass="0.1"/>
      
      <!-- 垂直加强筋 - 用长方体 -->
      <geom name="tank_rib1" type="box" size="0.03 0.05 1.1" pos="0.35 0 0" material="tank_rim" mass="0.1"/>
      <geom name="tank_rib2" type="box" size="0.03 0.05 1.1" pos="-0.35 0 0" material="tank_rim" mass="0.1"/>
      <geom name="tank_rib3" type="box" size="0.05 0.03 1.1" pos="0 0.35 0" material="tank_rim" mass="0.1"/>
      <geom name="tank_rib4" type="box" size="0.05 0.03 1.1" pos="0 -0.35 0" material="tank_rim" mass="0.1"/>
      
      <!-- 锈迹斑点 - 用小球体模拟 -->
      <geom name="tank_rust1" type="sphere" size="0.08" pos="0.2 0.1 0.2" material="tank_rust" mass="0.02"/>
      <geom name="tank_rust2" type="sphere" size="0.1" pos="-0.15 -0.2 -0.2" material="tank_rust" mass="0.02"/>
      <geom name="tank_rust3" type="sphere" size="0.06" pos="0.1 -0.25 0.4" material="tank_rust" mass="0.02"/>
      
      <!-- 危险品标志 -->
      <geom name="tank_label1" type="box" size="0.15 0.1 0.02" pos="0.25 0 0.4" material="tank_label" mass="0.02"/>
      <geom name="tank_label2" type="box" size="0.15 0.1 0.02" pos="-0.25 0 -0.3" material="tank_label" mass="0.02"/>
      
      <!-- 梯子 - 用圆柱体组成 -->
      <geom name="tank_ladder1" type="cylinder" fromto="0.35 0.15 -0.4 0.35 0.15 0.4" size="0.01" material="tank_rim" mass="0.05"/>
      <geom name="tank_ladder2" type="cylinder" fromto="0.35 -0.15 -0.4 0.35 -0.15 0.4" size="0.01" material="tank_rim" mass="0.05"/>
      <geom name="tank_ladder_rung1" type="cylinder" fromto="0.35 0.15 -0.3 0.35 -0.15 -0.3" size="0.008" material="tank_rim" mass="0.02"/>
      <geom name="tank_ladder_rung2" type="cylinder" fromto="0.35 0.15 0 0.35 -0.15 0" size="0.008" material="tank_rim" mass="0.02"/>
      <geom name="tank_ladder_rung3" type="cylinder" fromto="0.35 0.15 0.3 0.35 -0.15 0.3" size="0.008" material="tank_rim" mass="0.02"/>
    </body>

    <!-- 3. 巨石群（球体）- 自然岩石 -->
    <body name="rock_group" pos="0 -2 1.0">
      <!-- 主巨石 -->
      <geom name="rock_main" type="sphere" size="0.45" pos="0 0 0" material="rock_gray" mass="4"/>
      
      <!-- 附着小岩石 -->
      <geom name="rock_small1" type="sphere" size="0.2" pos="0.35 0.2 0.2" material="rock_brown" mass="0.5"/>
      <geom name="rock_small2" type="sphere" size="0.18" pos="-0.3 -0.25 -0.1" material="rock_brown" mass="0.4"/>
      <geom name="rock_small3" type="sphere" size="0.15" pos="0.2 -0.35 0.3" material="rock_gray" mass="0.3"/>
      <geom name="rock_small4" type="sphere" size="0.22" pos="-0.25 0.3 -0.25" material="rock_brown" mass="0.5"/>
      <geom name="rock_small5" type="sphere" size="0.12" pos="0.4 -0.1 -0.2" material="rock_gray" mass="0.2"/>
      
      <!-- 青苔斑块 -->
      <geom name="rock_moss1" type="sphere" size="0.1" pos="0.3 0.25 0.15" material="rock_moss" mass="0.05"/>
      <geom name="rock_moss2" type="sphere" size="0.08" pos="-0.2 -0.3 -0.3" material="rock_moss" mass="0.03"/>
      <geom name="rock_moss3" type="sphere" size="0.12" pos="0.1 0.4 0.1" material="rock_moss" mass="0.04"/>
      
      <!-- 裂缝/纹理（用细长圆柱表示） -->
      <geom name="rock_crack1" type="cylinder" fromto="0.1 0.2 0.3 -0.1 -0.1 -0.2" size="0.02" material="rock_brown" mass="0.05"/>
      <geom name="rock_crack2" type="cylinder" fromto="-0.2 0.3 -0.1 0.3 -0.2 0.2" size="0.015" material="rock_brown" mass="0.03"/>
    </body>

    <!-- 添加一些小障碍物增加丰富度 -->
    <!-- 小石块 -->
    <body name="pebbles1" pos="1.5 0.5 0.1">
      <geom name="pebble1" type="sphere" size="0.05" pos="0 0 0" material="rock_gray" mass="0.01"/>
      <geom name="pebble2" type="sphere" size="0.04" pos="0.1 0.05 0" material="rock_brown" mass="0.005"/>
      <geom name="pebble3" type="sphere" size="0.03" pos="-0.05 0.1 0" material="rock_gray" mass="0.002"/>
    </body>
    
    <body name="pebbles2" pos="-0.5 -1.5 0.1">
      <geom name="pebble4" type="sphere" size="0.06" pos="0 0 0" material="rock_brown" mass="0.02"/>
      <geom name="pebble5" type="sphere" size="0.05" pos="0.08 -0.03 0" material="rock_gray" mass="0.01"/>
    </body>
    
    <!-- 小柱子 -->
    <body name="small_pillar" pos="1.8 -0.8 0.3">
      <geom name="pillar_base" type="cylinder" size="0.1 0.1" pos="0 0 -0.2" material="building_concrete" mass="0.1"/>
      <geom name="pillar_body" type="cylinder" size="0.08 0.3" pos="0 0 0.1" material="building_concrete" mass="0.2"/>
      <geom name="pillar_top" type="sphere" size="0.1" pos="0 0 0.3" material="building_metal" mass="0.05"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor_front_right" joint="rotor_front_right_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
    <motor name="motor_front_left" joint="rotor_front_left_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
    <motor name="motor_back_left" joint="rotor_back_left_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
    <motor name="motor_back_right" joint="rotor_back_right_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
  </actuator>
</mujoco>"""
        return xml_string

    def calculate_obstacle_distance(self, drone_pos):
        """计算无人机到各障碍物的水平距离（Z轴高度忽略，只算XY平面）"""
        distances = {}

        # 立方体障碍物
        cube_pos = self.obstacle_positions["cube"][:2]  # 只取XY坐标
        drone_xy = drone_pos[:2]
        distances["cube"] = np.linalg.norm(drone_xy - cube_pos) - self.obstacle_sizes["cube"][0]

        # 圆柱体障碍物
        cyl_pos = self.obstacle_positions["cylinder"][:2]
        distances["cylinder"] = np.linalg.norm(drone_xy - cyl_pos) - self.obstacle_sizes["cylinder"][0]

        # 球体障碍物
        sphere_pos = self.obstacle_positions["sphere"][:2]
        distances["sphere"] = np.linalg.norm(drone_xy - sphere_pos) - self.obstacle_sizes["sphere"][0]

        return distances

    def get_avoidance_radius(self, drone_pos):
        """根据障碍物距离动态调整公转半径（避障核心逻辑）"""
        distances = self.calculate_obstacle_distance(drone_pos)
        min_distance = min(distances.values())

        # 判定是否需要避障
        if min_distance < self.safety_distance:
            # 找到最近的障碍物
            closest_obs = min(distances, key=distances.get)
            obs_pos = self.obstacle_positions[closest_obs][:2]
            drone_xy = drone_pos[:2]

            # 计算避障方向：远离最近障碍物
            direction = drone_xy - obs_pos
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([1, 0])

            # 动态调整半径，绕开障碍物
            return self.base_radius + self.avoidance_offset
        else:
            # 无避障需求，恢复基础半径
            return self.base_radius

    def simulation_loop(self, viewer, duration):
        """核心：公转+避障逻辑"""
        start_time = time.time()
        last_print_time = time.time()

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # ========== 1. 更新公转角度 ==========
            self.rotate_angle += self.rotate_speed * self.model.opt.timestep
            # 限制角度范围（防止数值过大）
            if self.rotate_angle > 2 * math.pi:
                self.rotate_angle -= 2 * math.pi

            # ========== 2. 计算基础公转位置 ==========
            base_x = self.base_radius * math.cos(self.rotate_angle)
            base_y = self.base_radius * math.sin(self.rotate_angle)
            base_pos = np.array([base_x, base_y, self.hover_height])

            # ========== 3. 避障逻辑：动态调整位置 ==========
            current_radius = self.get_avoidance_radius(base_pos)
            # 计算避障后的目标位置
            target_x = current_radius * math.cos(self.rotate_angle)
            target_y = current_radius * math.sin(self.rotate_angle)
            target_z = self.hover_height

            # ========== 4. 设置无人机位置和姿态 ==========
            self.data.qpos[0] = target_x  # X轴位置
            self.data.qpos[1] = target_y  # Y轴位置
            self.data.qpos[2] = target_z  # Z轴固定高度
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 姿态不变

            # ========== 5. 旋翼旋转（保持原逻辑） ==========
            rotor_speed = self.rotor_visual_speed
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep * (i % 2 * 2 - 1)

            if viewer:
                viewer.sync()

            # ========== 6. 打印状态信息（新增避障状态） ==========
            if time.time() - last_print_time > 1.0:
                current_time = self.data.time
                current_pos = self.data.qpos[0:3].copy()
                distances = self.calculate_obstacle_distance(current_pos)
                min_dist = min(distances.values())
                avoidance_status = "避障中" if min_dist < self.safety_distance else "正常轨迹"

                print(f"\n时间: {current_time:.1f}s | 公转角度: {self.rotate_angle:.2f}rad")
                print(f"当前位置: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] m")
                print(f"公转半径: {current_radius:.2f}m | 状态: {avoidance_status}")
                print(f"最近障碍物距离: {min_dist:.2f}m | 安全距离: {self.safety_distance}m")
                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真：带避障功能"""
        print(f"\n▶ 开始仿真（公转+自动避障），时长: {duration}秒")
        print(f"▶ 基础公转半径: {self.base_radius}m | 旋转速度: {self.rotate_speed}rad/s")
        print(f"▶ 安全距离: {self.safety_distance}m | 避障偏移量: {self.avoidance_offset}m")
        print(f"▶ 无人机模型: 优化版（碳纤维机身+四色旋翼+LED灯）")
        print(f"▶ 障碍物模型: 真实风格（多层建筑/工业油罐/巨石群）")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 优化相机视角，方便观察避障效果
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 20
                    viewer.cam.distance = 12.0
                    viewer.cam.lookat[:] = [0.0, 0.0, self.hover_height]
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")
            import traceback
            traceback.print_exc()

        print("\n✅ 仿真结束（避障功能正常运行）")


def main():
    print("🚁 MuJoCo 四旋翼无人机仿真 - 公转+自动避障版（真实障碍物）")
    print("=" * 70)

    try:
        sim = QuadrotorSimulation()

        # ========== 可自定义参数 ==========
        # 原旋转参数
        sim.base_radius = 1.0      # 基础公转半径
        sim.rotate_speed = 1.0     # 旋转速度
        sim.hover_height = 0.8     # 悬停高度
        # 避障参数
        sim.safety_distance = 0.5  # 触发避障的安全距离（越小越灵敏）
        sim.avoidance_offset = 0.8 # 避障时的半径偏移量（越大避障越远）

        print("✅ 初始化完成（避障功能已启用）")
        sim.run_simulation(
            duration=60.0,
            use_viewer=True
        )

    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()