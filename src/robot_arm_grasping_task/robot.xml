<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="simple_arm">
  <!-- 全局配置（MuJoCo 标准属性） -->
  <option timestep="0.001" integrator="RK4"/>
  <compiler angle="radian" inertiafromgeom="true"/>

  <!-- 物理参数（替代错误的 <global> 标签，MuJoCo v2.3+ 标准写法） -->
  <default>
    <joint armature="0.1" damping="0.1" frictionloss="0.01"/>
    <geom contype="1" conaffinity="1" rgba="0.8 0.6 0.4 1"/>
  </default>

  <!-- 世界体：地面 + 机械臂 + 目标物体 -->
  <worldbody>
    <!-- 地面 -->
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" rgba="0.2 0.2 0.2 1"/>

    <!-- 机械臂基座 -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" rgba="0.1 0.1 0.8 1"/>

      <!-- 关节1（绕Z轴旋转） -->
      <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0.1"/>
      <body name="link1" pos="0 0 0.1">
        <geom name="link1_geom" type="cylinder" size="0.05 0.2" pos="0 0 0.1" rgba="0.1 0.8 0.1 1"/>

        <!-- 关节2（绕Y轴旋转） -->
        <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0.2"/>
        <body name="link2" pos="0 0 0.2">
          <geom name="link2_geom" type="cylinder" size="0.05 0.2" pos="0 0 0.1" rgba="0.1 0.8 0.1 1"/>

          <!-- 关节3（绕Y轴旋转） -->
          <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0.2"/>
          <body name="link3" pos="0 0 0.2">
            <geom name="link3_geom" type="cylinder" size="0.05 0.15" pos="0 0 0.075" rgba="0.1 0.8 0.1 1"/>

            <!-- 末端执行器（夹爪） -->
            <body name="ee" pos="0 0 0.15">
              <site name="ee_site" pos="0 0 0" size="0.01"/>

              <!-- 左夹爪 -->
              <body name="gripper_left" pos="0.02 0 0">
                <joint name="gripper_left_joint" type="slide" axis="1 0 0" pos="0 0 0" range="-0.02 0"/>
                <geom name="gripper_left_geom" type="box" size="0.02 0.01 0.01" pos="0 0 0" rgba="0.8 0.1 0.1 1"/>
              </body>

              <!-- 右夹爪 -->
              <body name="gripper_right" pos="-0.02 0 0">
                <joint name="gripper_right_joint" type="slide" axis="1 0 0" pos="0 0 0" range="0 0.02"/>
                <geom name="gripper_right_geom" type="box" size="0.02 0.01 0.01" pos="0 0 0" rgba="0.8 0.1 0.1 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- 待抓取的目标物体 -->
    <body name="target_object" pos="0.4 0 0.1">
      <geom name="target_geom" type="box" size="0.05 0.05 0.05" pos="0 0 0" rgba="1 0 0 1"/>
      <freejoint name="object_joint"/>
    </body>
  </worldbody>

  <!-- 执行器：控制关节和夹爪 -->
  <actuator>
    <motor name="joint1_motor" joint="joint1" ctrlrange="-3.14 3.14" gear="100"/>
    <motor name="joint2_motor" joint="joint2" ctrlrange="-1.57 1.57" gear="100"/>
    <motor name="joint3_motor" joint="joint3" ctrlrange="-1.57 1.57" gear="100"/>
    <motor name="gripper_left_motor" joint="gripper_left_joint" ctrlrange="-10 10" gear="50"/>
    <motor name="gripper_right_motor" joint="gripper_right_joint" ctrlrange="-10 10" gear="50"/>
  </actuator>

  <!-- 传感器：检测末端力 -->
  <sensor>
    <force name="ee_force" site="ee_site"/>
  </sensor>
</mujoco>