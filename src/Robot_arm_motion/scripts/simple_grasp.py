import pybullet as p
import time
import math

# 初始化pybullet，使用GUI模式并重置仿真
physicsClient = p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)  # 非实时仿真，便于精确控制
print("✅ pybullet仿真环境初始化完成")

# ---------------------- 纯代码创建环境和物体 ----------------------
# 1. 创建地面
ground_shape = p.createCollisionShape(p.GEOM_PLANE)
ground_id = p.createMultiBody(0, ground_shape, basePosition=[0, 0, 0])
p.changeDynamics(ground_id, -1, lateralFriction=0.8)
print("✅ 已创建地面")

# 2. 创建机械臂底座（固定）
base_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
base_id = p.createMultiBody(0, base_shape, basePosition=[0, 0, 0.1])
print(f"✅ 已创建机械臂底座，ID：{base_id}")

# 3. 创建机械臂大臂（可动连杆1）
arm1_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.06, height=0.4)
arm1_id = p.createMultiBody(1.0, arm1_shape, basePosition=[0, 0, 0.3])
p.changeDynamics(arm1_id, -1, lateralFriction=0.5, restitution=0.1)
print(f"✅ 已创建机械臂大臂，ID：{arm1_id}")

# 4. 创建抓取目标立方体
cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
cube_id = p.createMultiBody(0.2, cube_shape, basePosition=[0.4, 0, 0.2])
p.changeDynamics(cube_id, -1, lateralFriction=0.5)
print(f"✅ 已创建抓取目标立方体，ID：{cube_id}")

# ---------------------- 机械臂运动与抓取逻辑（无关节约束版） ----------------------
def calculate_arm_position(angle):
    """根据旋转角度计算大臂的位置和姿态"""
    # 大臂的旋转中心在底座顶部（0,0,0.2）
    center_x, center_y, center_z = 0, 0, 0.2
    # 大臂长度（半高+底座高度）
    arm_length = 0.2  # 大臂半高0.2m
    # 计算大臂中心的新位置（绕Y轴旋转）
    new_x = center_x + arm_length * math.sin(angle)
    new_z = center_z + arm_length * math.cos(angle)
    # 计算大臂的姿态（四元数）
    orientation = p.getQuaternionFromEuler([0, angle, 0])  # 绕Y轴旋转angle弧度
    return (new_x, 0, new_z), orientation

def check_grasp(arm_pos, cube_pos, threshold=0.1):
    """检测是否可以抓取（距离判断）"""
    distance = math.sqrt(
        (arm_pos[0]-cube_pos[0])**2 +
        (arm_pos[1]-cube_pos[1])**2 +
        (arm_pos[2]-cube_pos[2])**2
    )
    return distance < threshold

# 初始化变量
grasped = False  # 是否已抓取立方体
angle = 0.0      # 机械臂旋转角度
angle_speed = 0.02  # 旋转速度（弧度/步）
max_angle = math.pi / 2  # 最大旋转角度（90°）

print("\n🚀 仿真开始，机械臂将开始运动，靠近立方体后自动抓取...")
print("💡 按下Ctrl+C可终止仿真")

# ---------------------- 主仿真循环 ----------------------
try:
    while True:
        # 1. 更新机械臂旋转角度（来回摆动）
        angle += angle_speed
        if abs(angle) > max_angle:
            angle_speed = -angle_speed  # 反向旋转

        # 2. 计算并设置大臂的位置和姿态
        arm_pos, arm_ori = calculate_arm_position(angle)
        p.resetBasePositionAndOrientation(arm1_id, arm_pos, arm_ori)

        # 3. 获取立方体位置，判断是否抓取
        cube_pos, cube_ori = p.getBasePositionAndOrientation(cube_id)
        if not grasped:
            if check_grasp(arm_pos, cube_pos):
                grasped = True
                print(f"\n✅ 已抓取立方体！当前机械臂角度：{math.degrees(angle):.1f}°")
        else:
            # 已抓取：将立方体位置绑定到机械臂末端
            # 机械臂末端位置（大臂顶部）
            end_effector_pos = (
                arm_pos[0] + 0.2 * math.sin(angle),
                0,
                arm_pos[2] + 0.2 * math.cos(angle)
            )
            p.resetBasePositionAndOrientation(cube_id, end_effector_pos, cube_ori)

        # 4. 执行物理仿真步长
        p.stepSimulation()
        time.sleep(1/240)  # 240Hz仿真频率

except KeyboardInterrupt:
    # 断开仿真连接
    p.disconnect()
    print("\n\n🔚 仿真已手动终止，感谢使用！")