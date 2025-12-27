import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端以避免兼容性问题
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ArmController:
    def __init__(self, model_path):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 获取关节ID
        self.joint_names = [f"joint{i}" for i in range(1, 7)]  # joint1 到 joint6
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        
        # 获取夹爪关节ID
        self.gripper_names = [
            "finger1_slide", "finger1_hinge1", "finger1_hinge2",
            "finger2_slide", "finger2_hinge1", "finger2_hinge2", 
            "finger3_slide", "finger3_hinge1", "finger3_hinge2"
        ]
        self.gripper_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                           for name in self.gripper_names]
        
        # 获取夹爪末端执行器位置
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_center")
        
        # 获取测试物体
        self.object_names = ["test_sphere", "test_box", "test_cylinder", "test_capsule", "test_ellipsoid"]
        self.object_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) 
                          for name in self.object_names]
        
    def move_to_target(self, target_pos, duration=5.0):
        """
        移动到目标位置（笛卡尔空间）
        """
        # 获取当前末端执行器位置
        current_pos = self.data.site_xpos[self.gripper_site_id].copy()
        
        # 生成线性轨迹
        steps = int(duration / self.model.opt.timestep)
        trajectory = np.linspace(current_pos, target_pos, steps)
        
        return trajectory
    
    def generate_joint_trajectory(self, target_angles, duration=5.0):
        """
        生成关节空间轨迹
        """
        # 获取当前关节角度
        current_angles = self.data.qpos[:6].copy()
        
        # 生成线性轨迹
        steps = int(duration / self.model.opt.timestep)
        trajectory = np.linspace(current_angles, target_angles, steps)
        
        return trajectory
    
    def set_gripper(self, open_close=0.0):
        """
        控制夹爪开合
        open_close: -1.0 (完全张开) 到 1.0 (完全闭合)
        """
        # 夹爪范围映射
        slide_range = self.model.jnt_range[self.gripper_ids[0]]  # 所有滑动关节使用相同范围
        hinge1_range = self.model.jnt_range[self.gripper_ids[1]]  # 所有第一铰链关节使用相同范围
        hinge2_range = self.model.jnt_range[self.gripper_ids[2]]  # 所有第二铰链关节使用相同范围
        
        # 计算目标位置
        slide_target = slide_range[0] + (open_close + 1.0) / 2.0 * (slide_range[1] - slide_range[0])
        hinge1_target = hinge1_range[0] + (open_close + 1.0) / 2.0 * (hinge1_range[1] - hinge1_range[0])
        hinge2_target = hinge2_range[0] + (open_close + 1.0) / 2.0 * (hinge2_range[1] - hinge2_range[0])
        
        # 设置所有夹爪关节
        for i in range(3):  # 3个手指
            self.data.ctrl[self.gripper_ids[i*3]] = slide_target     # 滑动关节
            self.data.ctrl[self.gripper_ids[i*3+1]] = hinge1_target  # 第一铰链关节
            self.data.ctrl[self.gripper_ids[i*3+2]] = hinge2_target  # 第二铰链关节
    
    def plan_and_save_path(self):
        """
        规划路径并保存用于后续显示
        """
        # 记录一系列目标点来模拟路径规划
        waypoints = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # 初始位置
            np.array([0.5, 0.0, -0.5, 0.0, 0.5, 0.0]),     # 中间位置1
            np.array([0.3, 0.5, -0.8, 0.2, 0.8, 0.3]),     # 中间位置2
            np.array([0.0, 0.8, -0.5, 0.0, 0.5, 0.0]),     # 目标位置
        ]
        
        # 生成轨迹点
        full_trajectory = []
        for i in range(len(waypoints) - 1):
            steps = 50
            for j in range(steps):
                ratio = j / (steps - 1)
                point = waypoints[i] * (1 - ratio) + waypoints[i + 1] * ratio
                full_trajectory.append(point)
        
        full_trajectory = np.array(full_trajectory)
        return full_trajectory
    
    def show_path_plot(self, trajectory):
        """
        显示路径规划图表
        """
        # 创建3D图表显示关节空间轨迹
        fig = plt.figure(figsize=(12, 5))
        
        # 显示关节角度变化
        ax1 = fig.add_subplot(121)
        for i in range(6):
            ax1.plot(trajectory[:, i], label=f'关节 {i+1}')
        ax1.set_title('关节空间轨迹')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('关节角度 (弧度)')
        ax1.legend()
        ax1.grid(True)
        
        # 显示3D工作空间轨迹（简化模拟）
        ax2 = fig.add_subplot(122, projection='3d')
        
        # 将关节角度转换为工作空间位置（简化处理）
        x_coords = np.sin(trajectory[:, 0]) * np.cos(trajectory[:, 1])
        y_coords = np.sin(trajectory[:, 0]) * np.sin(trajectory[:, 1])
        z_coords = np.cos(trajectory[:, 0])
        
        ax2.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='末端执行器轨迹')
        ax2.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='起始点')
        ax2.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='终点')
        ax2.set_title('工作空间轨迹')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("路径规划图表已显示，关闭图表窗口以继续...")
    
    def run_demo(self):
        """
        运行完整演示：先显示路径规划图表，然后进行物理仿真
        """
        # 1. 规划路径
        print("正在规划路径...")
        trajectory = self.plan_and_save_path()
        
        # 2. 显示路径规划图表
        print("显示路径规划结果...")
        self.show_path_plot(trajectory)
        
        # 3. 运行物理仿真
        print("启动物理仿真...")
        self.run_physics_simulation()
    
    def run_physics_simulation(self):
        """
        运行物理仿真演示
        """
        # 创建可视化环境
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -30
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0, 0, 0.2]
            
            print("开始演示序列...")
            
            try:
                # 1. 初始位置
                print("1. 移动到初始位置")
                initial_angles = np.array([0.0, -0.5, 0.0, -0.5, 0.0, 0.0])
                joint_trajectory = self.generate_joint_trajectory(initial_angles, duration=2.0)
                for target in joint_trajectory:
                    self.data.ctrl[:6] = target
                    self.set_gripper(-1.0)  # 张开夹爪
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
                
                time.sleep(1)
                
                # 2. 移动到球体位置
                print("2. 移动到球体位置")
                sphere_pos = self.data.body_xpos[self.object_ids[0]].copy()
                target_pos = sphere_pos + np.array([0, 0, 0.1])  # 球体上方
                
                # 简单的逆运动学（这里用直接关节控制代替）
                target_angles = np.array([0.5, 0.0, -0.8, 0.0, 0.8, 0.0])
                joint_trajectory = self.generate_joint_trajectory(target_angles, duration=3.0)
                for target in joint_trajectory:
                    self.data.ctrl[:6] = target
                    self.set_gripper(-1.0)  # 保持张开
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
                
                time.sleep(1)
                
                # 3. 下降到抓取位置
                print("3. 下降到抓取位置")
                target_angles = np.array([0.5, 0.3, -1.0, 0.0, 1.0, 0.0])
                joint_trajectory = self.generate_joint_trajectory(target_angles, duration=2.0)
                for target in joint_trajectory:
                    self.data.ctrl[:6] = target
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
                
                time.sleep(1)
                
                # 4. 闭合夹爪抓取
                print("4. 闭合夹爪抓取")
                for i in range(100):
                    self.set_gripper(1.0)  # 闭合夹爪
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
                
                time.sleep(1)
                
                # 5. 抬起物体
                print("5. 抬起物体")
                target_angles = np.array([0.5, -0.5, -0.5, 0.0, 0.5, 0.0])
                joint_trajectory = self.generate_joint_trajectory(target_angles, duration=3.0)
                for target in joint_trajectory:
                    self.data.ctrl[:6] = target
                    self.set_gripper(1.0)  # 保持闭合
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
                
                print("演示完成!")
                
                # 保持显示几秒钟
                for _ in range(500):
                    self.set_gripper(1.0)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
                    
            except KeyboardInterrupt:
                print("用户中断了演示")


def main():
    # 创建控制器实例
    controller = ArmController("model/robotic_arm/arm_with_gripper.xml")
    
    # 运行演示
    controller.run_demo()


if __name__ == "__main__":
    main()