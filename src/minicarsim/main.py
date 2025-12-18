"""
基于MuJoCo的自动驾驶仿真数据生成核心代码
功能：车辆动力学仿真、LiDAR点云生成、摄像头图像生成、自动标注、数据保存
"""
import os
import json
import numpy as np
import mujoco
from mujoco import viewer
import time

# -------------------------- 配置参数 --------------------------
# 场景文件路径
XML_PATH = "models/simple_car.xml"
# 输出目录
OUTPUT_DIR = "output/simulation_results"
# LiDAR参数
LIDAR_PARAMS = {
    "pos": [0, 0, 0.8],  # LiDAR在车辆上的安装位置
    "range": 30.0,        # 探测范围（m）
    "azimuth_res": 1.0,   # 方位角分辨率（°）
    "elevation_res": 2.0, # 俯仰角分辨率（°）
    "elevation_min": -15, # 最小俯仰角（°）
    "elevation_max": 15,  # 最大俯仰角（°）
    "lines": 16,          # 线束数
}
# 仿真帧数
SIMULATION_FRAMES = 1000
# -------------------------------------------------------------

class MojocoDataSim:
    def __init__(self, xml_path, output_dir):
        # 初始化输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/lidar", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)

        # 加载MuJoCo模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # 创建可视化窗口
        self.viewer = viewer.launch_passive(self.model, self.data)

        print("可视化窗口已启动")
        print("仿真将在3秒后开始...")
        time.sleep(3)

    def get_world_pose(self, body_name):
        """
        获取指定物体的世界位姿
        :param body_name: 物体名称
        :return: 位置和四元数
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的物体")
        pos = self.data.xpos[body_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[body_id])
        return pos, quat

    def generate_realistic_lidar_data(self):
        """基于MuJoCo光线追踪生成真实的LiDAR点云数据"""
        try:
            # 获取车辆位置和朝向
            vehicle_pos, vehicle_quat = self.get_world_pose("vehicle")
        except ValueError:
            vehicle_pos = np.array([0, 0, 0.5])
            
        # LiDAR相对位置
        lidar_offset = np.array(LIDAR_PARAMS["pos"])
        lidar_pos = vehicle_pos + lidar_offset
        
        # 生成角度范围
        azimuth_angles = np.arange(0, 360, LIDAR_PARAMS["azimuth_res"])  # 方位角：0~360°
        elevation_angles = np.arange(
            LIDAR_PARAMS["elevation_min"],
            LIDAR_PARAMS["elevation_max"] + LIDAR_PARAMS["elevation_res"],
            LIDAR_PARAMS["elevation_res"]
        )  # 俯仰角

        point_cloud = []
        
        # 遍历所有角度，生成激光束
        for az in azimuth_angles:
            for el in elevation_angles:
                # 转换为弧度
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)

                # 计算激光束的方向向量（局部坐标系）
                dir_local = np.array([
                    np.cos(el_rad) * np.cos(az_rad),
                    np.cos(el_rad) * np.sin(az_rad),
                    np.sin(el_rad)
                ])
                
                # 归一化方向向量
                dir_local = dir_local / np.linalg.norm(dir_local)
                
                # 创建参数
                geom_group = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
                geom_id = np.zeros(1, dtype=np.int32)
                
                # 调用射线检测
                distance = mujoco.mj_ray(
                    self.model, self.data,
                    lidar_pos,      # 射线起点
                    dir_local,      # 射线方向
                    geom_group,     # 几何体组
                    1,              # flg_static: 检测静态几何体
                    -1,             # bodyexclude: 不排除任何body
                    geom_id         # 返回碰撞的几何体ID
                )

                # 记录点云数据
                if distance >= 0 and distance <= LIDAR_PARAMS["range"]:  # 如果检测到碰撞且在范围内
                    # 计算交点位置
                    hit_pos = lidar_pos + dir_local * distance
                    point_cloud.append(hit_pos)

        # 转换为numpy数组
        if len(point_cloud) > 0:
            point_cloud = np.array(point_cloud)
        else:
            # 如果没有检测到点，返回空数组
            point_cloud = np.empty((0, 3))
            
        return point_cloud

    def detect_objects(self):
        """检测环境中的物体"""
        detected_objects = []
        
        # 遍历所有物体
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith("obstacle"):
                # 获取物体位置
                pos = self.data.xpos[i].copy()
                
                # 简单的距离检测（10米内认为可检测到）
                try:
                    vehicle_pos, _ = self.get_world_pose("vehicle")
                    distance = np.linalg.norm(pos - vehicle_pos)
                    
                    if distance <= 10.0:
                        # 获取物体类型（根据名称）
                        obj_type = "box"
                        
                        detected_objects.append({
                            "id": i,
                            "name": body_name,
                            "type": obj_type,
                            "position": pos.tolist(),
                            "distance": distance
                        })
                except ValueError:
                    # 如果找不到车辆，跳过距离检测
                    detected_objects.append({
                        "id": i,
                        "name": body_name,
                        "type": "box",
                        "position": pos.tolist(),
                        "distance": 0
                    })

        return detected_objects

    def generate_annotations(self):
        """生成物体检测标注数据"""
        # 检测到的物体
        detected_objects = self.detect_objects()
        
        annotations = {
            "frame": self.frame_count,
            "timestamp": time.time(),
            "objects": detected_objects
        }
        return annotations

    def save_data(self, lidar_data, annotations):
        """保存数据"""
        # 保存LiDAR点云（NPY格式）
        np.save(f"{self.output_dir}/lidar/frame_{self.frame_count:04d}.npy", lidar_data)
        print(f"已保存点云数据: frame_{self.frame_count:04d}.npy (共{len(lidar_data)}个点)")

        # 保存标注数据（JSON格式）
        with open(f"{self.output_dir}/annotations/frame_{self.frame_count:04d}.json", "w") as f:
            json.dump(annotations, f, indent=4)

        self.frame_count += 1

    def run_simulation(self):
        """运行MuJoCo仿真并生成数据"""
        print("开始仿真...")
        self.frame_count = 0
        
        # 设置简单的控制输入
        # 查找车辆的驱动关节
        rear_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_wheel_motor")
        rear_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_wheel_motor")
        
        if rear_left_idx >= 0 and rear_right_idx >= 0:
            print("找到了车辆驱动关节")
        
        prev_detected_count = 0
            
        for i in range(SIMULATION_FRAMES):
            # 设置控制输入（使车辆向前移动并轻微转向）
            if rear_left_idx >= 0:
                self.data.ctrl[rear_left_idx] = 5.0  # 左后轮速度
            if rear_right_idx >= 0:
                self.data.ctrl[rear_right_idx] = 5.0  # 右后轮速度
                
            # 执行仿真步长
            mujoco.mj_step(self.model, self.data)
            
            # 每20帧生成和保存一次数据
            if i % 20 == 0:
                # 生成传感器数据和标注
                lidar_data = self.generate_realistic_lidar_data()
                annotations = self.generate_annotations()
                
                # 显示检测到的物体数量
                detected_count = len(annotations["objects"])
                if detected_count != prev_detected_count:
                    if detected_count > 0:
                        print(f"检测到 {detected_count} 个物体:")
                        for obj in annotations["objects"]:
                            print(f"  - {obj['name']} 距离: {obj['distance']:.2f}m")
                    else:
                        print("未检测到附近物体")
                    prev_detected_count = detected_count

                # 保存数据
                self.save_data(lidar_data, annotations)
                
                print(f"已仿真 {i}/{SIMULATION_FRAMES} 帧")
                
            # 控制仿真速度以便观察
            time.sleep(0.02)

        print(f"仿真完成！数据已保存到：{self.output_dir}")

if __name__ == "__main__":
    print("正在初始化仿真器...")
    try:
        sim = MojocoDataSim(XML_PATH, OUTPUT_DIR)
        sim.run_simulation()
    except FileNotFoundError as e:
        print(f"找不到模型文件: {e}")
        print("请确认XML文件路径是否正确")
    except Exception as e:
        print(f"仿真过程中出现错误: {e}")
        import traceback
        traceback.print_exc()