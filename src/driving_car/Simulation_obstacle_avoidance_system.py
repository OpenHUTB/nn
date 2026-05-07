import carla
import random
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional

# -------------------------- 配置参数 --------------------------
# Carla服务器连接配置
CARLA_HOST = 'localhost'  # 服务器地址（本地调试用localhost）
CARLA_PORT = 2000  # 服务器端口（默认2000）
CARLA_TIMEOUT = 10.0  # 客户端超时时间（秒）

# 车辆运动控制配置
TARGET_SPEED = 5.0  # 目标行驶速度（米/秒）
CONTROL_FREQUENCY = 20.0  # 车辆控制频率（赫兹）
CONTROL_DELAY = 1.0 / CONTROL_FREQUENCY  # 控制循环延迟（秒）

# 避障核心阈值
OBSTACLE_DISTANCE_THRESHOLD = 8.0  # 障碍物距离阈值（米）
OBSTACLE_ANGLE_THRESHOLD = 30.0  # 障碍物检测角度范围（度），左右各30度
BRAKE_INTENSITY = 1.0  # 制动强度（0.0-1.0，1.0为全力制动）
STEER_ANGLE = 0.2  # 避障转向角度（-1.0左~1.0右）

# 激光雷达（LiDAR）配置字典
LIDAR_CONFIG = {
    'range': 50.0,  # 最大探测距离（米）
    'channels': 32,  # 激光雷达通道数
    'points_per_second': 100000,  # 每秒生成点云数量
    'rotation_frequency': 10,  # 旋转频率（赫兹）
    'upper_fov': 10.0,  # 上视场角（度）
    'lower_fov': -30.0,  # 下视场角（度）
    'location': (0.8, 0.0, 1.8)  # 安装位置（相对车辆：前0.8m，中0m，高1.8m）
}


# -------------------------- 全局注册与状态管理 --------------------------
@dataclass
class 传感器注册表:
    """传感器集中式注册表（生命周期管理）"""
    sensors: Dict[str, carla.Actor] = field(default_factory=dict)  # 传感器实例字典
    callbacks: Dict[str, Callable] = field(default_factory=dict)  # 回调函数字典

    def 注册传感器(self, 传感器名称: str, 传感器实例: carla.Actor, 回调函数: Callable):
        """
        注册传感器并绑定数据回调函数
        参数：
            传感器名称: 传感器唯一标识（如"lidar_front"）
            传感器实例: Carla已生成的传感器Actor
            回调函数: 数据处理回调函数
        """
        self.sensors[传感器名称] = 传感器实例
        self.sensors[传感器名称].listen(回调函数)
        print(f"✅ 传感器注册成功：{传感器名称}（ID：{传感器实例.id}）")

    def 注销所有传感器(self):
        """注销并销毁所有已注册的传感器（防止Carla内存泄漏）"""
        for 传感器名称, 传感器 in self.sensors.items():
            if 传感器.is_alive:
                传感器.stop()  # 停止数据监听
                传感器.destroy()  # 销毁传感器Actor
                print(f"❌ 传感器已注销：{传感器名称}")
        self.sensors.clear()
        self.callbacks.clear()


@dataclass
class 车辆状态注册表:
    """车辆状态跟踪注册表"""
    车辆实例: Optional[carla.Vehicle] = None  # 车辆Actor实例
    当前速度: float = 0.0  # 当前车速（米/秒）
    检测到障碍物: bool = False  # 障碍物检测状态
    控制指令历史: List[carla.VehicleControl] = field(default_factory=list)  # 控制指令历史

    def 更新车速(self):
        """从Carla API更新当前车速（米/秒）"""
        if self.车辆实例:
            速度矢量 = self.车辆实例.get_velocity()
            # 计算三维速度（x/y/z轴合成）
            self.当前速度 = np.sqrt(速度矢量.x ** 2 + 速度矢量.y ** 2 + 速度矢量.z ** 2)

    def 记录控制指令(self, 控制指令: carla.VehicleControl):
        """记录控制指令（用于调试）"""
        self.控制指令历史.append(控制指令)
        # 仅保留最近100条记录（节省内存）
        if len(self.控制指令历史) > 100:
            self.控制指令历史.pop(0)


# 初始化全局注册表
传感器注册表实例 = 传感器注册表()
车辆状态实例 = 车辆状态注册表()
激光雷达数据缓冲区 = deque(maxlen=1)  # 线程安全的激光雷达数据缓冲区（仅保留最新1帧）


# -------------------------- 核心功能模块 --------------------------
def 激光雷达数据回调(数据: carla.LidarMeasurement):
    """
    激光雷达数据处理回调函数
    将原始点云数据转换为numpy数组并存储到线程安全缓冲区
    数组格式：每个点[x, y, z, 强度值]
    """
    try:
        # 原始二进制数据转numpy数组（float32类型，每行4个值）
        点云数组 = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        激光雷达数据缓冲区.append(点云数组)
    except Exception as e:
        print(f"⚠️ 激光雷达回调异常：{str(e)}")


def 障碍物检测算法() -> bool:
    """
    车辆前方障碍物检测核心算法
    检测逻辑：
    1. 过滤有效点云（车辆前方：x>0）
    2. 计算每个点相对车辆航向的水平角度
    3. 校验是否有点云落在距离/角度阈值内
    4. 过滤地面点（避免误判）
    返回：
        bool: 检测到障碍物返回True，否则False
    """
    # 无点云数据直接返回无障碍物
    if not 激光雷达数据缓冲区:
        return False

    # 获取最新一帧点云数据
    点云数组 = 激光雷达数据缓冲区[0]

    # 步骤1：过滤前方距离阈值内的点云
    前方点云 = 点云数组[
        (点云数组[:, 0] > 0) &  # x>0表示车辆前方（车辆坐标系x轴向前）
        (点云数组[:, 0] < OBSTACLE_DISTANCE_THRESHOLD)  # 距离小于阈值
        ]
    # 无前方点云则返回无障碍物
    if len(前方点云) == 0:
        return False

    # 步骤2：计算每个点相对车辆航向的水平角度（度）
    # 车辆航向为x轴正方向；y轴正方向=左，y轴负方向=右
    水平角度 = np.degrees(np.arctan2(前方点云[:, 1], 前方点云[:, 0]))

    # 步骤3：过滤角度阈值内的点云（左右各OBSTACLE_ANGLE_THRESHOLD度）
    有效角度掩码 = np.abs(水平角度) <= OBSTACLE_ANGLE_THRESHOLD
    障碍物候选点云 = 前方点云[有效角度掩码]

    # 步骤4：过滤地面点（简单高度过滤，地面点z通常<-1.0，可根据车辆高度调整）
    非地面点云 = 障碍物候选点云[障碍物候选点云[:, 2] > -1.0]

    # 至少10个有效点才判定为障碍物（避免单点噪声误判）
    检测到障碍物 = len(非地面点云) > 10
    return 检测到障碍物


def 注册自动驾驶车辆(世界: carla.World, 蓝图库: carla.BlueprintLibrary) -> bool:
    """
    注册并生成标准化配置的自动驾驶车辆
    参数：
        世界: Carla世界实例
        蓝图库: Carla蓝图库实例
    返回：
        bool: 车辆注册成功返回True，失败返回False
    """
    try:
        # 选择特斯拉Model3蓝图（可替换为其他车型，如'vehicle.bmw.grandtourer'）
        车辆蓝图 = 蓝图库.find('vehicle.tesla.model3')
        车辆蓝图.set_attribute('color', '255,0,0')  # 设置车辆颜色为红色（RGB）
        车辆蓝图.set_attribute('role_name', 'autonomous')  # 设置角色名为自动驾驶车辆

        # 获取地图生成点
        生成点列表 = 世界.get_map().get_spawn_points()
        if not 生成点列表:
            # 无默认生成点时使用自定义坐标（x=100, y=100, z=2）
            生成点 = carla.Transform(carla.Location(x=100, y=100, z=2))
            print(f"⚠️ 无默认生成点，使用自定义坐标：{生成点}")
        else:
            生成点 = random.choice(生成点列表)

        # 生成车辆Actor
        车辆实例 = 世界.spawn_actor(车辆蓝图, 生成点)
        车辆状态实例.车辆实例 = 车辆实例

        # 初始化车辆状态
        车辆实例.set_autopilot(False)  # 关闭Carla默认自动驾驶
        车辆状态实例.更新车速()  # 初始化车速

        print(f"✅ 自动驾驶车辆注册成功：ID={车辆实例.id}，生成点={生成点.location}")
        return True

    except Exception as e:
        print(f"❌ 车辆注册失败：{str(e)}")
        return False


def 注册激光雷达传感器(世界: carla.World, 蓝图库: carla.BlueprintLibrary):
    """
    注册激光雷达传感器（绑定到已注册的车辆）
    参数：
        世界: Carla世界实例
        蓝图库: Carla蓝图库实例
    """
    if not 车辆状态实例.车辆实例:
        raise RuntimeError("无法注册激光雷达 - 未找到已注册的车辆")

    # 创建激光雷达蓝图
    激光雷达蓝图 = 蓝图库.find('sensor.lidar.ray_cast')

    # 应用激光雷达配置参数
    激光雷达蓝图.set_attribute('range', str(LIDAR_CONFIG['range']))
    激光雷达蓝图.set_attribute('channels', str(LIDAR_CONFIG['channels']))
    激光雷达蓝图.set_attribute('points_per_second', str(LIDAR_CONFIG['points_per_second']))
    激光雷达蓝图.set_attribute('rotation_frequency', str(LIDAR_CONFIG['rotation_frequency']))
    激光雷达蓝图.set_attribute('upper_fov', str(LIDAR_CONFIG['upper_fov']))
    激光雷达蓝图.set_attribute('lower_fov', str(LIDAR_CONFIG['lower_fov']))

    # 设置激光雷达安装位置
    安装位置 = carla.Transform(
        carla.Location(
            x=LIDAR_CONFIG['location'][0],
            y=LIDAR_CONFIG['location'][1],
            z=LIDAR_CONFIG['location'][2]
        )
    )

    # 生成激光雷达Actor并绑定到车辆
    激光雷达实例 = 世界.spawn_actor(激光雷达蓝图, 安装位置, attach_to=车辆状态实例.车辆实例)

    # 注册到传感器注册表
    传感器注册表实例.注册传感器(
        传感器名称="front_lidar",
        传感器实例=激光雷达实例,
        回调函数=激光雷达数据回调
    )


def 车辆避障控制循环():
    """
    主控制循环：感知-决策-控制
    按CONTROL_FREQUENCY频率执行避障逻辑
    """
    print("🚗 开始避障控制循环...")
    车辆控制指令 = carla.VehicleControl()  # 初始化控制指令

    while True:
        try:
            # 1. 更新车辆状态
            车辆状态实例.更新车速()

            # 2. 检测前方障碍物
            车辆状态实例.检测到障碍物 = 障碍物检测算法()

            # 3. 避障决策逻辑
            if 车辆状态实例.检测到障碍物:
                # 检测到障碍物：紧急制动 + 小幅度转向
                车辆控制指令.brake = BRAKE_INTENSITY  # 全力制动
                车辆控制指令.throttle = 0.0  # 关闭油门
                车辆控制指令.steer = STEER_ANGLE  # 向右转向避障（左转为负）
                print(f"⚠️ 检测到障碍物！当前车速：{车辆状态实例.当前速度:.2f}m/s，执行制动转向")
            else:
                # 无障碍物：保持匀速直行
                if 车辆状态实例.当前速度 < TARGET_SPEED:
                    车辆控制指令.throttle = 0.5  # 油门开度50%
                else:
                    车辆控制指令.throttle = 0.0  # 关闭油门（滑行）
                车辆控制指令.brake = 0.0  # 松开刹车
                车辆控制指令.steer = 0.0  # 保持直行

            # 4. 应用控制指令并记录
            车辆状态实例.车辆实例.apply_control(车辆控制指令)
            车辆状态实例.记录控制指令(车辆控制指令)

            # 5. 控制频率延迟
            time.sleep(CONTROL_DELAY)

        except KeyboardInterrupt:
            print("\n🛑 控制循环被用户中断")
            break
        except Exception as e:
            print(f"❌ 控制循环异常：{str(e)}")
            break


# -------------------------- 主程序入口 --------------------------
def main():
    客户端 = None
    世界 = None

    try:
        # 1. 连接Carla服务器
        客户端 = carla.Client(CARLA_HOST, CARLA_PORT)
        客户端.set_timeout(CARLA_TIMEOUT)
        世界 = 客户端.get_world()
        蓝图库 = 世界.get_blueprint_library()
        print(f"✅ 成功连接Carla服务器：{CARLA_HOST}:{CARLA_PORT}")

        # 2. 加载地图并设置天气
        世界 = 客户端.load_world('Town03')  # 加载Town03地图（可替换为Town01/Town02）
        天气参数 = carla.WeatherParameters(
            sun_altitude_angle=70.0,  # 太阳高度角（70度=晴天）
            fog_density=0.0  # 雾浓度（0=无雾）
        )
        世界.set_weather(天气参数)
        print("✅ 地图加载完成：Town03，天气：晴天无雾")

        # 3. 注册自动驾驶车辆
        if not 注册自动驾驶车辆(世界, 蓝图库):
            raise RuntimeError("车辆注册失败，程序终止")

        # 4. 注册激光雷达传感器
        注册激光雷达传感器(世界, 蓝图库)

        # 5. 启动避障控制循环
        车辆避障控制循环()

    except KeyboardInterrupt:
        print("\n🛑 程序被用户手动中断")
    except Exception as e:
        print(f"❌ 程序执行异常：{str(e)}")
    finally:
        # 6. 资源清理（关键：防止Carla残留Actor）
        print("\n🧹 开始清理资源...")
        # 注销所有传感器
        传感器注册表实例.注销所有传感器()
        # 销毁车辆实例
        if 车辆状态实例.车辆实例 and 车辆状态实例.车辆实例.is_alive:
            车辆状态实例.车辆实例.destroy()
            print(f"✅ 自动驾驶车辆已销毁（ID：{车辆状态实例.车辆实例.id}）")
        # 重置车辆状态
        车辆状态实例.车辆实例 = None
        print("✅ 所有资源清理完成")


if __name__ == '__main__':
    main()