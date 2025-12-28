# 导入Carla仿真器核心库，用于与仿真世界、车辆、传感器交互
import carla
# 导入时间库，用于控制循环频率和添加延时
import time
# 导入numpy数值计算库，用于位姿、误差的矩阵/数组运算
import numpy as np
# 导入数学库，用于三角函数、弧度转换等基础数学运算
import math
# 导入pygame库，用于可视化摄像头画面和处理窗口事件
import pygame

# 初始化pygame模块（主要用于实时显示摄像头画面，也可扩展为键盘控制）
pygame.init()
# 创建pygame显示窗口，分辨率设置为800x600像素
screen = pygame.display.set_mode((800, 600))
# 设置pygame窗口标题，便于识别当前仿真程序
pygame.display.set_caption("Carla Parking Control")

class CarlaParking:
    """
    Carla无人车倒车入库核心类
    功能概述：
    1. 建立与Carla仿真器的连接，初始化仿真场景（车辆、传感器、目标车位）
    2. 实时获取车辆位姿，计算与目标车位的误差
    3. 基于纯追踪+PID比例控制实现倒车入库的车辆控制
    4. 可视化摄像头画面，管理仿真资源（车辆、传感器的创建与销毁）
    """
    def __init__(self):
        """类初始化函数：建立Carla连接，初始化核心变量"""
        # 1. 连接Carla服务器（默认本地地址localhost，端口2000）
        self.client = carla.Client('localhost', 2000)
        # 设置服务器连接超时时间为10秒，避免程序无限等待
        self.client.set_timeout(10.0)
        # 获取当前Carla仿真世界对象（所有演员、地图的容器）
        self.world = self.client.get_world()
        # 获取蓝图库（包含所有可生成的车辆、传感器等演员的模板）
        self.blueprint_library = self.world.get_blueprint_library()
        # 获取当前仿真地图对象（用于获取道路生成点、车位参考等）
        self.map = self.world.get_map()

        # 2. 初始化演员对象变量（后续在_init_scene中生成）
        self.vehicle = None          # 无人车演员对象（核心控制目标）
        self.camera = None           # RGB摄像头传感器对象（可视化用）
        self.lidar = None            # LiDAR传感器对象（预留车位检测用）
        self.spectator = self.world.get_spectator()  # 仿真器旁观者视角（用于跟随车辆）

        # 3. 初始化车辆控制相关参数
        self.control = carla.VehicleControl()  # Carla车辆控制指令对象（封装油门、刹车、转向、倒车等指令）
        self.max_speed = 5.0  # 倒车入库的最大速度（单位：km/h），低速保证入库精度和安全性
        self.target_parking_transform = None  # 目标车位的位姿（Carla.Transform类型，包含位置和朝向）

        # 4. 调用场景初始化函数，生成车辆、传感器和目标车位
        self._init_scene()

    def _init_scene(self):
        """
        私有场景初始化函数：
        1. 清理场景中现有车辆和传感器，避免冲突
        2. 生成指定型号的无人车，设置初始位置
        3. 手动定义目标车位的位姿（可扩展为自动检测）
        4. 生成并挂载摄像头传感器（LiDAR预留）
        """
        # 第一步：清理场景中已存在的车辆和传感器（重置仿真环境）
        # 筛选并销毁所有车辆演员
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        # 筛选并销毁所有传感器演员
        for actor in self.world.get_actors().filter('*sensor*'):
            actor.destroy()

        # 第二步：生成无人车（特斯拉Model3）
        # 从蓝图库中筛选出特斯拉Model3的车辆蓝图（返回列表，取第一个）
        vehicle_bp = self.blueprint_library.filter('model3')[0]
        # 设置车辆颜色为红色（RGB值：255,0,0），增强可视化辨识度
        vehicle_bp.set_attribute('color', '255,0,0')

        # 获取地图预定义的车辆生成点（道路合法位置，避免生成在障碍物/人行道）
        spawn_points = self.map.get_spawn_points()
        # 选择第30个生成点作为车辆初始位置（不同地图的生成点分布不同，可按需调整索引）
        # 初始位置要求：离目标车位有一定距离（约5m），便于展示倒车入库过程
        start_transform = spawn_points[30]
        # 在初始位置生成车辆演员，并将对象赋值给self.vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, start_transform)
        # 打印车辆ID，便于调试时识别车辆对象
        print(f"车辆生成成功：{self.vehicle.id}")

        # 第三步：定义目标车位的位姿（手动设置，可替换为LiDAR/视觉自动检测）
        # 车位位置：在车辆初始位置的x轴负方向（后方）5m，y轴正方向（右侧）2m，z轴不变（地面高度）
        parking_location = start_transform.location + carla.Location(x=-5.0, y=2.0, z=0.0)
        # 车位朝向：偏航角（yaw）比初始位置大180°，即车辆倒车后车头朝向与初始方向相反
        parking_rotation = start_transform.rotation + carla.Rotation(yaw=180.0)
        # 组合位置和朝向，生成目标车位的Transform对象（Carla中位姿的标准表示）
        self.target_parking_transform = carla.Transform(parking_location, parking_rotation)

        # 第四步：生成RGB摄像头传感器（用于实时可视化车辆视角）
        # 从蓝图库中获取RGB摄像头的传感器蓝图
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        # 设置摄像头分辨率：宽度800像素，高度600像素（与pygame窗口匹配）
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        # 设置摄像头视场角（FOV）为90°，模拟人眼视角范围
        camera_bp.set_attribute('fov', '90')
        # 定义摄像头安装位姿：在车辆中心前1.5m、上2.4m处（俯视前方视角）
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        # 生成摄像头传感器，并挂载到车辆上（随车辆移动）
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        # 注册摄像头数据回调函数：每帧图像生成后自动调用_process_camera_image处理
        self.camera.listen(lambda image: self._process_camera_image(image))

        # 第五步：生成LiDAR传感器（预留，用于后续自动车位检测功能）
        # lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')  # 获取LiDAR蓝图
        # lidar_bp.set_attribute('range', '50')  # 设置LiDAR检测范围为50m
        # lidar_transform = carla.Transform(carla.Location(z=2.0))  # 安装在车辆顶部2m处
        # self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)  # 生成并挂载LiDAR
        # self.lidar.listen(lambda point_cloud: self._process_lidar(point_cloud))  # 注册点云处理回调

    def _process_camera_image(self, image):
        """
        摄像头图像回调处理函数：
        将Carla摄像头的原始数据转换为pygame可显示的格式，并更新窗口画面
        参数：
            image: carla.Image类型，包含摄像头原始像素数据、分辨率等信息
        """
        # 1. 将Carla图像原始字节数据转换为numpy数组（dtype=uint8，对应0-255像素值）
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        # 2. 调整数组形状为（图像高度, 图像宽度, 4），对应RGBA四通道（红、绿、蓝、透明度）
        array = np.reshape(array, (image.height, image.width, 4))
        # 3. 去除alpha（透明度）通道，仅保留RGB三通道（pygame显示无需透明度）
        array = array[:, :, :3]
        # 4. 交换数组的0轴和1轴（Carla图像是(height, width)，pygame是(width, height)）
        array = array.swapaxes(0, 1)
        # 5. 将numpy数组转换为pygame的Surface对象（可直接绘制到窗口）
        surface = pygame.surfarray.make_surface(array)
        # 6. 将Surface对象绘制到pygame窗口的(0,0)位置（左上角）
        screen.blit(surface, (0, 0))
        # 7. 更新pygame窗口显示，使新绘制的画面生效
        pygame.display.update()

    def _process_lidar(self, point_cloud):
        """
        LiDAR点云处理回调函数（占位）：
        后续可扩展为：点云聚类→车位轮廓提取→目标车位位姿计算
        参数：
            point_cloud: carla.LidarMeasurement类型，包含LiDAR点云数据
        """
        pass

    def get_vehicle_pose(self):
        """
        获取车辆当前位姿（简化为2D平面）
        返回：
            numpy数组 [x, y, yaw]，其中：
            x: 车辆在地图中的x坐标（米）
            y: 车辆在地图中的y坐标（米）
            yaw: 车辆偏航角（弧度，绕z轴旋转，0为东，逆时针为正）
        """
        # 获取车辆当前的Transform对象（包含位置和旋转）
        transform = self.vehicle.get_transform()
        # 提取x、y坐标（z坐标为高度，2D平面入库无需考虑）
        x = transform.location.x
        y = transform.location.y
        # 提取偏航角（Carla中为角度，转换为弧度便于三角函数计算）
        yaw = math.radians(transform.rotation.yaw)
        # 返回位姿数组
        return np.array([x, y, yaw])

    def calculate_parking_error(self):
        """
        计算车辆当前位姿与目标车位的误差（倒车入库的核心参考量）
        返回：
            pos_error: numpy数组 [x_error, y_error]，位置误差（米），目标-当前
            yaw_error: 浮点数，朝向误差（弧度），归一化到[-π, π]
        """
        # 获取车辆当前位姿
        current_pose = self.get_vehicle_pose()
        # 构建目标车位的位姿数组（与当前位姿格式一致）
        target_pose = np.array([
            self.target_parking_transform.location.x,  # 车位x坐标
            self.target_parking_transform.location.y,  # 车位y坐标
            math.radians(self.target_parking_transform.rotation.yaw)  # 车位偏航角（转弧度）
        ])

        # 计算xy平面的位置误差（目标位置 - 当前位置）
        pos_error = target_pose[:2] - current_pose[:2]
        # 计算朝向误差（目标偏航角 - 当前偏航角）
        yaw_error = target_pose[2] - current_pose[2]
        # 归一化朝向误差到[-π, π]范围（避免误差超过360°，如350°误差等价于-10°）
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        return pos_error, yaw_error

    def parking_control(self):
        """
        倒车入库核心控制算法（纯追踪+PID比例控制）
        控制逻辑：
        1. 优先调整车辆朝向，消除偏航误差
        2. 速度与距离误差成正比，近距离减速，保证入库平稳
        3. 误差小于阈值时停止车辆，判定入库完成
        返回：
            bool类型，True表示入库完成（距离误差<0.2m），False表示未完成
        """
        # 第一步：获取位姿误差
        pos_error, yaw_error = self.calculate_parking_error()
        # 计算xy平面的距离误差（欧氏距离，即车辆到车位的直线距离）
        distance = np.linalg.norm(pos_error)

        # 第二步：定义控制参数（可根据车辆特性调优）
        k_p_pos = 0.1        # 位置比例系数（预留，当前仅用朝向控制）
        k_p_yaw = 0.5        # 朝向比例系数（越大，转向响应越灵敏）
        max_steer = math.radians(30)  # 最大转向角（30°，模拟真实车辆的转向限位）

        # 第三步：朝向控制（优先消除偏航误差，避免倒车时偏离车位）
        # 比例控制：转向角 = 比例系数 × 朝向误差
        steer = k_p_yaw * yaw_error
        # 限制转向角在[-max_steer, max_steer]范围内，避免过度转向
        steer = np.clip(steer, -max_steer, max_steer)

        # 第四步：速度控制（倒车模式，负速度表示倒车）
        # 速度策略：距离5m时达到最大速度，距离越近速度越慢（线性减速）
        speed = -min(self.max_speed * (distance / 5.0), self.max_speed)
        # 距离误差小于0.2m时，判定入库完成，停止车辆
        if distance < 0.2:
            speed = 0.0       # 速度置0
            steer = 0.0       # 转向角置0
            print("倒车入库完成！")

        # 第五步：转换为Carla车辆控制指令（Carla的控制范围有固定要求）
        self.control.steer = float(steer / max_steer)  # 转向角归一化到[-1, 1]
        # 油门控制：倒车时油门为0（依赖reverse标志实现倒车），前进时油门0.5（本场景仅倒车）
        self.control.throttle = 0.0 if speed < 0 else 0.5
        self.control.brake = 0.0                       # 无刹车（靠减速策略停车）
        self.control.reverse = speed < 0               # 设置倒车标志（True表示倒车）

        # 第六步：将控制指令应用到车辆
        self.vehicle.apply_control(self.control)

        # 返回入库完成状态（距离误差<0.2m）
        return distance < 0.2

    def run(self):
        """
        主循环函数：
        1. 处理pygame窗口事件（如关闭窗口）
        2. 循环执行倒车入库控制逻辑
        3. 更新旁观者视角，跟随车辆移动
        4. 保证循环频率，清理仿真资源
        """
        try:
            # 初始化入库完成标志
            finished = False
            # 主循环：直到入库完成或关闭窗口
            while not finished:
                # 处理pygame事件（仅处理窗口关闭事件，可扩展为键盘控制）
                for event in pygame.event.get():
                    # 检测到窗口关闭事件时，终止循环
                    if event.type == pygame.QUIT:
                        finished = True

                # 执行倒车入库控制逻辑，更新完成标志
                finished = self.parking_control() or finished

                # 更新旁观者视角（便于观察车辆入库过程）
                # 获取车辆当前位姿
                vehicle_transform = self.vehicle.get_transform()
                # 定义旁观者位姿：在车辆后方10m、上方5m，俯角15°，与车辆同向
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(x=-10, z=5),
                    carla.Rotation(yaw=vehicle_transform.rotation.yaw, pitch=-15)
                )
                # 设置旁观者视角
                self.spectator.set_transform(spectator_transform)

                # 控制循环频率为20Hz（每次循环休眠0.05秒），避免CPU占用过高
                time.sleep(0.05)

        finally:
            """无论循环正常结束还是异常终止，都要清理仿真资源"""
            # 销毁车辆演员
            if self.vehicle:
                self.vehicle.destroy()
            # 销毁摄像头传感器
            if self.camera:
                self.camera.destroy()
            # 销毁LiDAR传感器（若启用）
            if self.lidar:
                self.lidar.destroy()
            # 退出pygame模块
            pygame.quit()
            # 打印清理完成提示
            print("资源清理完成")

# 程序入口：当脚本直接运行时执行
if __name__ == '__main__':
    # 实例化倒车入库类
    parking = CarlaParking()
    # 运行主循环
    parking.run()