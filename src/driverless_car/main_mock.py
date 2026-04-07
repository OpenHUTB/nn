# 导入必要的库
import cv2  # OpenCV库，用于图像/视频处理、绘图和显示
import numpy as np  # 数值计算库，用于数组操作和矩阵运算
from ultralytics import YOLO  # 导入YOLOv8模型（来自ultralytics库）
from filterpy.kalman import KalmanFilter  # 导入卡尔曼滤波器（来自filterpy库）
import random  # 随机数库，用于生成虚拟目标的初始参数和随机运动

# --------------------------
# 虚拟环境类：生成随机移动的目标，模拟无人机拍摄的场景
# --------------------------
class VirtualEnv:
    """
    虚拟环境类：生成指定数量的随机移动目标，模拟无人机的视觉场景
    功能：初始化目标、更新目标位置（含边界反弹和不规则运动）、渲染场景画面
    """
    def __init__(self, width=800, height=600, num_objects=3):
        """
        初始化虚拟环境
        :param width: 场景画面宽度（像素），默认800
        :param height: 场景画面高度（像素），默认600
        :param num_objects: 虚拟目标数量，默认3
        """
        self.width = width  # 画面宽度
        self.height = height  # 画面高度
        self.objects = []  # 存储目标信息：每个元素为[x, y, w, h, velocity_x, velocity_y, color]
                           # x/y：目标左上角坐标；w/h：目标宽高；velocity_x/y：x/y方向速度；color：目标颜色
        self._init_objects(num_objects)  # 初始化虚拟目标

    def _init_objects(self, num_objects):
        """
        初始化随机移动的目标（内部方法，外部无需调用）
        :param num_objects: 目标数量
        """
        for _ in range(num_objects):
            # 随机初始位置（确保目标完全在画面内，避免超出边界）
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            # 随机目标尺寸（宽高在30-60像素之间）
            w, h = random.randint(30, 60), random.randint(30, 60)
            # 随机速度（x/y方向，范围-2到2像素/帧，支持正负方向）
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            # 随机目标颜色（RGB格式，0-255）
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # 将目标信息添加到列表
            self.objects.append([x, y, w, h, vx, vy, color])

    def update(self):
        """更新目标位置（模拟随机移动和环境约束）"""
        for obj in self.objects:
            # 解包目标信息
            x, y, w, h, vx, vy, color = obj
            # 1. 更新位置（根据速度移动）
            x += vx
            y += vy
            # 2. 边界反弹（当目标碰到画面边界时，反向并随机调整速度）
            if x <= 0 or x >= self.width - w:
                vx = -vx * random.uniform(0.8, 1.2)  # 反向速度，同时0.8-1.2倍随机调整（模拟非弹性碰撞）
            if y <= 0 or y >= self.height - h:
                vy = -vy * random.uniform(0.8, 1.2)
            # 3. 随机小幅度改变速度（模拟目标的不规则运动，更贴近真实场景）
            vx += random.uniform(-0.3, 0.3)
            vy += random.uniform(-0.3, 0.3)
            # 4. 更新目标信息（覆盖原列表）
            obj[:] = [x, y, w, h, vx, vy, color]

    def render(self):
        """
        渲染环境画面（生成包含目标的图像）
        :return: 渲染后的画面（numpy数组，形状为(height, width, 3)）
        """
        # 创建黑色背景（全0数组，uint8类型）
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # 遍历目标，绘制矩形框
        for obj in self.objects:
            x, y, w, h, _, _, color = obj
            # 绘制矩形：参数为画面、左上角坐标、右下角坐标、颜色、线宽
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        return frame

# --------------------------
# 目标跟踪器类：结合YOLOv8检测和卡尔曼滤波预测，实现目标跟踪
# --------------------------
class Tracker:
    """
    目标跟踪器类：结合YOLOv8目标检测和卡尔曼滤波预测
    功能：加载YOLO模型、初始化卡尔曼滤波器、匹配检测结果与跟踪器、更新跟踪状态
    """
    def __init__(self, model_path="yolov8n.pt"):
        """
        初始化跟踪器
        :param model_path: YOLOv8模型路径，默认使用轻量化的yolov8n.pt（nano版本）
        """
        self.detector = YOLO(model_path)  # 加载YOLOv8预训练模型
        self.trackers = {}  # 跟踪器字典：{track_id: 卡尔曼滤波器实例}，存储每个目标的跟踪器
        self.next_id = 0  # 下一个分配的目标ID（从0开始递增，用于区分不同目标）

    def _init_kalman(self, bbox):
        """
        初始化卡尔曼滤波器（针对边界框[x, y, w, h]，8维状态，4维测量）
        :param bbox: 初始边界框，格式为[x, y, w, h]（x/y：左上角坐标，w/h：宽高）
        :return: 初始化后的卡尔曼滤波器实例
        """
        # 初始化卡尔曼滤波器：dim_x=8（状态维度：x, y, w, h, vx, vy, vw, vh），dim_z=4（测量维度：x, y, w, h）
        # 状态说明：x/y/w/h为边界框参数，vx/vy/vw/vh为对应参数的速度（假设匀速运动）
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # 1. 状态转移矩阵F（定义状态的变化规律，这里采用匀速运动模型）
        # 例如：x(t+1) = x(t) + vx(t)*1；vx(t+1) = vx(t)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ])

        # 2. 测量矩阵H（定义测量值如何映射到状态，只测量x, y, w, h）
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # 测量x
            [0, 1, 0, 0, 0, 0, 0, 0],  # 测量y
            [0, 0, 1, 0, 0, 0, 0, 0],  # 测量w
            [0, 0, 0, 1, 0, 0, 0, 0]   # 测量h
        ])

        # 3. 噪声协方差矩阵（调整滤波的稳定性和响应速度）
        kf.R *= 10.0  # 测量噪声协方差（增大表示信任检测结果的程度降低，默认值*10）
        kf.P *= 1000.0  # 初始状态协方差（增大表示初始状态的不确定性高，默认值*1000）
        kf.Q[-4:, -4:] *= 0.5  # 过程噪声协方差（速度部分，减小表示信任运动模型的程度高，默认值*0.5）

        # 4. 初始化卡尔曼滤波器的状态（前4维为初始边界框，后4维速度初始化为0）
        kf.x[:4] = np.array(bbox).reshape(4, 1)  # 转换为列向量赋值

        return kf

    def update(self, frame):
        """
        更新跟踪状态：检测目标 -> 匹配跟踪器 -> 预测位置 -> 返回跟踪结果
        :param frame: 输入画面（numpy数组，来自虚拟环境或无人机摄像头）
        :return: 跟踪结果列表，每个元素为(track_id, x1, y1, x2, y2)（x2/y2为右下角坐标）
        """
        # --------------------------
        # 1. YOLOv8目标检测（简化版：检测类别0（人），虚拟目标可适配）
        # --------------------------
        # 调用YOLO模型检测，classes=[0]表示只检测COCO数据集中的"人"类别（虚拟目标会被识别为其他类别，可根据需求调整）
        results = self.detector(frame, classes=[0])
        detections = []  # 存储检测到的边界框，格式为[x, y, w, h]
        for result in results:
            for box in result.boxes:
                # 提取边界框的xyxy格式（x1, y1为左上角，x2, y2为右下角）
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # 转换为[x, y, w, h]格式（与卡尔曼滤波器的状态匹配）
                w, h = x2 - x1, y2 - y1
                detections.append([x1, y1, w, h])

        # --------------------------
        # 2. 卡尔曼滤波预测与匹配（简化版：欧氏距离最近匹配）
        # --------------------------
        new_trackers = {}  # 存储更新后的跟踪器（避免修改原字典时的迭代问题）
        for det in detections:
            # 初始化最小距离和最佳匹配ID
            min_dist = float('inf')
            best_id = None
            # 遍历现有跟踪器，寻找最匹配的目标
            for track_id, kf in self.trackers.items():
                # 卡尔曼滤波器预测当前状态（根据运动模型推测目标位置）
                kf.predict()
                # 获取预测的边界框（前4维，展平为1维数组）
                pred = kf.x[:4].flatten()
                # 计算检测框与预测框的欧氏距离（距离越小，匹配度越高）
                dist = np.linalg.norm(np.array(det) - pred)
                # 筛选距离小于50（阈值）的最近匹配（阈值可根据场景调整）
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    best_id = track_id
            # 情况1：匹配成功（找到最佳ID），更新对应的卡尔曼滤波器
            if best_id is not None:
                # 用检测到的边界框更新卡尔曼滤波器（校正预测结果）
                self.trackers[best_id].update(np.array(det).reshape(4, 1))
                # 将更新后的跟踪器加入新字典
                new_trackers[best_id] = self.trackers[best_id]
            # 情况2：无匹配（新目标），初始化新的卡尔曼滤波器
            else:
                # 为新目标初始化卡尔曼滤波器
                new_trackers[self.next_id] = self._init_kalman(det)
                # 下一个目标ID递增
                self.next_id += 1
        # 更新跟踪器字典为新的跟踪器（移除未匹配的旧跟踪器）
        self.trackers = new_trackers

        # --------------------------
        # 3. 整理跟踪结果并返回
        # --------------------------
        tracks = []
        for track_id, kf in self.trackers.items():
            # 获取卡尔曼滤波器的状态（预测的边界框）
            x, y, w, h = kf.x[:4].flatten()
            # 转换为xyxy格式（用于绘图）
            tracks.append((track_id, int(x), int(y), int(x + w), int(y + h)))
        return tracks

# --------------------------
# 主函数：运行虚拟环境和目标跟踪
# --------------------------
def main():
    """主程序：初始化环境和跟踪器，实时运行目标跟踪并显示结果"""
    # 1. 初始化虚拟环境（宽度1024，高度768，4个虚拟目标）
    env = VirtualEnv(width=1024, height=768, num_objects=4)
    # 2. 初始化目标跟踪器（使用默认的yolov8n.pt模型）
    tracker = Tracker()

    # 3. 实时循环：更新环境 -> 跟踪目标 -> 绘制结果 -> 显示画面
    while True:
        # 更新虚拟环境（目标移动）
        env.update()
        # 渲染环境画面
        frame = env.render()

        # 无人机视角跟踪：调用跟踪器更新，获取跟踪结果
        tracks = tracker.update(frame)

        # 绘制跟踪结果（绿色矩形框+目标ID）
        for track_id, x1, y1, x2, y2 in tracks:
            # 绘制跟踪框（绿色，线宽2）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制目标ID（在框的左上角，字体大小0.5，绿色，线宽2）
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示画面（窗口标题为"Drone Tracking (Virtual Objects)"）
        cv2.imshow("Drone Tracking (Virtual Objects)", frame)
        # 按键退出：按下q键（30ms延迟，控制帧率）
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 释放窗口资源
    cv2.destroyAllWindows()

# 程序入口
if __name__ == "__main__":
    main()