这是一个关于利用yolov3和carla模拟器进行·自主车辆目标检测与轨迹规划，与模拟器相结合，以提升车辆的感知与决策能力。
使用说明：
使用YOLOv3算法进行物体检测
与CARLA模拟器集成，实现真实的自动驾驶场景
基于探测到的物体和环境约束的轨迹规划
使用TensorBoard的性能监控与可视化

安装
先修条件：
Python 3.7 或更高版本（本人使用的为py3.7.9）
CARLA 模拟器 0.9.11（由于pycharm可能不支持0.9.11，也可自行切换成0.9.12，但同时版本也不可过于高。）

项目结构：
object_detection.py： Python主脚本，用于对象检测和轨迹规划
requirements.txt： 必备 Python 库列表
models/： 包含训练好的YOLOv3模型权重的目录
config/： 包含训练过的YOLOv3模型配置的目录
logs/： TensorBoard 日志文件用于性能监控