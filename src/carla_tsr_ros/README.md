# Carla 交通标志识别 ROS 封装

`carla_tsr_ros` 是 `carla_traffic_sign_recognition` 的 ROS 封装，将 YOLOv8 神经网络感知和车辆控制逻辑接入 ROS Noetic，实现从图像输入到控制指令输出的完整话题链路。

## 功能

- **image_publisher**：从本地图片目录循环发布图像到 `/camera/image_raw`
- **perception_node**：订阅图像，用 YOLOv8n 神经网络推理，发布检测结果和标注图像
- **control_node**：订阅检测结果，当 stop sign 检测框面积超过阈值时发布刹车指令
- **roslaunch 启动**：一条命令启动所有节点

## 话题

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 发布 | 图像源 |
| `/perception/traffic_signs` | `std_msgs/String` | 发布 | JSON 格式检测结果 |
| `/perception/annotated_image` | `sensor_msgs/Image` | 发布 | 带检测框的标注图像 |
| `/vehicle/control_cmd` | `geometry_msgs/Twist` | 发布 | 控制指令（linear.x=0 刹车，=1 前进） |

## 运行环境

- Ubuntu 20.04 + ROS Noetic
- Python 3.8（conda 环境 `carla38`）
- PyTorch 2.0.1+cpu
- Ultralytics 8.0.196
- OpenCV 5.0.0
- NumPy 1.24.3

## 依赖安装

    conda activate carla38
    pip install rospkg catkin_pkg empy pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu --no-deps
    pip install ultralytics==8.0.196 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

## 下载 YOLOv8n 权重

本模块不包含模型权重，运行前需下载 `yolov8n.pt`（约 6MB）到模块根目录：

    wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt -P ~/nn/src/carla_tsr_ros/

或手动下载后放到 `~/nn/src/carla_tsr_ros/yolov8n.pt`。

## 编译与运行

### 1. 编译 catkin 工作空间

    mkdir -p ~/carla_tsr_ws/src
    cd ~/carla_tsr_ws/src
    ln -sfn ~/nn/src/carla_tsr_ros carla_tsr_ros
    cd ~/carla_tsr_ws
    source /opt/ros/noetic/setup.bash
    catkin_make

### 2. 启动

    conda activate carla38
    source /opt/ros/noetic/setup.bash
    source ~/carla_tsr_ws/devel/setup.bash
    roslaunch carla_tsr_ros main.launch

或使用入口脚本：

    bash main.sh

### 3. 查看检测结果

另开一个终端（不激活 conda）：

    source /opt/ros/noetic/setup.bash
    source ~/carla_tsr_ws/devel/setup.bash
    rostopic echo /perception/traffic_signs

查看标注图像：

    rosrun image_view image_view image:=/perception/annotated_image

## 神经网络原理

YOLOv8 是单阶段目标检测网络，将检测问题转化为回归问题。损失函数：

    L = lambda_box * L_CIoU + lambda_cls * L_BCE + lambda_dfl * L_DFL

- `L_CIoU`：边界框回归损失（Complete IoU）
- `L_BCE`：分类损失（二元交叉熵）
- `L_DFL`：分布焦点损失

推理输出：对输入图像 I，网络输出 N 个检测结果，每个包含 (cx, cy, w, h, confidence, class_prob)。

## 算法流程

1. `image_publisher` 从 `data/` 目录循环读取图片，按指定频率发布到 `/camera/image_raw`
2. `perception_node` 订阅图像，每 2 帧执行一次 YOLOv8 推理，只检测 COCO 类别 11（stop sign）
3. 检测结果序列化为 JSON，发布到 `/perception/traffic_signs`；同时把可视化图发布到 `/perception/annotated_image`
4. `control_node` 订阅检测结果，当 stop sign 检测框面积占图像面积比例超过 0.3 时，发布 linear.x=0（刹车），否则 linear.x=1（前进）

## 演示

![demo](demo.gif)

## 注意事项

- `cv_bridge` 与 conda 环境的 OpenCV 5.0.0 不兼容（CvType 编码冲突），本模块使用手动构造/解析 `sensor_msgs/Image` 消息，绕过 `cv_bridge`
- 图像源为本地图片，若需接入真实 Carla 相机流，将 `image_publisher` 替换为 Carla ROS bridge 的相机节点即可

## 相关模块

- `carla_traffic_sign_recognition`：Windows 端 Carla 实时感知与控制（无 ROS）
