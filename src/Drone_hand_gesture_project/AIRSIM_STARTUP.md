# AirSim 手势控制无人机 - 快速启动指南

## 📦 项目说明

本项目实现了通过手势控制 AirSim 无人机，支持：
- ✅ 实时手势识别（MediaPipe）
- ✅ AirSim 模拟器集成
- ✅ 3D 可视化
- ✅ 飞行数据记录
- ✅ 传感器数据采集

## 🚀 快速启动

### 步骤 1：启动 AirSim 模拟器

1. 打开 AirSim 环境：
   ```
   双击运行：d:\机械学习\air\Blocks\WindowsNoEditor\Blocks.exe
   ```

2. 等待模拟器加载完成（看到无人机和场景）

### 步骤 2：运行手势控制程序

```bash
cd d:\机械学习\nn\src\Drone_hand_gesture_project
python main.py
```

### 步骤 3：连接无人机

1. 程序启动后，按 `C` 键连接到 AirSim
2. 看到 "✅ 成功连接到 AirSim 模拟器" 提示
3. 按 `空格键` 起飞无人机

## 🎮 手势控制说明

### 基础手势（MediaPipe 识别）

| 手势 | 动作 | 说明 |
|------|------|------|
| ✋ 张开手掌 | 悬停 | 保持当前位置 |
| 👆 食指向上 | 上升 | 无人机上升 |
| 👇 食指向下 | 下降 | 无人机下降 |
| 👈 食指向左 | 左移 | 无人机向左移动 |
| 👉 食指向右 | 右移 | 无人机向右移动 |
| ✌️ 两指向前 | 前进 | 无人机向前飞行 |
| 👊 握拳 | 停止 | 紧急停止 |

### 键盘控制（备用）

| 按键 | 动作 |
|------|------|
| `W` | 上升 |
| `S` | 下降 |
| `A` | 左移 |
| `D` | 右移 |
| `F` | 前进 |
| `B` | 后退 |
| `空格` | 起飞/降落 |
| `C` | 连接无人机 |
| `ESC` | 退出 |

## 📊 数据采集

### 记录飞行数据

```python
from airsim_controller import AirSimController

controller = AirSimController()
controller.connect()
controller.takeoff()

# 记录 10 秒飞行数据
data = controller.record_flight_data(duration=10.0)

# 保存数据
controller.save_flight_data(data, "my_flight_data.npy")

controller.land()
controller.disconnect()
```

### 获取相机图像

```python
# 获取场景图像
img = controller.get_camera_image(image_type="scene")

# 获取深度图像
depth_img = controller.get_camera_image(image_type="depth")

# 保存图像
import cv2
cv2.imwrite("camera_view.png", img)
```

## 🔧 故障排除

### 问题 1：无法连接 AirSim

**症状**: 按 C 键后显示连接失败

**解决方案**:
1. 确保 Blocks.exe 已启动
2. 检查 settings.json 是否在正确位置
3. 防火墙是否阻止连接
4. 重启 AirSim 模拟器

### 问题 2：手势识别不准确

**解决方案**:
1. 确保光线充足
2. 手部在摄像头中心位置
3. 调整手势检测阈值
4. 重新训练手势模型

### 问题 3：无人机不响应

**解决方案**:
1. 检查是否已按 `C` 键连接
2. 确认无人机已解锁（armed）
3. 检查 AirSim 控制台是否有错误信息

## 📁 项目结构

```
Drone_hand_gesture_project/
├── main.py                    # 主程序入口
├── airsim_controller.py       # AirSim 控制器（新增）
├── drone_controller.py        # 无人机控制器
├── simulation_3d.py          # 3D 可视化
├── physics_engine.py         # 物理引擎
├── gesture_detector.py       # 手势检测
├── gesture_detector_enhanced.py  # 增强手势检测
├── gesture_classifier.py     # 手势分类器
├── gesture_data_collector.py # 数据收集
├── train_gesture_model.py    # 模型训练
├── requirements.txt          # 依赖
└── README.md                # 说明文档
```

## 🎯 下一步

1. **采集手势数据**:
   ```bash
   python gesture_data_collector.py
   ```

2. **训练自定义模型**:
   ```bash
   python train_gesture_model.py --model_type ensemble
   ```

3. **探索其他场景**:
   - 解压更多 AirSim 环境（Africa.zip, Coastline.zip 等）
   - 修改 settings.json 中的场景配置

## 📝 数据导出示例

```python
import numpy as np
import pandas as pd

# 加载飞行数据
data = np.load("flight_data.npy", allow_pickle=True)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 分析数据
print(f"总飞行时间：{df['timestamp'].max():.1f} 秒")
print(f"最大高度：{max(p[2] for p in df['position']):.2f} 米")
print(f"平均速度：{np.mean([np.linalg.norm(v) for v in df['velocity']]):.2f} m/s")

# 导出为 CSV
df.to_csv("flight_data.csv", index=False)
```

## 🔗 参考资源

- [AirSim 官方文档](https://microsoft.github.io/AirSim/)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [nn 仓库贡献指南](https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md)
