# AutoLane: OpenCV 车道线检测系统

这是一个基于 Python 和 OpenCV 的自动驾驶车道线检测项目。它不依赖深度学习模型，而是通过经典的计算机视觉算法（Canny 边缘检测、霍夫变换、色彩过滤）来实现车道线的实时识别与追踪。

本项目包含一个强大的**实时调参工具 (`tuner.py`)**，允许开发者在视频播放过程中动态调整算法参数，从而快速适配不同的光照和道路环境。

## ✨ 功能特性

* **视觉管线**: 灰度化 -> 高斯模糊 -> Canny 边缘检测 -> ROI 掩码 -> 霍夫直线变换。
* **实时调参器**: 提供可视化滑动条界面，实时寻找最佳的 Canny 阈值和 ROI 区域。
* **鲁棒性优化**:
    * **斜率过滤**: 剔除水平线（阴影）和垂直线（路标）。
    * **历史帧平滑**: 使用 `deque` 队列对最近 10 帧数据取平均，消除抖动。
    * **动态 ROI**: 可配置梯形区域，仅关注路面。

## 🛠️ 环境准备

### 1. 运行环境
* Python 3.8+ (推荐 Python 3.13)
* PyCharm 或 VS Code

### 2. 安装依赖
请在项目根目录下运行：
```bash
pip install -r requirements.txt

```

*(如果尚未创建 `requirements.txt`，手动安装: `pip install opencv-python numpy matplotlib`)*

### 3. 准备数据

请确保项目目录下有测试视频文件（例如 `sample.hevc` 或 `.mp4` 文件）。

* 如果你没有视频，可以从 [这里下载 sample.hevc](https://drive.google.com/file/d/1hP-v8lLn1g1jEaJUBYJhv1mEb32hkMvG/view?usp=sharing)。

---

## 🚀 使用指南 (完整流程)

### 第一步：使用 Tuner 寻找最佳参数

不同的视频（白天/夜晚/阴天）需要不同的参数。运行调参工具来获得最佳效果：

```bash
python tuner.py sample.hevc

```

**操作方法：**

1. **空格键**: 暂停视频，观察当前帧。
2. **滑动条**:
* `Canny Low/High`: 调整边缘检测的灵敏度（左侧黑白窗口）。
* `ROI Top W / Height`: 调整红色梯形框的大小，确保它只包裹车道线，避开天空和树木。
* `Hough Thresh`: 调整识别直线的严格程度（右侧绿色线条）。


3. **记录参数**: 记下效果最好时的 6 个数值。

### 第二步：更新主程序

打开 `main.py`，找到顶部的 **【参数配置区】**，将你刚刚记下的数值填入：

```python
# main.py 顶部
CANNY_LOW = 50        # <--- 填入你的数值
CANNY_HIGH = 150
ROI_TOP_WIDTH = 0.40
...

```

### 第三步：运行车道检测

参数配置完成后，运行主程序查看最终效果：

```bash
python main.py sample.hevc

```

* **按 `q` 键**: 退出程序。

---

## 📂 项目结构

```text
Project_Root/
├── main.py            # [核心] 车道检测主程序 (需填入调优后的参数)
├── tuner.py           # [工具] 可视化调参工具 (含滑动条)
├── requirements.txt   # 依赖库列表
├── README.md          # 项目文档
└── sample.hevc        # 测试视频文件

```

## ⚠️ 常见问题

**Q1: 运行 `tuner.py` 或 `main.py` 时报错 `Assertion failed` 或无法打开视频？**

* **原因**: OpenCV 可能缺少 HEVC (H.265) 解码器。
* **解决**: 将 `sample.hevc` 转换为 `.mp4` 格式，或者重新编译安装带 FFmpeg 支持的 OpenCV。

**Q2: 检测框乱跳或消失？**

* **原因**: 参数不适合当前视频的光照。
* **解决**: 请务必先运行 `tuner.py`，针对当前视频调整 `Canny` 阈值和 `ROI` 范围。

## 🔮 路线图 (Roadmap)

* [x] 基础 OpenCV 车道线检测
* [x] 增加可视化调参工具
* [x] 增加帧间平滑 (Smoothing)
* [ ] 集成 YOLOv8 进行车辆/行人目标检测
* [ ] 结合深度学习实现端到端控制 (类似 comma.ai)

## 📚 参考资料

* [OpenCV Documentation](https://www.google.com/search?q=https://docs.opencv.org/)
* [Hough Circle Transform Explained](https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html)

```

```