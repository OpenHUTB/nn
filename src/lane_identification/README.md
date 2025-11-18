# 道路方向识别系统 (Lane Direction Identification System)

一个基于 Python 和 OpenCV 的智能道路方向识别系统，能够自动检测道路轮廓并判断行驶方向（左转/右转/直行）。

## 功能特性

### 🛣️ 道路检测能力
- **轮廓检测**：基于颜色分割和形态学操作，精确识别道路区域
- **弯曲道路识别**：采用凸包算法处理弯曲道路轮廓
- **多方法融合**：结合轮廓检测和车道线检测，提高识别鲁棒性

### 🧭 方向判断
- **智能方向识别**：自动判断左转、右转或直行方向
- **轮廓分析**：通过道路轮廓的几何特征判断方向
- **车道线分析**：基于车道线交汇点进行方向验证

### 🎨 可视化界面
- **用户友好GUI**：基于 Tkinter 的图形界面，操作简单直观
- **双图对比**：同时显示原图和道路轮廓标出结果
- **实时反馈**：显示检测状态、方向结果和置信度信息

### 🔧 技术特点
- **自适应处理**：支持不同分辨率、光照条件的道路图片
- **回退机制**：当主要算法失效时自动切换备用方案
- **参数可调**：关键参数易于调整以适应不同场景

## 环境要求

### 系统要求
- **操作系统**：Windows 10/11, macOS 10.14+, 或 Ubuntu 16.04+
- **Python**：3.7 或更高版本
- **内存**：至少 4GB RAM
- **存储**：至少 500MB 可用空间

### Python 依赖包
```bash
pip install opencv-python==4.5.5.64
pip install numpy==1.21.6
pip install pillow==9.0.1
pip install tkinter  # 通常Python自带
```

## 安装步骤

### 方法一：直接运行
1. 下载项目文件
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行主程序：
   ```bash
   python lane_detection_app.py
   ```

### 方法二：从源码构建
1. 克隆仓库：
   ```bash
   git clone https://github.com/your-username/lane-direction-detection.git
   cd lane-direction-detection
   ```
2. 创建虚拟环境（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. 安装依赖并运行：
   ```bash
   pip install -r requirements.txt
   python lane_detection_app.py
   ```

## 使用说明

### 基本操作流程
1. **启动应用**：运行 `lane_detection_app.py`
2. **选择图片**：点击"选择道路图片"按钮，选择要分析的图像文件
3. **查看原图**：原图将显示在左侧面板
4. **开始检测**：点击"检测道路方向"按钮开始分析
5. **查看结果**：
   - 检测结果将显示在右侧面板
   - 道路方向将显示在结果标签中
   - 详细状态信息显示在状态栏

### 结果解读
- **道路轮廓**：黄色线条标出检测到的道路边界
- **道路区域**：半透明绿色填充显示识别出的道路区域
- **方向指示**：红色箭头指示检测到的行驶方向
- **检测信息**：显示检测到的线条数量和置信度

### 支持的图像格式
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 算法原理

### 道路轮廓检测
1. **颜色空间转换**：将图像从BGR转换到HSV颜色空间
2. **道路区域分割**：基于预定义的颜色范围提取道路区域
3. **形态学处理**：使用开闭运算去除噪声和填充空洞
4. **轮廓提取**：查找最大轮廓并使用凸包算法平滑边界

### 方向判断逻辑
1. **几何特征分析**：
   - 计算道路轮廓在图像顶部和底部的宽度
   - 分析轮廓质心相对于图像中心的位置
   - 检测道路收敛方向

2. **车道线辅助**：
   - 使用霍夫变换检测车道线
   - 根据车道线斜率分类左右车道
   - 分析车道线交汇点判断方向

### 可视化处理
1. **轮廓绘制**：使用不同颜色标记检测元素
2. **区域填充**：半透明叠加显示道路区域
3. **方向指示**：箭头和文本标注明确显示检测结果

## 项目结构

```
lane-direction-detection/
│
├── lane_detection_app.py      # 主应用程序
├── requirements.txt           # 依赖包列表
├── README.md                 # 项目说明文档
│
├── examples/                 # 示例图片
│   ├── straight_road.jpg
│   ├── left_turn.jpg
│   └── right_turn.jpg
│
└── docs/                     # 文档资料
    ├── algorithm_explanation.md
    └── user_manual.md
```

## 参数调整指南

### 道路颜色范围调整
```python
# 在 _detect_road_direction 方法中调整HSV范围
lower_gray = np.array([0, 0, 50])    # 调整下限
upper_gray = np.array([180, 50, 200]) # 调整上限
```

### 方向判断阈值
```python
# 在 _determine_direction_from_contour 方法中调整
if top_width < bottom_width * 0.7:   # 收敛阈值
if cx < width // 2 - width * 0.15:   # 质心偏移阈值
```

### 图像处理参数
```python
# 边缘检测参数
edges = cv2.Canny(blur, 50, 150)     # 调整阈值

# 霍夫变换参数
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                       threshold=30, minLineLength=20, maxLineGap=50)
```

## 常见问题解答

### Q: 系统无法正确识别道路区域
**A**: 尝试调整HSV颜色范围，确保包含实际道路的颜色特征。不同光照条件下的道路颜色可能有所不同。

### Q: 方向判断不准确
**A**: 检查图像中道路是否清晰可见，尝试调整方向判断阈值参数。

### Q: 处理速度较慢
**A**: 对于大尺寸图像，系统会自动缩放以提高处理速度。如需更高精度，可以禁用自动缩放功能。

### Q: 系统无法启动
**A**: 确保所有依赖包已正确安装，特别是OpenCV和Pillow库。

## 性能优化建议

1. **图像预处理**：对于实时应用，可以降低图像分辨率
2. **区域限制**：限定ROI区域，减少不必要的计算
3. **参数调优**：根据具体场景优化算法参数
4. **硬件加速**：考虑使用GPU加速OpenCV运算

## 扩展开发

### 添加新功能
- 实时视频流处理
- 多车道识别
- 道路障碍物检测
- 车速建议功能

### 集成到其他系统
```python
# 作为模块导入使用
from lane_detection_app import LaneDetectionApp

# 或者直接使用检测函数
from lane_detection import detect_road_direction
result = detect_road_direction("road_image.jpg")
```

## 版本历史

### v1.0 (当前版本)
- 基础道路轮廓检测功能
- 图形用户界面
- 方向判断算法
- 可视化结果展示

### 计划功能
- [ ] 实时视频处理
- [ ] 深度学习模型集成
- [ ] 多平台支持（Web、移动端）
- [ ] 性能优化和加速

