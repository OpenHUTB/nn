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

## 快速开始

### 基础使用示例
```python
import cv2
import numpy as np

def quick_detect_direction(image_path):
    """快速道路方向检测函数"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return "无法读取图像"
    
    height, width = image.shape[:2]
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义道路颜色范围
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    
    # 创建道路掩码
    road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓的凸包
        hull = cv2.convexHull(largest_contour)
        
        # 计算轮廓质心
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            
            # 判断方向
            if cx < width // 2 - width * 0.15:
                return "左转"
            elif cx > width // 2 + width * 0.15:
                return "右转"
            else:
                return "直行"
    
    return "未知方向"

# 使用示例
if __name__ == "__main__":
    result = quick_detect_direction("test_road.jpg")
    print(f"检测结果: {result}")
```

### 完整系统使用
```python
# 完整的道路方向识别系统
from lane_detection_app import LaneDetectionApp
import tkinter as tk

def run_complete_system():
    """运行完整的道路方向识别系统"""
    root = tk.Tk()
    app = LaneDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_complete_system()
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

## 核心代码解析

### 道路轮廓检测核心代码
```python
def detect_road_contour(image):
    """道路轮廓检测核心函数"""
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 道路颜色范围定义
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    
    # 创建道路掩码
    road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    
    # 提取轮廓
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大轮廓并使用凸包
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        return hull
    return None
```

### 方向判断核心代码
```python
def determine_direction(contour, image_width, image_height):
    """方向判断核心函数"""
    # 计算轮廓边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 计算轮廓质心
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w//2, y + h//2
    
    # 分析轮廓点
    contour_points = contour.reshape(-1, 2)
    
    # 计算顶部和底部宽度
    top_y = image_height // 3
    bottom_y = image_height * 2 // 3
    
    top_points = [p for p in contour_points if abs(p[1] - top_y) < 5]
    bottom_points = [p for p in contour_points if abs(p[1] - bottom_y) < 5]
    
    if top_points and bottom_points:
        top_min_x = min(p[0] for p in top_points)
        top_max_x = max(p[0] for p in top_points)
        top_width = top_max_x - top_min_x
        
        bottom_min_x = min(p[0] for p in bottom_points)
        bottom_max_x = max(p[0] for p in bottom_points)
        bottom_width = bottom_max_x - bottom_min_x
        
        # 判断收敛方向
        if top_width < bottom_width * 0.7:
            top_center = (top_min_x + top_max_x) // 2
            if top_center < image_width // 2:
                return "左转"
            else:
                return "右转"
    
    # 基于质心位置判断
    if cx < image_width // 2 - image_width * 0.15:
        return "左转"
    elif cx > image_width // 2 + image_width * 0.15:
        return "右转"
    else:
        return "直行"
```

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
├── utils/                    # 工具函数
│   ├── image_processing.py
│   ├── contour_analysis.py
│   └── visualization.py
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

## 实用工具函数

### 批量处理函数
```python
import os
from pathlib import Path

def batch_process_images(input_folder, output_folder):
    """批量处理文件夹中的道路图片"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    results = []
    for image_file in input_path.glob("*.jpg"):
        # 处理每张图片
        direction, result_image = detect_road_direction(str(image_file))
        
        # 保存结果
        output_file = output_path / f"result_{image_file.name}"
        cv2.imwrite(str(output_file), result_image)
        
        results.append({
            'file': image_file.name,
            'direction': direction,
            'output_file': output_file.name
        })
    
    return results

# 使用示例
if __name__ == "__main__":
    results = batch_process_images("input_images", "output_results")
    for result in results:
        print(f"{result['file']}: {result['direction']}")
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

