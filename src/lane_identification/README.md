这是一个用于智能车自主识别道路的系统
# Lane Identification

**Lane Identification** 是一个基于 **Python + OpenCV** 的图像处理项目，用于识别道路方向（左转 / 右转 / 直行）。  
该项目主要用于自动驾驶小车、视觉循迹机器人或道路图像分析任务。

## 功能特性

- 从**本地图片**中检测车道线
- 自动判断道路方向（Left / Right / Straight）
- 输出一张**标记好车道线和方向**的结果图片
- 支持不同分辨率和角度的道路图片
- 简洁的代码结构，便于二次开发或移植

---

##  环境要求

- Python ≥ 3.7  
- OpenCV ≥ 4.5  
- Numpy

安装依赖：
```bash
pip install opencv-python numpy
