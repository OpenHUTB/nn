# main.py 车道线预测程序说明
## 核心功能
main.py实现：视频帧预处理→supercombo模型推理→车道线/路径可视化（蓝=左、红=右、绿=路径）。

## 运行命令
cd nn
python src/openpilot_model/main.py 测试视频.mp4

## 依赖安装
pip install numpy==1.24.3 opencv-python==4.8.1.78 tensorflow==2.15.0 matplotlib==3.7.2
sudo apt install ffmpeg -y

## 常见问题
1. No module named "common"：在main.py开头加sys.path.append("openpilot路径")
2. 视频打不开：装FFmpeg，用MP4格式
3. 窗口卡：把max_frames改为5
