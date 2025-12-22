该项目是一个基于人工智能的无人机控制系统，能够用YOLOv8检测图像中的人物，用OpenCV检测人脸，用DeepFace识别人，并能飞近用户从视频流中选定的目标。

C:\Users\20728\nn\
├── drone_system_complete.py      # 主程序（所有功能集成）
├── install_deps.py              # 安装依赖脚本
├── run.bat                      # Windows启动脚本
├── requirements.txt             # 依赖列表
└── flight_logs\                 # 自动创建：数据记录目录
    └── flight_20240101_120000.json  # 数据记录文件