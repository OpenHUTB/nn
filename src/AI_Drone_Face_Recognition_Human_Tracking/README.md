该项目是一个基于人工智能的无人机控制系统，能够用YOLOv8检测图像中的人物，用OpenCV检测人脸，用DeepFace识别人，并能飞近用户从视频流中选定的目标。

AI_Drone_Face_Tracking/
├── main.py                # 主程序（无人机控制+检测+追踪）
├── drone_control.py       # 无人机底层控制封装
├── detection_module.py    # YOLO检测+人脸匹配模块
├── map_overlay.py         # 地图叠加模块（优化版）
├── face_database.py       # 人脸数据库管理
├── utils.py               # 通用工具函数
├── requirements.txt       # 依赖清单
├── map.png                # 地图图片（自行放入）
└── face_database/         # 人脸库文件夹
    ├── person1.jpg
    └── person2.jpg