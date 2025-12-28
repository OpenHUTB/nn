该项目是一个基于人工智能的无人机控制系统，能够用YOLOv8检测图像中的人物，用OpenCV检测人脸，用DeepFace识别人，并能飞近用户从视频流中选定的目标。

AI-Drone-Face-Tracking/
│
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包列表
│
├── modules/
│   ├── drone_controller.py    # 无人机控制模块
│   ├── face_detector.py       # 人脸检测模块
│   ├── person_detector.py     # 人物检测模块（YOLOv8）
│   ├── face_recognizer.py     # 人脸识别模块（DeepFace）
│   ├── ui_controller.py       # 用户界面控制模块
│   └── voice_synthesizer.py   # 语音合成模块
│
├── faces/                  # 人脸数据库文件夹
│   ├── person1/
│   │   ├── face1.jpg
│   │   └── face2.jpg
│   └── person2/
│       └── face1.jpg
│
├── models/                 # 模型文件夹（可选）
└── utils/                  # 工具函数
    └── helpers.py