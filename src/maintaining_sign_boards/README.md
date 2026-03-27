## CARLA中的交通标志检测与车辆控制

​	本项目利用实时交通标志检测，在CARLA模拟器中模拟自动驾驶车辆的行为。YOLOv8物体检测模型识别交通标志（例如，停车、限速），车辆会相应调整行为。模拟还通过Pygame显示了驾驶员的实时摄像画面。

## 项目结构

“Main.py”——运行CARLA模拟的主脚本，集成基于YOLO的交通标志检测和车辆控制。

## 运行环境

请确保Python环境安装了以下内容：

```
pip install carla pygame numpy torch ultralytics
```

### 附加需求：

1.CARLA 模拟器（版本 ≥ 0.9.13）：从[CARLA GitHub](https://github.com/carla-simulator/carla).下载。

2.CUDA（可选）：如果有GPU的话，用YOLOv8推理更快。

3.Python ≥ 3.7