import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 获取当前脚本所在目录（scripts），然后构建 data.yaml 的绝对路径
    script_dir = os.path.dirname(__file__)
    data_yaml_abs = os.path.abspath(os.path.join(script_dir, '../dataset/data.yaml'))

    model = YOLO('ultralytics/cfg/models/12/yolo12n.yaml')
    model.train(data=data_yaml_abs,  # 使用绝对路径
                amp=False,
                cache=False,
                imgsz=640,
                epochs=100,
                batch=32,
                close_mosaic=0,
                workers=4,
                optimizer='SGD',
                project='runs/train',
                name='baseline',
                )