# -*- coding: utf-8 -*-
import os
import requests


def check_yolo_weights():
    """
    检查 YOLOv3 权重文件是否存在，若不存在则提供下载指引。
    """
    weights_path = "models/yolov3.weights"
    cfg_path = "models/yolov3.cfg"

    # YOLOv3 官方权重下载地址
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"

    if not os.path.exists(weights_path):
        print("=" * 50)
        print("警告: 未发现 YOLOv3 权重文件 (yolov3.weights)！")
        print(f"请从以下链接下载并放入 models/ 文件夹:")
        print(weights_url)
        print("=" * 50)
        return False

    print("确认: YOLOv3 权重文件已就绪。")
    return True


if __name__ == "__main__":
    # 简单的模块测试
    check_yolo_weights()