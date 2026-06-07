import cv2
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Sensors import YOLOPv2Detector

def debug_yolopv2_on_image(image_path: str, output_path: str):
    # Check file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read image.")
        return

    detector = YOLOPv2Detector()

    result_img, results = detector.detect(image)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result_img)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    input_dir = 'tools\yolov2p_debug_tool\input'
    output_dir = 'tools\yolov2p_debug_tool\output'
    filenames = []
    for file in os.listdir(os.fsencode(input_dir)):
        filename = os.fsdecode(file)
        if filename.endswith(('.jpg')):
            debug_yolopv2_on_image(os.path.join(input_dir, filename), os.path.join(output_dir, filename))