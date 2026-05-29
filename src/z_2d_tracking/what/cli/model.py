import os.path
from pathlib import Path

WHAT_MODEL_PATH =  os.path.join(Path.home(), '.what', 'models')

WHAT_MODEL_NAME_INDEX = 0
WHAT_MODEL_TYPE_INDEX = 1
WHAT_MODEL_DESC_INDEX = 2
WHAT_MODEL_FILE_INDEX = 3
WHAT_MODEL_URL_INDEX  = 4
WHAT_MODEL_HASH_INDEX = 5

what_model_list = [
    ('YOLOv3      (    Darknet    )', 'Object Detection', 'YOLOv3 pretrained on MS COCO dataset.', 'yolov3.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov3.h5', 'e557c671de8c46ff7d83a9a9b9750bcb2958a7275638c931106ada5d6057a26d'),
    ('YOLOv3      (   Mobilenet   )', 'Object Detection', 'YOLOv3 pretrained on MS COCO dataset.', 'yolov3_mobilenet_lite_416_coco.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov3_mobilenet_lite_416_coco.h5', '93eb7be204fca8dd8bafa5146b9795978c9ca6a0ba1d2e2c002f4ed0a1322436'),
    ('YOLOv3 Tiny (    Darknet    )', 'Object Detection', 'YOLOv3 Tiny pretrained on MS COCO dataset.', 'yolov3-tiny.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov3-tiny.h5', '344387880e8ff5e45a313c85f2eeeb2fa4f6d3511ed1e5611aaed438db1f3876'),
    ('YOLOv3 Tiny (   MobileNet   )', 'Object Detection', 'YOLOv3 Tiny pretrained on MS COCO dataset.', 'tiny_yolo3_mobilenet_lite_416_coco.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/tiny_yolo3_mobilenet_lite_416_coco.h5', 'e124316a47915936baa39a157aca58cac86813b2c9bf49646e26c24e80252000'),
    ('YOLOv4      (    Darknet    )', 'Object Detection', 'YOLOv4 pretrained on MS COCO dataset.', 'yolov4.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov4.h5', '54802f99cdbddcb0f31180a55d30be1a80d0b73edaa13e9152829318387512e4'),
    ('YOLOv4 Tiny (    Darknet    )', 'Object Detection', 'YOLOv4 Tiny pretrained on MS COCO dataset.', 'yolov4-tiny.h5', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolov4-tiny.h5', '867f54dced382170538a9ca2374e14e778f80d4abd6011652b911b6aca77384e'),
    ('SSD         ( MobileNet  v1 )', 'Object Detection', 'SSD pretrained on VOC-2012 dataset.', 'mobilenet-v1-ssd-mp-0_675.pth', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/mobilenet-v1-ssd-mp-0_675.pth', '58694cafa60456eeab4e81ae50ff49a01c46ab387bfea5200f047143ecd973a9'),
    ('SSD         ( MobileNet  v2 )', 'Object Detection', 'SSD pretrained on VOC-2012 dataset.', 'mobilenet-v2-ssd-lite-mp-0_686.pth', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/mobilenet-v2-ssd-lite-mp-0_686.pth', 'b0d1ac2cdbf3c241ba837f51eeebc565ea37b95b7258e2604506a2f991e398a4'),
    ('FasterRCNN  (     VGG16     )', 'Object Detection', 'Faster-RCNN pretrained on VOC-2012 dataset.', 'fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth', '3fd279284b536da3eac754404779e32e2e9fdd82d8511bbc7f6c50e14f0c69d2'),
    ('YOLOX X-Large                ', 'Object Detection', 'YOLOX-X pretrained on MS COCO dataset.', 'yolox-x.pth', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolox_x.pth', '5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf'),
    ('YOLOX Large                  ', 'Object Detection', 'YOLOX-L pretrained on MS COCO dataset.', 'yolox-l.pth', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolox_l.pth', '1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917'),
    ('YOLOX Medium                 ', 'Object Detection', 'YOLOX-M pretrained on MS COCO dataset.', 'yolox-m.pth', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolox_m.pth', '60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6'),
    ('YOLOX Small                  ', 'Object Detection', 'YOLOX-S pretrained on MS COCO dataset.', 'yolox-s.pth', 'https://wuhanstudio.nyc3.cdn.digitaloceanspaces.com/what/yolox_s.pth', 'f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30'),
]
