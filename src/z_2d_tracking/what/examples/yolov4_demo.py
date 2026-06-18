import cv2
import os.path

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.utils.box_utils import draw_bounding_boxes

from what.models.detection.yolo.yolov4 import YOLOV4
from what.models.detection.yolo.yolov4_tiny import YOLOV4_TINY

from what.cli.model import *

from what.utils.file import get_file

what_yolov4_model_list = what_model_list[4:6]

def yolov4_inference_demo():

    max_len = max([len(x[WHAT_MODEL_NAME_INDEX]) for x in what_yolov4_model_list])
    for i, model in enumerate(what_yolov4_model_list, start=1):
        if os.path.isfile(os.path.join(WHAT_MODEL_PATH, model[WHAT_MODEL_FILE_INDEX])):
            downloaded = 'x'
        else:
            downloaded = ' '
        print('[{}] {} : {:<{w}s}\t{}\t{}'.format(downloaded, i, model[WHAT_MODEL_NAME_INDEX], model[WHAT_MODEL_TYPE_INDEX], model[WHAT_MODEL_DESC_INDEX], w=max_len))

    index = input(f"Please input the model index: ")
    while not index.isdigit() or int(index) > len(what_yolov4_model_list):
        index = input(f"Model [{index}] does not exist. Please try again: ")

    index = int(index) - 1

    # Download the model first if not exists
    WHAT_YOLOV4_MODEL_FILE = what_yolov4_model_list[index][WHAT_MODEL_FILE_INDEX]
    WHAT_YOLOV4_MODEL_URL  = what_yolov4_model_list[index][WHAT_MODEL_URL_INDEX]
    WHAT_YOLOV4_MODEL_HASH = what_yolov4_model_list[index][WHAT_MODEL_HASH_INDEX]

    if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE)):
        get_file(WHAT_YOLOV4_MODEL_FILE,
                WHAT_MODEL_PATH,
                WHAT_YOLOV4_MODEL_URL,
                WHAT_YOLOV4_MODEL_HASH)

    if index == 0:
        model = YOLOV4(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE))

    if index == 1:
        model = YOLOV4_TINY(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE))

    video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

    while not video.isdigit():
        video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

    try:
        # Capture from camera or video
        if video.isdigit():
            cap = cv2.VideoCapture(int(video))
        else:
            cap = cv2.VideoCapture(video)

        #cap.set(3, 1920)
        #cap.set(4, 1080)

        while True:
            _, orig_image = cap.read()
            if orig_image is None:
                continue

            # Image preprocessing
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            # Run inference
            images, boxes, labels, probs = model.predict(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw bounding boxes onto the image
            if len(boxes) > 0:
                output = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

            cv2.imshow('YOLOv4 Demo', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
