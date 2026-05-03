import cv2
import torch

from what.cli.model import *
from what.utils.file import get_file

from what.models.detection.ssd.mobilenet_v1_ssd import MobileNetV1SSD
from what.models.detection.ssd.mobilenet_v2_ssd_lite import MobileNetV2SSDLite
from what.models.detection.datasets.voc import VOC_CLASS_NAMES

from what.models.detection.utils.box_utils import draw_bounding_boxes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

what_ssd_model_list = what_model_list[6:8]

def mobilenet_ssd_inference_demo():

    max_len = max([len(x[WHAT_MODEL_NAME_INDEX]) for x in what_ssd_model_list])
    for i, model in enumerate(what_ssd_model_list, start=1):
        if os.path.isfile(os.path.join(WHAT_MODEL_PATH, model[WHAT_MODEL_FILE_INDEX])):
            downloaded = 'x'
        else:
            downloaded = ' '
        print('[{}] {} : {:<{w}s}\t{}\t{}'.format(downloaded, i, model[WHAT_MODEL_NAME_INDEX], model[WHAT_MODEL_TYPE_INDEX], model[WHAT_MODEL_DESC_INDEX], w=max_len))

    index = input(f"Please input the model index: ")
    while not index.isdigit() or int(index) > len(what_ssd_model_list):
        index = input(f"Model [{index}] does not exist. Please try again: ")

    index = int(index) - 1

    # Download the model first if not exists
    # Check what_model_list for all available models
    if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, what_ssd_model_list[index][WHAT_MODEL_FILE_INDEX])):
        get_file(what_ssd_model_list[index][WHAT_MODEL_FILE_INDEX],
                    WHAT_MODEL_PATH,
                    what_ssd_model_list[index][WHAT_MODEL_URL_INDEX],
                    what_ssd_model_list[index][WHAT_MODEL_HASH_INDEX])

    if index == 0:
        # Initialize the model
        model = MobileNetV1SSD(os.path.join(WHAT_MODEL_PATH, what_ssd_model_list[index][WHAT_MODEL_FILE_INDEX]),
                               VOC_CLASS_NAMES,
                               is_test=True,
                               device=device)

    if index == 1:
        # Initialize the model
        model = MobileNetV2SSDLite(os.path.join(WHAT_MODEL_PATH, what_ssd_model_list[index][WHAT_MODEL_FILE_INDEX]),
                                   VOC_CLASS_NAMES,
                                   is_test=True,
                                   device=device)

    video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

    while not video.isdigit():
        video = input(f"Please input the OpenCV capture device (e.g. 0, 1, 2): ")

    # Capture from camera
    cap = cv2.VideoCapture(int(video))
    #cap.set(3, 1920)
    #cap.set(4, 1080)

    try:
        while True:
            _, orig_image = cap.read()
            if orig_image is None:
                continue

            # Image preprocessing
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            # Run inference
            images, boxes, labels, probs = model.predict(image, 10, 0.4)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw bounding boxes onto the image
            height, width, _ = image.shape

            if len(boxes) > 0:
                output = draw_bounding_boxes(image, boxes, labels, model.class_names, probs);

            cv2.imshow('MobileNet SSD Demo', output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(enumerate)
