r'''
WHite-box Adversarial Toolbox (WHAT) is a python library for Deep Learning Security that focuses on realtime white-box attacks.

## Installation

```
pip install whitebox-adversarial-toolbox
```

Then you can use the cli tool `what` to try real-time adversarial attacks.

```
sage: what [OPTIONS] COMMAND [ARGS]...

  The CLI tool for WHite-box Adversarial Toolbox (WHAT).

Options:
  --help  Show this message and exit.

Commands:
  attack   Manage Attacks
  example  Manage Examples
  model    Manage Deep Learning Models
```

<br />

## what.models

Use `what model list` to list available models:

```
                Model                      Model Type           Description
----------------------------------------------------------------------------------------------------
[ ] 1 : YOLOv3      (    Darknet    )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[ ] 2 : YOLOv3      (   Mobilenet   )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[ ] 3 : YOLOv3 Tiny (    Darknet    )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[ ] 4 : YOLOv3 Tiny (   MobileNet   )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[ ] 5 : YOLOv4      (    Darknet    )   Object Detection        YOLOv4 pretrained on MS COCO dataset.
[ ] 6 : YOLOv4 Tiny (    Darknet    )   Object Detection        YOLOv4 Tiny pretrained on MS COCO dataset.
[ ] 7 : SSD         ( MobileNet  v1 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[ ] 8 : SSD         ( MobileNet  v2 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[ ] 9 : FasterRCNN  (     VGG16     )   Object Detection        Faster-RCNN pretrained on VOC-2012 dataset.
```

Use `what model download` to download pre-trained models:

```
                Model                      Model Type           Description
----------------------------------------------------------------------------------------------------
[x] 1 : YOLOv3      (    Darknet    )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[x] 2 : YOLOv3      (   Mobilenet   )   Object Detection        YOLOv3 pretrained on MS COCO dataset.
[x] 3 : YOLOv3 Tiny (    Darknet    )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[x] 4 : YOLOv3 Tiny (   MobileNet   )   Object Detection        YOLOv3 Tiny pretrained on MS COCO dataset.
[x] 5 : YOLOv4      (    Darknet    )   Object Detection        YOLOv4 pretrained on MS COCO dataset.
[x] 6 : YOLOv4 Tiny (    Darknet    )   Object Detection        YOLOv4 Tiny pretrained on MS COCO dataset.
[x] 7 : SSD         ( MobileNet  v1 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[x] 8 : SSD         ( MobileNet  v2 )   Object Detection        SSD pretrained on VOC-2012 dataset.
[x] 9 : FasterRCNN  (     VGG16     )   Object Detection        Faster-RCNN pretrained on VOC-2012 dataset.

Please input the model index: 
```

<br />

## what.attacks

Use `what attack list` to list available attacks:

```
1 : TOG Attack  Object Detection
2 : PCB Attack  Object Detection
```

Related Papers:

- [Adversarial Objectness Gradient Attacks in Real-time Object Detection Systems](https://ieeexplore.ieee.org/document/9325397).
- [A Man-in-the-Middle Attack against Object Detection Systems](https://arxiv.org/abs/2208.07174).

<br />

## what.examples

Use `what example list` to list available examples:

```
           Demo                Type             Description
--------------------------------------------------------------------------------
1 :     Yolov3 Demo      Model Inference        Yolov3 Object Detection.
2 :     Yolov4 Demo      Model Inference        Yolov4 Object Detection.
3 :   FasterRCNN Demo    Model Inference        FRCNN Object Detection.
4 : MobileNet SSD Demo   Model Inference        MobileNet SSD Object Detection.
5 :  TOG Attack Demo    Adversarial Attack      Real-time TOG Attack against Yolov3 Tiny.
6 :  PCB Attack Demo    Adversarial Attack      Real-time PCB Attack against Yolov3 Tiny.
```

Use `what example run` to run examples.

```
           Demo                Type             Description
--------------------------------------------------------------------------------
1 :     Yolov3 Demo      Model Inference        Yolov3 Object Detection.
2 :     Yolov4 Demo      Model Inference        Yolov4 Object Detection.
3 :   FasterRCNN Demo    Model Inference        FRCNN Object Detection.
4 : MobileNet SSD Demo   Model Inference        MobileNet SSD Object Detection.
5 :  TOG Attack Demo    Adversarial Attack      Real-time TOG Attack against Yolov3 Tiny.
6 :  PCB Attack Demo    Adversarial Attack      Real-time PCB Attack against Yolov3 Tiny.

Please input the example index: 
```

<br />

## what.utils

This module implements several utility functions.

<br />

'''

# Project Imports
from what import models
from what import attacks
from what import utils

# Semantic Version
__version__ = "0.2.3"
