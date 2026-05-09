r'''
This module implements several examples, including model inference and adversarial attacks.

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

# Model Inference:

Object Detection Demos.

## what.examples.yolov3_demo
## what.examples.yolov4_demo

## what.examples.faster_rcnn_demo
## what.examples.mobilenet_ssd_demo

<br />

# Adversarial Attacks:

Real-time Adversarial Attacks against Object Detection.

## what.examples.yolov3_tog_attack_demo
## what.examples.yolov3_pcb_attack_demo

'''

from what.examples.yolov3_demo import yolov3_inference_demo
from what.examples.yolov4_demo import yolov4_inference_demo
from what.examples.faster_rcnn_demo import frcnn_inference_demo
from what.examples.mobilenet_ssd_demo import mobilenet_ssd_inference_demo

from what.examples.yolov3_tog_attack_demo import yolov3_tog_attack_demo
from what.examples.yolov3_pcb_attack_demo import yolov3_pcb_attack_demo
