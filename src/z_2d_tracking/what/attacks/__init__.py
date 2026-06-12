r'''
This module implements several white-box attacks against Deep Learning models.

Use `what attack list` to list available attacks:

```
1 : TOG Attack  Object Detection
2 : PCB Attack  Object Detection
```

<br />

## what.attacks.detection.yolo.PCB

- [A Man-in-the-Middle Attack against Object Detection Systems](https://arxiv.org/abs/2208.07174).

## what.attacks.detection.yolo.TOG

- [Adversarial Objectness Gradient Attacks in Real-time Object Detection Systems](https://ieeexplore.ieee.org/document/9325397).

'''

from what.attacks.detection.yolo.PCB import PCBAttack
from what.attacks.detection.yolo.TOG import TOGAttack
