import cv2
import numpy as np

def draw_bounding_boxes(image, boxes, labels, class_names, ids):
    '''
    boxes: (x_min, y_min, x_max, y_max)
    '''
    if not hasattr(draw_bounding_boxes, "colours"):
        draw_bounding_boxes.colours = np.random.randint(0, 256, size=(32, 3))

    if len(boxes) > 0:
        assert(boxes.shape[1] == 4)

    # Draw bounding boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i]
        label = f"{class_names[labels[i]]}: {int(ids[i])}"
        # print(label)

        # Draw bounding boxes
        cv2.rectangle(  image, 
                        (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), 
                        tuple([int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]), 
                        4)

        # Draw labels
        cv2.putText(image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    tuple([int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]),
                    2)  # line type
    return image
