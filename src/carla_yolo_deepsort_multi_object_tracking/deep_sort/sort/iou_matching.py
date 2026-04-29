# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from deep_sort.sort import linear_assignment


def iou(bbox, candidate_boxes):

    """This function calculates the intersection over union score.
        Higher the score , higher the fraction of bounding box occuluded.

    Parameters
    ----------
    bbox : ndarray
            bounding box in format `(x, y, width, height)`.
    candidate_boxes : ndarray
        A matrix of candidate bounding boxes in the same format
        as `bbox`.
        
    """

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]

    candidate_boxes_tl = candidate_boxes[:, :2]  # top left coordinates
    candidate_boxes_br = candidate_boxes[:, :2] + candidate_boxes[:, 2:] # bottom right coordinates

    iou_tl = np.maximum(bbox_tl, candidate_boxes_tl)
    iou_br = np.minimum(bbox_br, candidate_boxes_br)
              
    iou_wh = np.maximum(0., iou_br - iou_tl)

    intersection_area = iou_wh.prod(axis=1)
    bbox_area = bbox[2:].prod()
    candidate_boxes_area = candidate_boxes[:, 2:].prod(axis=1)

    iou_score = intersection_area / (bbox_area + candidate_boxes_area - intersection_area)

    return iou_score


def iou_matrix(tracks, detections, tracked_index=None,
                dect_index=None):
    
    """This function calculates the IoU distance matrix between tracks and detections.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
    detections : List[deep_sort.detection.Detection]
    tracked_index : Optional[List[int]] , optional
        A list of indices to tracks that should be matched. 
    dect_indices : Optional[List[int]] , optional
        A list of indices to detections that should be matched. 


    """
    if tracked_index is None:
        tracked_index = np.arange(len(tracks))
    if dect_index is None:
        dect_index = np.arange(len(detections))

    cost_matrix = np.zeros((len(tracked_index), len(dect_index)))

    for line, track_idx in enumerate(tracked_index):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[line, :] = 1e+5
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidate_boxes = np.asarray(
            [detections[i].tlwh for i in dect_index])
        cost_matrix[line, :] = 1. - iou(bbox, candidate_boxes)

    return cost_matrix
    
