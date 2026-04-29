# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from deep_sort.sort import kalman_filter

#INFTY_COST = 1e+5

def optimize_track_detection_matching(
        distance_metric, max_distance, tracks, detections, tracked_index=None,
        dect_index=None):
    
    """
     This function matches trackes with detections using linear assignment . 
     The goal is to find best possible matching pair

    Parameters:
    ----------
        distance_metric: function   calculates the distance between tracks and detections
        max_distance: float The maximum distance (threshold value)
        tracks: [list]    A list of tracks (objects) 
        detections: [list]   A list of detections (observations) 
        tracked_index : Optional[List[int]]  A list of indices to tracks that should be matched. 
        dect_indices : Optional[List[int]]   A list of indices to detections that should be matched.

    """
    
    if tracked_index is None:
        tracked_index = np.arange(len(tracks))
    if dect_index is None:
        dect_index = np.arange(len(detections))

    if len(dect_index) == 0 or len(tracked_index) == 0:
        return [], tracked_index, dect_index  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, tracked_index, dect_index)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_assignment(cost_matrix)

    # Create lists for matches, unmatched_tracks, and unmatched_detections
    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    for i in range(len(tracked_index)):
        row = tracked_index[i]
        if i not in row_indices:
            unmatched_tracks.append(row)        

    for i in range(len(dect_index)):
        col = dect_index[i]
        if i not in col_indices:
            unmatched_detections.append(col)
    
    for row, col in zip(row_indices, col_indices):
        track_idx = tracked_index[row]
        detection_idx = dect_index[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def track_detection_cascade_association(
        distance_metric, max_distance, cascade_age, tracks, detections,
        tracked_index=None, dect_index=None):
    """
    This function is used to associate predicted and observed objects i.e., 
    tracks and detections this is done using cascade approach

    Parameters
    ----------
        distance_metric: function   calculates the distance between tracks and detections
        max_distance: float The maximum distance (threshold value)
        cascade_age: int max track age
        tracks: [list]    A list of tracks (objects) 
        detections: [list]   A list of detections (observations) 
        tracked_index : Optional[List[int]]  A list of indices to tracks that should be matched. 
        dect_indices : Optional[List[int]]   A list of indices to detections that should be matched.

    """
    if tracked_index is None:
        tracked_index = list(range(len(tracks)))
    if dect_index is None:
        dect_index = list(range(len(detections)))

    unmatched_detections = dect_index
    matches = []

    for level in range(cascade_age):
        if len(unmatched_detections) == 0:  
            break

        new_tracked_index = [
            k for k in tracked_index
            if tracks[k].time_since_update == 1 + level
        ]
        if len(new_tracked_index) > 0:  
            #continue

            matched_track, _, unmatched_detections = optimize_track_detection_matching(
                                                    distance_metric, max_distance, tracks, detections,
                                                    new_tracked_index, unmatched_detections)
            matches += matched_track
            
    unmatched_tracks = [k for k in tracked_index if k not in {k for k, _ in matches}]


    return matches, unmatched_tracks, unmatched_detections


def kalman_association_matrix(kf, cost_matrix, tracks, detections, tracked_index, dect_index):
    """
    This function is used to validate which track-detection pairs are 
    useful for association obtained from Kalman filter.

    Parameters:
    ----------
    kf : Kalman filter instance
    cost_matrix : ndarray
        The NxM dimensional cost matrix [tracked_index, dect_index] 
    tracks : List[track.Track]
        Current predicted tracks
    detections : List[detection.Detection] 
    tracked_index : List[int]
        List of tracked indices (rows in `cost_matrix`) 
    dect_index : List[int]
        List of detection indices (columns in `cost_matrix`)
    """

    gating_dim = 4  # Consider both position and velocity components
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    
    measurements = np.asarray(
        [detections[i].convert_to_xyah() for i in dect_index])
    
    for row, track_idx in enumerate(tracked_index):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, False)
        cost_matrix[row, gating_distance > gating_threshold] = 1e+5
    
    return cost_matrix

