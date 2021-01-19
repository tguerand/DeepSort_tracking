# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 23:19:58 2021

@author: trist
"""

import numpy as np
from kalman import 

def matching_cascade(tracker, detections):
        """Performs the matching cascade
        Args
        --------
        bboxes: list of the detection bboxes
        
        Returns
        --------
        matches: list of [track_id, detection_id] that matches
        unmatched_tracks: list of unmatched track_id
        unmatched_detection: list of unmatched detections id"""
        
        
        
        track_indices = list(range(len(tracker.tracks_list)))
        detection_indices = list(range(len(detections)))
        tracks = tracker.track_list
        unmatched_detections = detection_indices
        matches = []
        
        for level in range(tracker.age_max):
            if len(unmatched_detections) == 0:  # No detections left
                break
    
            track_indices_l = [k for k in track_indices
                               if tracks[k].time_since_update == 1 + level]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue
    
            matches_l, _, unmatched_detections = min_cost_matching(gated_metric,
                                                                   max_distance,
                                                                   tracks,
                                                                   detections,
                                                                   track_indices_l,
                                                                   unmatched_detections)
            matches += matches_l
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    
        
        return matches, unmatched_tracks, unmatched_detections

def gated_metric(tracker, dets, track_indices, detection_indices):
        
        tracks = tracker.tracks_list
    
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        
        cost_matrix = tracker.metric.distance(features, targets)
        cost_matrix = gate_cost_matrix(tracker.kf, cost_matrix, tracks,
                                       dets, track_indices, detection_indices)

        return cost_matrix

def gate_cost_matrix(
    kf, cost_matrix, tracks, detections, track_indices, detection_indices,
    gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        
    return cost_matrix


def gate():
    pass