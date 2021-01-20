# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 23:19:58 2021

@author: trist
"""

import numpy as np
import kalman
from sklearn.utils.linear_assignment_ import linear_assignment

INF = 1e+5

def matching_cascade(tracker, detections, match_thresh):
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
        
         # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(tracks) if t.state == 0]
        unconfirmed_tracks = [i for i, t in enumerate(tracks) if not t.state == 0]
        
        
        for level in range(tracker.age_max):
            if len(unmatched_detections) == 0:  # No detections left
                break
    
            track_indices_l = [k for k in track_indices
                               if tracks[k].time_since_update == 1 + level]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue
    
            matches_l, _, unmatched_detections = min_cost_matching(gated_metric,
                                                                   match_thresh,
                                                                   tracks,
                                                                   detections,
                                                                   track_indices_l,
                                                                   unmatched_detections)
            matches += matches_l
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        
        matches_a = matches
        unmatched_tracks_a = unmatched_tracks
        
        
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if tracks[k].time_since_update == 1]
        
        unmatched_tracks_a = [k for k in unmatched_tracks_a if tracks[k].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
                iou_cost, tracker.max_iou_distance, tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections
    
        
        return matches, unmatched_tracks, unmatched_detections

def gated_metric(tracker, dets, track_indices, detection_indices):
        
        tracks = tracker.tracks_list
    
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        
        cost_matrix = tracker.metric.distance(features, targets)
        cost_matrix = gate_cost_matrix(tracker.kf, cost_matrix, tracks,
                                       dets, track_indices, detection_indices)

        return cost_matrix

def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices,
                     detection_indices, gated_cost=INF, only_position=False):
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
    
    if only_position:
        gating_dim = 2
    else:
        gating_dim = 4
        
    gating_threshold = kalman.chi2inv95[gating_dim]
    measurements = np.asarray(
        [to_xyah(detections[i]) for i in detection_indices])
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance,
                                             measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        
    return cost_matrix

def to_xyah(bbox):
    """Transforms a xywh bbox to a x_center,y_center,a,h"""
    x, y, w, h = bbox
    a = w/h
    x_center = (x+w)/2
    y_center = (y+h)/2
    
    return [x_center, y_center, a, h]


def min_cost_matching(distance_metric, max_distance, tracks,
                      detections, track_indices=None, detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
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

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INF
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
