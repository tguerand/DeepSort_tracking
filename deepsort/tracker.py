# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:59:56 2021

@author: trist
"""

class Track():
    """Each track corresponds to an object detected on the screen"""
    
    
    def __init__(self, mean, cov, track_id):
        
        self.state = -1 # three states: init = -1, confirmed = 0, deleted = 1
        
        # variables for the kalman filter
        self.mean = mean
        self.cov = cov
        
        self.track_id = track_id
        self.age = 1 # number of frames since last successful measurement
        self.hits = 1 # total of measurement updates
        
    def get_position(self):
        """Returns the top left coordinates, width and height of the bbox"""
        
        x_center, y_center, a, height = self.mean[:4].copy()
        width = a*height
        x_left = (x_center-width)/2
        y_left = (y_center-height)/2
        
        bbox = [x_left, y_left, width, height]
        
        return bbox
    
    def check_age(self, max_age):
        """Check if the track is to old
        
        Args
        ----------
        max_age : int, the maximum of age """
        
        if self.age > max_age:
            self.state = 1 # deleted as out of window
        elif self.age > 3:
            self.state = 0 # confirmed as three tentatives
    
    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        
        Args
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
    
    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == -1 and self.hits >= self._n_init:
            self.state = 0


class Tracker():
    
    def __init__(self, max_age, max_iou=0.7, kf):
        
        
        self.tracks_list = []
        self.max_age = max_age
        self.max_iou = 0.7
        self.kf = kf
    
    
    def predict(self):
        
        for track in self.tracks_list:
            track.predict(self.kf)
    
    def update(self, detections):
        """Args
        --------
        detections: list of detections bboxes"""
        
        matches, unmatched_tracks, unmatched_detections = self.matching_cascade(detections)
        
        # update matches
        ...
        # update unmatched tracks
        ...
        # update unmatched_detections
        for detection_id in unmatched_detections:
            self.tracks.append(self.init_track(detecions[detections_id]))
    
    def matching_cascade(self, bboxes):
        """Performs the matching cascade
        Args
        --------
        bboxes: list of the detection bboxes
        
        Returns
        --------
        matches: list of [track_id, detection_id] that matches
        unmatched_tracks: list of unmatched track_id
        unmatched_detection: list of unmatched detections id"""
        
        
        
        track_indices = list(range(len(self.tracks)))
        detection_indices = list(range(len(detections)))

        unmatched_detections = detection_indices
        matches = []
        for level in range(self.age_max):
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

    def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

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
        gating_threshold = kalman_filter.chi2inv95[gating_dim]
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

    def init_track(self, detection_bbox):
        """Initialize a track with a detection bbox
        Args
        --------
        detection_bbox: a xywh bbox
        
        Returns
        --------
        track: a track"""
        mean, covariance = self.kf.initiate(detection)
        self._next_id += 1
        return self.tracks.append(Track(mean, covariance,
                                        self._next_id-1, self.n_init,
                                        self.max_age, detection.feature))