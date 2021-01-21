# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:59:56 2021

@author: trist
"""
import numpy as np
from . import cascade

class Track():
    """Each track corresponds to an object detected on the screen"""
    
    
    def __init__(self, mean, cov, track_id, match_thresh):
        
        self.state = -1 # three states: init = -1, confirmed = 0, deleted = 1
        
        # variables for the kalman filter
        self.mean = mean
        self.cov = cov
        
        self.track_id = track_id
        self.age = 1 # number of frames since last successful measurement
        self.hits = 1 # total of measurement updates
        self.age_update = 1 # number of frames since last update
        self.match_thresh = match_thresh
        
    def get_position(self):
        """Returns the top left coordinates, width and height of the bbox"""
        
        x_center, y_center, a, height = self.mean[:4].copy()
        width = a*height
        x_left = (x_center-width)/2
        y_left = (y_center-height)/2
        
        bbox = [x_left, y_left, width, height]
        
        return bbox
    
    def to_xyah(self, bbox):
        
        x_left, y_left, width, height = bbox
        a = width/height
        x_center = 2*x_left + width
        y_center = 2*y_left + height
        
        new_bbox = [x_center, y_center, a, height]
        
        return new_bbox
    
    def check_age(self, max_age):
        """Check if the track is to old
        
        Args
        ----------
        max_age : int, the maximum of age """
        
        if self.age > max_age:
            self.state = 1 # deleted as out of window
        elif self.age > 3:
            self.state = 0 # confirmed as three tentatives
            
    def delete(self, max_age):
        
        if self.state == -1:
            self.state = 1
        elif self.age_update > max_age:
            self.state = 1
    
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
        x,y,w,h, feature = detection
        bbox = [x,y,w,h]
        
        self.mean, self.covariance = kf.update(self.mean, self.covariance,
                                               self.to_xyah(bbox))
        self.features.append(feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == -1 and self.hits >= self._n_init:
            self.state = 0


class Tracker():
    
    def __init__(self, metric, kf, max_iou=0.7, max_age=20, match_thresh=100):
        
        self.metric = metric
        self.tracks_list = []
        self.max_age = max_age
        self.max_iou = 0.7
        self.kf = kf
        self.match_thresh = match_thresh
    
    
    def predict(self):
        
        for track in self.tracks_list:
            track.predict(self.kf)
    
    def update(self, detections):
        """Args
        --------
        detections: list of detections bboxes"""
        
        matches, unmatched_tracks, unmatched_detections = cascade.matching_cascade(self, 
                                                                                   detections,
                                                                                   self.match_thresh)
        # Update track set
        # update matches
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        # update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].delete(self.max_age)
        # update unmatched_detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    
    

    def init_track(self, detection):
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