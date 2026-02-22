import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    return wh / (
        (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
        (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh
    )

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.missed = 0

class Sort:
    def __init__(self, max_missed=5):
        self.tracks = []
        self.track_id = 0
        self.max_missed = max_missed

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            matched = False
            for track in self.tracks:
                if iou(det, track.bbox) > 0.3:
                    track.bbox = det
                    track.missed = 0
                    updated_tracks.append(track)
                    matched = True
                    break
            if not matched:
                new_track = Track(det, self.track_id)
                self.track_id += 1
                updated_tracks.append(new_track)

        for track in self.tracks:
            if track not in updated_tracks:
                track.missed += 1
                if track.missed < self.max_missed:
                    updated_tracks.append(track)

        self.tracks = updated_tracks

        results = []
        for track in self.tracks:
            x1,y1,x2,y2 = track.bbox
            results.append([x1,y1,x2,y2,track.id])

        return results