import numpy as np
from scipy.spatial.distance import cdist

class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = None  # Placeholder for Kalman filter (simplified)
        self.id = np.random.randint(10000)
        self.bbox = bbox
        self.hits = 1
        self.time_since_update = 0
        self.history = [bbox]

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)

    def predict(self):
        self.time_since_update += 1
        return self.bbox

    def get_state(self):
        return self.bbox

class SORT:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        self.frame_count += 1
        pred_bboxes = []
        for tracker in self.trackers:
            pred_bboxes.append(tracker.predict())

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, pred_bboxes)

        # Update matched trackers
        for t, (trk_idx, det_idx) in enumerate(matched):
            self.trackers[trk_idx].update(detections[det_idx]['box'])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]['box']))

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Collect output
        output = []
        for tracker in self.trackers:
            if tracker.time_since_update == 0 and tracker.hits >= self.min_hits:
                output.append({
                    'id': tracker.id,
                    'box': tracker.get_state(),
                    'class_name': detections[0]['class_name'] if detections else 'unknown'
                })

        return output

    def associate_detections_to_trackers(self, detections, pred_bboxes):
        if not detections or not pred_bboxes:
            return [], list(range(len(detections))), list(range(len(pred_bboxes)))

        iou_matrix = np.zeros((len(pred_bboxes), len(detections)))
        for t, trk in enumerate(pred_bboxes):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self.iou(trk, det['box'])

        matched_indices = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(pred_bboxes)))

        for t in range(len(pred_bboxes)):
            for d in range(len(detections)):
                if iou_matrix[t, d] > self.iou_threshold:
                    matched_indices.append([t, d])
                    if d in unmatched_dets:
                        unmatched_dets.remove(d)
                    if t in unmatched_trks:
                        unmatched_trks.remove(t)

        return matched_indices, unmatched_dets, unmatched_trks