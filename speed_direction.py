import numpy as np
from collections import deque

class SpeedDirectionTracker:
    def __init__(self, fps=30, pixel_to_meter=0.1):
        self.tracks = {}  # Dictionary to store tracks: {track_id: {'positions': deque, 'last_time': float}}
        self.fps = fps  # Frames per second
        self.pixel_to_meter = pixel_to_meter  # Conversion factor for speed estimation
    
    def update(self, detections):
        speeds = {}
        directions = {}
        current_time = 1.0 / self.fps  # Time per frame
        
        # Update existing tracks
        for det in detections:
            track_id = det['track_id']
            if track_id == -1:  # Skip if no track ID (no tracking enabled)
                continue
            x1, y1, x2, y2 = det['box']
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    'positions': deque(maxlen=2),  # Store last 2 positions
                    'last_time': 0.0
                }
            
            track = self.tracks[track_id]
            track['positions'].append(centroid)
            track['last_time'] += current_time
            
            # Calculate speed and direction if we have at least 2 positions
            if len(track['positions']) == 2:
                (x1_prev, y1_prev), (x1_curr, y1_curr) = track['positions']
                dx = (x1_curr - x1_prev) * self.pixel_to_meter
                dy = (y1_curr - y1_prev) * self.pixel_to_meter
                dt = current_time
                speed = np.sqrt(dx**2 + dy**2) / dt  # Speed in meters/second
                direction = np.arctan2(dy, dx) * 180 / np.pi  # Direction in degrees
                speeds[track_id] = speed
                directions[track_id] = direction
        
        # Remove stale tracks
        self.tracks = {k: v for k, v in self.tracks.items() if k in [det['track_id'] for det in detections]}
        
        return speeds, directions