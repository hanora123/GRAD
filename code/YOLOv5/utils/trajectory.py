import numpy as np
import torch
from collections import deque

class TrajectoryPredictor:
    def __init__(self, history_size=30, future_steps=10):
        """Initialize trajectory predictor.
        
        Args:
            history_size (int): Number of historical positions to store
            future_steps (int): Number of steps to predict into future
        """
        self.history_size = history_size
        self.future_steps = future_steps
        self.tracks = {}
        
    def update_tracks(self, detections, frame_id):
        """Update object tracks with new detections.
        
        Args:
            detections (torch.Tensor): Tensor of detections [x1, y1, x2, y2, conf, cls_id, obj_id]
            frame_id (int): Current frame ID
        """
        for det in detections:
            obj_id = int(det[-1])
            bbox = det[:4].cpu().numpy()
            center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            
            if obj_id not in self.tracks:
                self.tracks[obj_id] = {
                    'positions': deque(maxlen=self.history_size),
                    'timestamps': deque(maxlen=self.history_size),
                    'last_seen': frame_id
                }
            
            track = self.tracks[obj_id]
            track['positions'].append(center)
            track['timestamps'].append(frame_id)
            track['last_seen'] = frame_id
            
    def predict_trajectory(self, obj_id, current_frame):
        """Predict future trajectory for a specific object.
        
        Args:
            obj_id (int): Object ID to predict trajectory for
            current_frame (int): Current frame number
            
        Returns:
            np.ndarray: Predicted future positions (future_steps x 2)
        """
        if obj_id not in self.tracks:
            return None
            
        track = self.tracks[obj_id]
        positions = np.array(track['positions'])
        timestamps = np.array(track['timestamps'])
        
        if len(positions) < 2:
            return None
            
        # Calculate velocity using recent positions
        time_diff = timestamps[-1] - timestamps[-2]
        if time_diff == 0:
            return None
            
        velocity = (positions[-1] - positions[-2]) / time_diff
        
        # Simple linear prediction
        future_timestamps = np.arange(1, self.future_steps + 1)
        predictions = positions[-1] + velocity * future_timestamps.reshape(-1, 1)
        
        return predictions
        
    def cleanup_old_tracks(self, current_frame, max_age=30):
        """Remove tracks that haven't been updated recently.
        
        Args:
            current_frame (int): Current frame number
            max_age (int): Maximum frames since last update before removal
        """
        remove_ids = []
        for obj_id, track in self.tracks.items():
            if current_frame - track['last_seen'] > max_age:
                remove_ids.append(obj_id)
                
        for obj_id in remove_ids:
            del self.tracks[obj_id]