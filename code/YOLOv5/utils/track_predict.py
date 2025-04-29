import cv2
import numpy as np
import torch
from collections import defaultdict, deque

class TrackPredictor:
    def __init__(self, history_size=30, future_steps=10, real_sizes=None):
        self.history_size = history_size
        self.future_steps = future_steps
        self.real_sizes = real_sizes or {}
        self.tracks = defaultdict(lambda: deque(maxlen=history_size))
        self.class_tracks = {}  # Track class for each ID
        self.colors = {}  # Consistent colors for each ID
        
    def process_frame(self, detections, img, class_names, tracked_ids=None):
        """Process detections and draw trajectories
        
        Args:
            detections: Tensor of detections [x1, y1, x2, y2, conf, class]
            img: Image to draw on
            class_names: Dictionary of class names
            tracked_ids: Optional array of tracked IDs corresponding to detections
        """
        height, width = img.shape[:2]
        
        # Process each detection
        for i, (*xyxy, conf, cls_id) in enumerate(detections):
            # Get object ID - either from tracker or generate a new one
            if tracked_ids is not None:
                obj_id = int(tracked_ids[i])
            else:
                # This is a fallback if no tracker is used (not recommended)
                obj_id = i + 1
                
            # Get class name
            cls_id = int(cls_id)
            cls_name = class_names[cls_id]
            
            # Store class for this ID
            self.class_tracks[obj_id] = cls_name
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            
            # Calculate center point of the bottom of the box (feet position)
            center_x = (x1 + x2) // 2
            center_y = y2
            
            # Store the position in track history
            self.tracks[obj_id].append((center_x, center_y))
            
            # Assign consistent color for this ID if not already assigned
            if obj_id not in self.colors:
                # Generate a color based on ID to ensure consistency
                self.colors[obj_id] = (
                    (obj_id * 123) % 255,
                    (obj_id * 85) % 255,
                    (obj_id * 47) % 255
                )
            
            # Draw bounding box
            color = self.colors[obj_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and class
            label = f"{cls_name} {obj_id}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw trajectory (past positions)
            if len(self.tracks[obj_id]) > 1:
                points = list(self.tracks[obj_id])
                for j in range(1, len(points)):
                    # Draw line with increasing thickness for more recent positions
                    thickness = int(np.sqrt(float(j) / 5) + 1)
                    cv2.line(img, points[j-1], points[j], color, thickness)
            
            # Predict future trajectory if we have enough history
            if len(self.tracks[obj_id]) >= 2:
                future_points = self._predict_trajectory(self.tracks[obj_id], self.future_steps)
                
                # Draw predicted trajectory
                last_point = self.tracks[obj_id][-1]
                for point in future_points:
                    cv2.line(img, last_point, point, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(img, point, 3, color, -1)
                    last_point = point
        
        # Clean up old tracks that haven't been updated
        current_ids = set([int(tracked_ids[i]) for i in range(len(tracked_ids))] if tracked_ids is not None 
                          else set(range(1, len(detections) + 1)))
        
        # Remove tracks that haven't been seen in this frame
        ids_to_remove = []
        for track_id in self.tracks:
            if track_id not in current_ids:
                ids_to_remove.append(track_id)
        
        # Only remove tracks if they haven't been updated for a while
        for track_id in ids_to_remove:
            if len(self.tracks[track_id]) < self.history_size:
                del self.tracks[track_id]
                if track_id in self.class_tracks:
                    del self.class_tracks[track_id]
                if track_id in self.colors:
                    del self.colors[track_id]
        
        return img
    
    def _predict_trajectory(self, track, steps):
        """Predict future trajectory based on past positions"""
        if len(track) < 2:
            return []
        
        # Use the last few points to calculate velocity
        positions = list(track)
        if len(positions) >= 5:
            positions = positions[-5:]  # Use last 5 positions for better prediction
            
        # Calculate average velocity vector
        velocities = []
        for i in range(1, len(positions)):
            vx = positions[i][0] - positions[i-1][0]
            vy = positions[i][1] - positions[i-1][1]
            velocities.append((vx, vy))
        
        # Average velocity
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        # Predict future positions
        future_points = []
        last_x, last_y = positions[-1]
        
        for i in range(1, steps + 1):
            # Apply some damping to make predictions more realistic
            damping = 0.9 ** i
            next_x = int(last_x + avg_vx * i * damping)
            next_y = int(last_y + avg_vy * i * damping)
            future_points.append((next_x, next_y))
        
        return future_points