import cv2
import numpy as np
import torch
from collections import defaultdict, deque
import random
from scipy.ndimage import gaussian_filter1d

class TrackPredictor:
    def __init__(self, history_size=30, future_steps=10, real_sizes=None):
        self.history_size = history_size
        self.future_steps = future_steps
        self.real_sizes = real_sizes or {}
        self.tracks = defaultdict(lambda: deque(maxlen=history_size))
        self.class_tracks = {}  # Track class for each ID
        self.colors = {}  # Consistent colors for each ID
        self.movement_patterns = {
            'person': {
                'straight': 0.6,    # 60% chance to continue straight
                'slight_turn': 0.3, # 30% chance to make slight turns
                'sharp_turn': 0.1   # 10% chance to make sharp turns
            },
            'car': {
                'straight': 0.7,    # 70% chance to continue straight (cars tend to move more predictably)
                'slight_turn': 0.25, # 25% chance to make slight turns
                'sharp_turn': 0.05  # 5% chance to make sharp turns
            }
        }
        # Default pattern for other classes
        self.default_pattern = {
            'straight': 0.6,
            'slight_turn': 0.3,
            'sharp_turn': 0.1
        }
        
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
                # Get multiple possible trajectories with weights
                future_trajectories = self._predict_weighted_trajectories(
                    self.tracks[obj_id], 
                    self.future_steps, 
                    cls_name.lower()
                )
                
                # Draw each possible trajectory with opacity based on probability
                for trajectory, probability in future_trajectories:
                    # Adjust alpha (transparency) based on probability
                    alpha = min(1.0, probability * 2)  # Scale up for visibility
                    
                    # Create a color with alpha for this trajectory
                    # Higher probability = more solid line
                    traj_color = (
                        int(color[0] * alpha),
                        int(color[1] * alpha),
                        int(color[2] * alpha)
                    )
                    
                    # Draw the trajectory
                    last_point = self.tracks[obj_id][-1]
                    for point in trajectory:
                        # Convert to integer coordinates
                        point = (int(point[0]), int(point[1]))
                        
                        # Draw line segment
                        cv2.line(img, last_point, point, traj_color, 
                                int(1 + probability * 2), cv2.LINE_AA)
                        
                        # Draw small circle at each prediction point
                        circle_size = int(2 + probability * 3)
                        cv2.circle(img, point, circle_size, traj_color, -1)
                        
                        last_point = point
                    
                    # Add probability text at the end of each trajectory
                    if len(trajectory) > 0:
                        end_point = trajectory[-1]
                        prob_text = f"{probability:.2f}"
                        cv2.putText(img, prob_text, 
                                   (int(end_point[0]), int(end_point[1])),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, traj_color, 1)
        
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
    
    def _predict_weighted_trajectories(self, track, steps, class_name):
        """Predict multiple possible future trajectories with weights
        
        Args:
            track: List of past positions [(x1, y1), (x2, y2), ...]
            steps: Number of steps to predict into future
            class_name: Class name of the object (person, car, etc.)
            
        Returns:
            List of (trajectory, probability) tuples
        """
        if len(track) < 2:
            return []
        
        # Get movement pattern for this class
        pattern = self.movement_patterns.get(class_name, self.default_pattern)
        
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
        
        # Calculate velocity magnitude and direction
        velocity_mag = np.sqrt(avg_vx**2 + avg_vy**2)
        
        # If object is nearly stationary, return single trajectory
        if velocity_mag < 1.0:
            # Just predict straight line with small random movement
            trajectory = self._predict_straight_trajectory(positions[-1], avg_vx, avg_vy, steps)
            return [(trajectory, 1.0)]
        
        # Generate multiple trajectories based on movement patterns
        trajectories = []
        
        # 1. Straight trajectory
        if pattern['straight'] > 0:
            straight_traj = self._predict_straight_trajectory(positions[-1], avg_vx, avg_vy, steps)
            trajectories.append((straight_traj, pattern['straight']))
        
        # 2. Slight turn trajectories (left and right)
        if pattern['slight_turn'] > 0:
            # Split probability between left and right turns
            turn_prob = pattern['slight_turn'] / 2
            
            # Left slight turn
            left_slight = self._predict_turning_trajectory(
                positions[-1], avg_vx, avg_vy, steps, turn_angle=15, smoothing=0.8
            )
            trajectories.append((left_slight, turn_prob))
            
            # Right slight turn
            right_slight = self._predict_turning_trajectory(
                positions[-1], avg_vx, avg_vy, steps, turn_angle=-15, smoothing=0.8
            )
            trajectories.append((right_slight, turn_prob))
        
        # 3. Sharp turn trajectories (left and right)
        if pattern['sharp_turn'] > 0:
            # Split probability between left and right turns
            turn_prob = pattern['sharp_turn'] / 2
            
            # Left sharp turn
            left_sharp = self._predict_turning_trajectory(
                positions[-1], avg_vx, avg_vy, steps, turn_angle=45, smoothing=0.6
            )
            trajectories.append((left_sharp, turn_prob))
            
            # Right sharp turn
            right_sharp = self._predict_turning_trajectory(
                positions[-1], avg_vx, avg_vy, steps, turn_angle=-45, smoothing=0.6
            )
            trajectories.append((right_sharp, turn_prob))
        
        return trajectories
    
    def _predict_straight_trajectory(self, start_pos, vx, vy, steps):
        """Predict a straight trajectory with some noise"""
        trajectory = []
        x, y = start_pos
        
        for i in range(1, steps + 1):
            # Add small random noise to make it more realistic
            noise_x = random.gauss(0, 0.5) * i
            noise_y = random.gauss(0, 0.5) * i
            
            # Apply damping for more realistic prediction
            damping = 0.95 ** i
            
            next_x = x + vx * i * damping + noise_x
            next_y = y + vy * i * damping + noise_y
            
            trajectory.append((next_x, next_y))
        
        return trajectory
    
    def _predict_turning_trajectory(self, start_pos, vx, vy, steps, turn_angle=30, smoothing=0.7):
        """Predict a turning trajectory
        
        Args:
            start_pos: Starting position (x, y)
            vx, vy: Initial velocity components
            steps: Number of steps to predict
            turn_angle: Angle to turn in degrees (positive=left, negative=right)
            smoothing: How smooth the turn should be (0-1)
        """
        trajectory = []
        x, y = start_pos
        
        # Convert velocity to polar coordinates
        speed = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        # Convert turn angle to radians
        turn_angle_rad = np.radians(turn_angle)
        
        # Total angle to distribute over steps
        total_angle_change = turn_angle_rad
        
        # Generate raw points with increasing turn
        raw_points = []
        for i in range(1, steps + 1):
            # Calculate how much to turn at this step (more turn as we go further)
            # Using a sigmoid-like function to make turn more natural
            progress = i / steps
            turn_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            
            # Apply the turn progressively
            current_angle = angle + total_angle_change * turn_progress
            
            # Calculate new position
            damping = 0.95 ** i  # Slow down a bit over time
            step_distance = speed * damping
            
            next_x = x + step_distance * np.cos(current_angle)
            next_y = y + step_distance * np.sin(current_angle)
            
            raw_points.append((next_x, next_y))
            
            # Update for next iteration
            x, y = next_x, next_y
        
        # Apply smoothing if needed
        if smoothing > 0 and len(raw_points) > 2:
            # Extract x and y coordinates
            xs = np.array([p[0] for p in raw_points])
            ys = np.array([p[1] for p in raw_points])
            
            # Apply Gaussian smoothing
            sigma = (1 - smoothing) * 5 + 0.1  # Convert smoothing factor to sigma
            xs_smooth = gaussian_filter1d(xs, sigma)
            ys_smooth = gaussian_filter1d(ys, sigma)
            
            # Recombine into trajectory
            trajectory = list(zip(xs_smooth, ys_smooth))
        else:
            trajectory = raw_points
        
        return trajectory