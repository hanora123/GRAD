import cv2
import numpy as np
import torch
from collections import defaultdict, deque
import random
from scipy.ndimage import gaussian_filter1d
import math

class KalmanFilter:
    """Simple Kalman filter implementation for tracking"""
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State estimate
        self.x = np.zeros((dim_x, 1))
        
        # Covariance matrix
        self.P = np.eye(dim_x)
        
        # State transition matrix
        self.F = np.eye(dim_x)
        
        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x))
        
        # Measurement noise
        self.R = np.eye(dim_z)
        
        # Process noise
        self.Q = np.eye(dim_x)
        
    def predict(self):
        """Predict next state"""
        # x = Fx
        self.x = np.dot(self.F, self.x)
        
        # P = FPF' + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x
        
    def update(self, z):
        """Update state with measurement z"""
        # y = z - Hx
        y = z - np.dot(self.H, self.x)
        
        # S = HPH' + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # K = PH'S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # x = x + Ky
        self.x = self.x + np.dot(K, y)
        
        # P = (I - KH)P
        I = np.eye(self.dim_x)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.x

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
        
        # Road context information (will be updated dynamically)
        self.road_context = {}
        
        # Trajectory history for each location in the frame
        # This will be used to adjust weights based on common paths
        self.location_history = np.zeros((1080, 1920), dtype=np.float32)  # Default size, will resize as needed
        
        # Decay factor for location history (older trajectories have less influence)
        self.history_decay = 0.95

        # Kalman filters for each track
        self.kalman_filters = {}

        # Motion context learning
        self.scene_flow_map = None
        self.flow_update_rate = 0.05
        
    def update_scene_flow(self, obj_id, positions):
        """Update scene flow map with new trajectory data"""
        if len(positions) < 2:
            return
            
        if self.scene_flow_map is None:
            # Initialize flow map on first use
            h, w = 1080, 1920  # Default size, will be adjusted
            self.scene_flow_map = np.zeros((h, w, 2), dtype=np.float32)
            
        # Update flow map with this trajectory
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]
            x2, y2 = positions[i]
            
            # Calculate flow vector
            flow_x = x2 - x1
            flow_y = y2 - y1
            
            # Skip if out of bounds
            if not (0 <= x1 < self.scene_flow_map.shape[1] and 0 <= y1 < self.scene_flow_map.shape[0]):
                continue
                
            # Update flow map with exponential moving average
            self.scene_flow_map[y1, x1, 0] = (1 - self.flow_update_rate) * self.scene_flow_map[y1, x1, 0] + self.flow_update_rate * flow_x
            self.scene_flow_map[y1, x1, 1] = (1 - self.flow_update_rate) * self.scene_flow_map[y1, x1, 1] + self.flow_update_rate * flow_y
    
    def _initialize_kalman(self, track_id):
        """Initialize a Kalman filter for a new track"""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, vx, y, vy], Measurement: [x, y]
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0],  # vx = vx
            [0, 0, 1, 1],  # y = y + vy
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        kf.H = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 0, 1, 0]   # measure y
        ])
        
        # Measurement noise
        kf.R = np.eye(2) * 5.0  # Measurement uncertainty
        
        # Process noise
        kf.Q = np.eye(4) * 0.1  # Process uncertainty
        
        # Initial state covariance
        kf.P = np.eye(4) * 100.0  # High uncertainty in initial state
        
        return kf
    
    def process_frame(self, detections, img, class_names, tracked_ids=None):
        """Process detections and draw trajectories
        
        Args:
            detections: Tensor of detections [x1, y1, x2, y2, conf, class]
            img: Image to draw on
            class_names: Dictionary of class names
            tracked_ids: Optional array of tracked IDs corresponding to detections
        """
        height, width = img.shape[:2]
        
        # Resize location history if needed
        if self.location_history.shape[0] != height or self.location_history.shape[1] != width:
            self.location_history = np.zeros((height, width), dtype=np.float32)
        
        # Apply decay to location history
        self.location_history *= self.history_decay
        
        # Create heat map overlay for visualization
        heat_map = np.zeros((height, width, 3), dtype=np.uint8)
        
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
            
            # Update location history with this position
            if len(self.tracks[obj_id]) > 1:
                prev_x, prev_y = self.tracks[obj_id][-2]
                # Draw line on location history
                cv2.line(self.location_history, (prev_x, prev_y), (center_x, center_y), 1.0, 2)
            
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
                
                # Create a heat map for this object's predicted trajectories
                obj_heat_map = np.zeros((height, width), dtype=np.float32)
                
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
                    
                    # Points for heat map polygon
                    heat_points = [last_point]
                    
                    for point in trajectory:
                        # Convert to integer coordinates
                        point = (int(point[0]), int(point[1]))
                        
                        # Add point to heat map polygon
                        heat_points.append(point)
                        
                        # Draw line segment
                        cv2.line(img, last_point, point, traj_color, 
                                int(1 + probability * 2), cv2.LINE_AA)
                        
                        # Draw small circle at each prediction point
                        circle_size = int(2 + probability * 3)
                        cv2.circle(img, point, circle_size, traj_color, -1)
                        
                        # Add to object heat map with intensity based on probability
                        cv2.line(obj_heat_map, last_point, point, probability, 
                                int(5 + probability * 10))
                        
                        last_point = point
                    
                    # Add probability text at the end of each trajectory
                    if len(trajectory) > 0:
                        end_point = trajectory[-1]
                        prob_text = f"{probability:.2f}"
                        cv2.putText(img, prob_text, 
                                   (int(end_point[0]), int(end_point[1])),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, traj_color, 1)
                
                # Apply Gaussian blur to create smooth heat map
                obj_heat_map = cv2.GaussianBlur(obj_heat_map, (31, 31), 0)
                
                # Normalize heat map
                if np.max(obj_heat_map) > 0:
                    obj_heat_map = obj_heat_map / np.max(obj_heat_map)
                
                # Convert heat map to color
                for c in range(3):
                    heat_map[:,:,c] = np.maximum(heat_map[:,:,c], 
                                               (obj_heat_map * color[c]).astype(np.uint8))
        
        # Apply heat map overlay to image
        heat_alpha = 0.4  # Transparency of heat map
        mask = (heat_map.sum(axis=2) > 0).astype(np.float32)[:,:,np.newaxis]
        img = img * (1 - mask * heat_alpha) + heat_map * heat_alpha * mask
        
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
                if track_id in self.kalman_filters:
                    del self.kalman_filters[track_id]
        
        # Initialize or update Kalman filters for each track
        for i, (*xyxy, conf, cls_id) in enumerate(detections):
            # Get object ID
            if tracked_ids is not None:
                obj_id = int(tracked_ids[i])
            else:
                obj_id = i + 1
                
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = y2
            
            # Initialize or update Kalman filter for this track
            if obj_id not in self.kalman_filters:
                self.kalman_filters[obj_id] = self._initialize_kalman(obj_id)
                # Initialize state with first position and zero velocity
                self.kalman_filters[obj_id].x = np.array([[center_x], [0], [center_y], [0]])
            else:
                # Predict
                self.kalman_filters[obj_id].predict()
                # Update with new measurement
                self.kalman_filters[obj_id].update(np.array([[center_x], [center_y]]))
        
        return img.astype(np.uint8)
    
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
        
        # Get current position
        current_x, current_y = positions[-1]
        
        # Check location history to adjust weights
        # Sample the location history in the direction of movement
        location_weights = self._sample_location_history(current_x, current_y, avg_vx, avg_vy)
        
        # Adjust pattern weights based on location history
        adjusted_pattern = self._adjust_pattern_weights(pattern, location_weights)
        
        # Generate multiple trajectories based on adjusted movement patterns
        trajectories = []
        
        # 1. Straight trajectory
        if adjusted_pattern['straight'] > 0:
            straight_traj = self._predict_straight_trajectory(positions[-1], avg_vx, avg_vy, steps)
            trajectories.append((straight_traj, adjusted_pattern['straight']))
        
        # 2. Slight turn trajectories (left and right)
        if adjusted_pattern['slight_turn'] > 0:
            # Split probability between left and right turns based on location history
            left_weight, right_weight = self._get_turn_weights(location_weights, 'slight')
            turn_prob_left = adjusted_pattern['slight_turn'] * left_weight
            turn_prob_right = adjusted_pattern['slight_turn'] * right_weight
            
            # Left slight turn
            if turn_prob_left > 0.05:  # Only add if probability is significant
                left_slight = self._predict_turning_trajectory(
                    positions[-1], avg_vx, avg_vy, steps, turn_angle=15, smoothing=0.8
                )
                trajectories.append((left_slight, turn_prob_left))
            
            # Right slight turn
            if turn_prob_right > 0.05:  # Only add if probability is significant
                right_slight = self._predict_turning_trajectory(
                    positions[-1], avg_vx, avg_vy, steps, turn_angle=-15, smoothing=0.8
                )
                trajectories.append((right_slight, turn_prob_right))
        
        # 3. Sharp turn trajectories (left and right)
        if adjusted_pattern['sharp_turn'] > 0:
            # Split probability between left and right turns based on location history
            left_weight, right_weight = self._get_turn_weights(location_weights, 'sharp')
            turn_prob_left = adjusted_pattern['sharp_turn'] * left_weight
            turn_prob_right = adjusted_pattern['sharp_turn'] * right_weight
            
            # Left sharp turn
            if turn_prob_left > 0.02:  # Only add if probability is significant
                left_sharp = self._predict_turning_trajectory(
                    positions[-1], avg_vx, avg_vy, steps, turn_angle=45, smoothing=0.6
                )
                trajectories.append((left_sharp, turn_prob_left))
            
            # Right sharp turn
            if turn_prob_right > 0.02:  # Only add if probability is significant
                right_sharp = self._predict_turning_trajectory(
                    positions[-1], avg_vx, avg_vy, steps, turn_angle=-45, smoothing=0.6
                )
                trajectories.append((right_sharp, turn_prob_right))
        
        # Normalize probabilities to ensure they sum to 1.0
        total_prob = sum(prob for _, prob in trajectories)
        if total_prob > 0:
            trajectories = [(traj, prob / total_prob) for traj, prob in trajectories]
        
        # If we have a Kalman filter for this track, use it for prediction
        obj_id = None
        for id, positions in self.tracks.items():
            if list(positions) == list(track):
                obj_id = id
                break
                
        if obj_id is not None and obj_id in self.kalman_filters:
            kf = self.kalman_filters[obj_id]
            
            # Create a copy of the filter to avoid modifying the original
            kf_copy = KalmanFilter(dim_x=4, dim_z=2)
            kf_copy.x = kf.x.copy()
            kf_copy.P = kf.P.copy()
            kf_copy.F = kf.F.copy()
            kf_copy.Q = kf.Q.copy()
            
            # Predict future trajectory using Kalman filter
            kalman_trajectory = []
            for _ in range(steps):
                kf_copy.predict()
                x = int(kf_copy.x[0, 0])
                y = int(kf_copy.x[2, 0])
                kalman_trajectory.append((x, y))
            
            # Add Kalman prediction as the most likely trajectory
            trajectories.append((kalman_trajectory, 0.6))
            
            # Renormalize probabilities
            total_prob = sum(prob for _, prob in trajectories)
            trajectories = [(traj, prob / total_prob) for traj, prob in trajectories]
        
        return trajectories
    
    def _sample_location_history(self, x, y, vx, vy):
        """Sample the location history in the direction of movement
        
        Args:
            x, y: Current position
            vx, vy: Velocity vector
            
        Returns:
            Dictionary with weights for different directions
        """
        # Normalize velocity vector
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag < 0.001:  # Avoid division by zero
            return {'straight': 1.0, 'left': 0.5, 'right': 0.5}
            
        vx_norm, vy_norm = vx / v_mag, vy / v_mag
        
        # Sample points in front, left, and right of current position
        samples = {
            'straight': 0.0,
            'left': 0.0,
            'right': 0.0
        }
        
        # Sample count and distance
        sample_count = 5
        sample_dist = 20
        
        # Get perpendicular vector (for left/right sampling)
        perp_x, perp_y = -vy_norm, vx_norm
        
        # Sample straight ahead
        for i in range(1, sample_count + 1):
            # Sample point straight ahead
            sx = int(x + vx_norm * i * sample_dist)
            sy = int(y + vy_norm * i * sample_dist)
            
            # Check bounds
            if 0 <= sx < self.location_history.shape[1] and 0 <= sy < self.location_history.shape[0]:
                # Weight by distance (closer samples have more weight)
                weight = 1.0 - (i - 1) / sample_count
                samples['straight'] += self.location_history[sy, sx] * weight
        
        # Sample left
        for i in range(1, sample_count + 1):
            # Angle for sampling (from 0 to 45 degrees)
            angle = i * (np.pi / 8) / sample_count
            
            # Rotate velocity vector to the left
            rx = vx_norm * np.cos(angle) - vy_norm * np.sin(angle)
            ry = vx_norm * np.sin(angle) + vy_norm * np.cos(angle)
            
            # Sample point
            sx = int(x + rx * sample_dist)
            sy = int(y + ry * sample_dist)
            
            # Check bounds
            if 0 <= sx < self.location_history.shape[1] and 0 <= sy < self.location_history.shape[0]:
                # Weight by distance and angle
                weight = 1.0 - (i - 1) / sample_count
                samples['left'] += self.location_history[sy, sx] * weight
        
        # Sample right
        for i in range(1, sample_count + 1):
            # Angle for sampling (from 0 to 45 degrees)
            angle = i * (np.pi / 8) / sample_count
            
            # Rotate velocity vector to the right
            rx = vx_norm * np.cos(angle) + vy_norm * np.sin(angle)
            ry = -vx_norm * np.sin(angle) + vy_norm * np.cos(angle)
            
            # Sample point
            sx = int(x + rx * sample_dist)
            sy = int(y + ry * sample_dist)
            
            # Check bounds
            if 0 <= sx < self.location_history.shape[1] and 0 <= sy < self.location_history.shape[0]:
                # Weight by distance and angle
                weight = 1.0 - (i - 1) / sample_count
                samples['right'] += self.location_history[sy, sx] * weight
        
        # Normalize samples
        total = sum(samples.values())
        if total > 0:
            for k in samples:
                samples[k] /= total
        else:
            # If no samples, use default weights
            samples = {'straight': 0.6, 'left': 0.2, 'right': 0.2}
        
        return samples
    
    def _adjust_pattern_weights(self, pattern, location_weights):
        """Adjust movement pattern weights based on location history
        
        Args:
            pattern: Original movement pattern
            location_weights: Weights from location history
            
        Returns:
            Adjusted pattern
        """
        # Create a copy of the pattern
        adjusted = pattern.copy()
        
        # Adjust straight probability based on location history
        straight_factor = 1.0 + location_weights['straight'] * 0.5
        adjusted['straight'] *= straight_factor
        
        # Adjust turn probabilities based on location history
        # If there's a strong history of turning, increase turn probability
        turn_factor_left = 1.0 + location_weights['left'] * 0.5
        turn_factor_right = 1.0 + location_weights['right'] * 0.5
        
        # Split slight turn probability between left and right
        slight_turn_total = adjusted['slight_turn']
        adjusted['slight_turn_left'] = slight_turn_total * 0.5 * turn_factor_left
        adjusted['slight_turn_right'] = slight_turn_total * 0.5 * turn_factor_right
        del adjusted['slight_turn']
        
        # Split sharp turn probability between left and right
        sharp_turn_total = adjusted['sharp_turn']
        adjusted['sharp_turn_left'] = sharp_turn_total * 0.5 * turn_factor_left
        adjusted['sharp_turn_right'] = sharp_turn_total * 0.5 * turn_factor_right
        del adjusted['sharp_turn']
        
        # Normalize to ensure probabilities sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total
        
        # Recombine for return format
        result = {
            'straight': adjusted['straight'],
            'slight_turn': adjusted['slight_turn_left'] + adjusted['slight_turn_right'],
            'sharp_turn': adjusted['sharp_turn_left'] + adjusted['sharp_turn_right']
        }
        
        return result
    
    def _get_turn_weights(self, location_weights, turn_type):
        """Get weights for left and right turns based on location history
        
        Args:
            location_weights: Weights from location history
            turn_type: 'slight' or 'sharp'
            
        Returns:
            (left_weight, right_weight) tuple
        """
        # Base weights
        left_weight = 0.5
        right_weight = 0.5
        
        # Adjust based on location history
        if location_weights['left'] > location_weights['right']:
            # More history of left turns
            factor = min(0.8, location_weights['left'] / max(0.001, location_weights['right']))
            left_weight = 0.5 * (1.0 + factor * 0.5)
            right_weight = 1.0 - left_weight
        elif location_weights['right'] > location_weights['left']:
            # More history of right turns
            factor = min(0.8, location_weights['right'] / max(0.001, location_weights['left']))
            right_weight = 0.5 * (1.0 + factor * 0.5)
            left_weight = 1.0 - right_weight
        
        # For sharp turns, make the difference more pronounced
        if turn_type == 'sharp':
            # Exaggerate the difference
            diff = left_weight - right_weight
            left_weight = 0.5 + diff * 1.5
            right_weight = 1.0 - left_weight
            
            # Clamp to valid range
            left_weight = max(0.1, min(0.9, left_weight))
            right_weight = max(0.1, min(0.9, right_weight))
            
            # Renormalize
            total = left_weight + right_weight
            left_weight /= total
            right_weight /= total
        
        return left_weight, right_weight
    
    def _sample_location_history(self, x, y, vx, vy):
        """Sample location history to determine common turning patterns
        
        Args:
            x, y: Current position
            vx, vy: Current velocity vector
            
        Returns:
            Dictionary with weights for different directions
        """
        # Initialize weights
        weights = {
            'straight': 0.0,
            'left': 0.0,
            'right': 0.0
        }
        
        # If velocity is too small, return default weights
        velocity_mag = np.sqrt(vx**2 + vy**2)
        if velocity_mag < 1.0:
            return {'straight': 1.0, 'left': 0.5, 'right': 0.5}
        
        # Normalize velocity vector
        nvx = vx / velocity_mag
        nvy = vy / velocity_mag
        
        # Calculate perpendicular vectors (left and right)
        left_vx, left_vy = -nvy, nvx  # 90 degrees counter-clockwise
        right_vx, right_vy = nvy, -nvx  # 90 degrees clockwise
        
        # Sample points in front, left, and right
        samples = 10
        max_dist = 50  # Maximum sampling distance
        
        # Sample straight ahead
        straight_sum = 0.0
        for i in range(1, samples + 1):
            dist = i * (max_dist / samples)
            sample_x = int(x + nvx * dist)
            sample_y = int(y + nvy * dist)
            
            # Check if within bounds
            if 0 <= sample_x < self.location_history.shape[1] and 0 <= sample_y < self.location_history.shape[0]:
                straight_sum += self.location_history[sample_y, sample_x]
        
        # Sample left
        left_sum = 0.0
        for i in range(1, samples + 1):
            dist = i * (max_dist / samples)
            # Mix of forward and left vectors
            mix_factor = i / samples  # Gradually increase left component
            sample_x = int(x + (nvx * (1.0 - mix_factor) + left_vx * mix_factor) * dist)
            sample_y = int(y + (nvy * (1.0 - mix_factor) + left_vy * mix_factor) * dist)
            
            # Check if within bounds
            if 0 <= sample_x < self.location_history.shape[1] and 0 <= sample_y < self.location_history.shape[0]:
                left_sum += self.location_history[sample_y, sample_x]
        
        # Sample right
        right_sum = 0.0
        for i in range(1, samples + 1):
            dist = i * (max_dist / samples)
            # Mix of forward and right vectors
            mix_factor = i / samples  # Gradually increase right component
            sample_x = int(x + (nvx * (1.0 - mix_factor) + right_vx * mix_factor) * dist)
            sample_y = int(y + (nvy * (1.0 - mix_factor) + right_vy * mix_factor) * dist)
            
            # Check if within bounds
            if 0 <= sample_x < self.location_history.shape[1] and 0 <= sample_y < self.location_history.shape[0]:
                right_sum += self.location_history[sample_y, sample_x]
        
        # Normalize weights
        total_sum = straight_sum + left_sum + right_sum
        if total_sum > 0:
            weights['straight'] = straight_sum / total_sum
            weights['left'] = left_sum / total_sum
            weights['right'] = right_sum / total_sum
        else:
            # Default weights if no history
            weights['straight'] = 0.6
            weights['left'] = 0.2
            weights['right'] = 0.2
        
        return weights
    
    def _adjust_pattern_weights(self, pattern, location_weights):
        """Adjust movement pattern weights based on location history
        
        Args:
            pattern: Base movement pattern
            location_weights: Weights from location history
            
        Returns:
            Adjusted pattern
        """
        # Create a copy of the pattern
        adjusted = pattern.copy()
        
        # Adjust straight probability based on location history
        straight_factor = location_weights['straight'] / 0.6  # Normalize relative to default
        adjusted['straight'] = pattern['straight'] * straight_factor
        
        # Adjust turn probabilities
        turn_total = pattern['slight_turn'] + pattern['sharp_turn']
        if turn_total > 0:
            # Calculate left vs right bias from location weights
            left_right_ratio = location_weights['left'] / max(0.001, location_weights['right'])
            
            # If strong bias to one side, increase turn probability in that direction
            if left_right_ratio > 1.5 or left_right_ratio < 0.67:
                turn_boost = min(1.5, max(1.0, abs(left_right_ratio - 1.0) * 0.5 + 1.0))
                adjusted['slight_turn'] = pattern['slight_turn'] * turn_boost
                adjusted['sharp_turn'] = pattern['sharp_turn'] * turn_boost
        
        # Normalize to ensure probabilities sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            for key in adjusted:
                adjusted[key] /= total
        
        return adjusted

    def _predict_straight_trajectory(self, current_pos, vx, vy, steps):
        """Predict a straight trajectory based on current position and velocity
        
        Args:
            current_pos: Current position (x, y)
            vx: Velocity in x direction
            vy: Velocity in y direction
            steps: Number of steps to predict
            
        Returns:
            List of predicted positions [(x1, y1), (x2, y2), ...]
        """
        trajectory = []
        x, y = current_pos
        
        for i in range(steps):
            # Add small random noise to make prediction more realistic
            noise_x = random.uniform(-0.5, 0.5) * min(1.0, abs(vx) * 0.1)
            noise_y = random.uniform(-0.5, 0.5) * min(1.0, abs(vy) * 0.1)
            
            # Update position based on velocity
            x = x + vx + noise_x
            y = y + vy + noise_y
            
            trajectory.append((x, y))
            
        return trajectory
    
    def _predict_turning_trajectory(self, current_pos, vx, vy, steps, turn_angle=15, smoothing=0.8):
        """Predict a turning trajectory
        
        Args:
            current_pos: Current position (x, y)
            vx: Velocity in x direction
            vy: Velocity in y direction
            steps: Number of steps to predict
            turn_angle: Angle to turn in degrees (positive=left, negative=right)
            smoothing: How smooth the turn should be (0-1, higher=smoother)
            
        Returns:
            List of predicted positions [(x1, y1), (x2, y2), ...]
        """
        trajectory = []
        x, y = current_pos
        
        # Convert velocity to speed and direction
        speed = math.sqrt(vx**2 + vy**2)
        angle = math.atan2(vy, vx)
        
        # Convert turn angle to radians
        turn_angle_rad = math.radians(turn_angle)
        
        # Total angle to turn over the trajectory
        total_turn = turn_angle_rad
        
        for i in range(steps):
            # Calculate turn for this step (apply more turn in the middle of trajectory)
            progress = i / (steps - 1) if steps > 1 else 0
            turn_weight = 4 * progress * (1 - progress)  # Parabolic weight, max at 0.5
            step_turn = total_turn * turn_weight * (1 - smoothing)
            
            # Update angle with turn
            angle += step_turn
            
            # Add small random noise
            noise_angle = random.uniform(-0.02, 0.02)
            angle += noise_angle
            
            # Calculate new velocity components
            new_vx = speed * math.cos(angle)
            new_vy = speed * math.sin(angle)
            
            # Update position
            x = x + new_vx
            y = y + new_vy
            
            trajectory.append((x, y))
            
        return trajectory
    
    def _sample_location_history(self, x, y, vx, vy):
        """Sample the location history in the direction of movement
        
        Args:
            x, y: Current position
            vx, vy: Velocity vector
            
        Returns:
            Dictionary with weights for different directions
        """
        # Default weights
        weights = {
            'straight': 1.0,
            'left': 0.5,
            'right': 0.5
        }
        
        # If no significant motion or no history data, return default weights
        if abs(vx) < 0.1 and abs(vy) < 0.1:
            return weights
            
        # Calculate direction angle
        angle = math.atan2(vy, vx)
        
        # Sample points ahead in different directions
        straight_samples = []
        left_samples = []
        right_samples = []
        
        # Distance to sample
        sample_dist = 30
        
        # Sample straight ahead
        for d in range(10, sample_dist, 5):
            sample_x = int(x + d * math.cos(angle))
            sample_y = int(y + d * math.sin(angle))
            
            if 0 <= sample_x < self.location_history.shape[1] and 0 <= sample_y < self.location_history.shape[0]:
                straight_samples.append(self.location_history[sample_y, sample_x])
        
        # Sample left
        left_angle = angle + math.pi/6  # 30 degrees left
        for d in range(10, sample_dist, 5):
            sample_x = int(x + d * math.cos(left_angle))
            sample_y = int(y + d * math.sin(left_angle))
            
            if 0 <= sample_x < self.location_history.shape[1] and 0 <= sample_y < self.location_history.shape[0]:
                left_samples.append(self.location_history[sample_y, sample_x])
        
        # Sample right
        right_angle = angle - math.pi/6  # 30 degrees right
        for d in range(10, sample_dist, 5):
            sample_x = int(x + d * math.cos(right_angle))
            sample_y = int(y + d * math.sin(right_angle))
            
            if 0 <= sample_x < self.location_history.shape[1] and 0 <= sample_y < self.location_history.shape[0]:
                right_samples.append(self.location_history[sample_y, sample_x])
        
        # Calculate average values
        straight_avg = sum(straight_samples) / max(len(straight_samples), 1)
        left_avg = sum(left_samples) / max(len(left_samples), 1)
        right_avg = sum(right_samples) / max(len(right_samples), 1)
        
        # Normalize to get weights
        total = straight_avg + left_avg + right_avg
        if total > 0:
            weights['straight'] = straight_avg / total
            weights['left'] = left_avg / total
            weights['right'] = right_avg / total
        
        return weights
    
    def _adjust_pattern_weights(self, pattern, location_weights):
        """Adjust movement pattern weights based on location history
        
        Args:
            pattern: Original movement pattern
            location_weights: Weights from location history
            
        Returns:
            Adjusted pattern
        """
        # Copy original pattern
        adjusted = pattern.copy()
        
        # Adjust straight probability based on location history
        straight_factor = location_weights['straight'] * 2  # Amplify the effect
        adjusted['straight'] = pattern['straight'] * straight_factor
        
        # Adjust turn probabilities
        turn_total = pattern['slight_turn'] + pattern['sharp_turn']
        if turn_total > 0:
            # Calculate left vs right bias from location weights
            left_weight = location_weights['left'] / (location_weights['left'] + location_weights['right'])
            right_weight = 1 - left_weight
            
            # Store these for use in trajectory generation
            self.left_turn_bias = left_weight
            self.right_turn_bias = right_weight
        
        # Normalize to ensure probabilities sum to 1
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total
        
        return adjusted
    
    def _get_turn_weights(self, location_weights, turn_type):
        """Get weights for left vs right turns
        
        Args:
            location_weights: Weights from location history
            turn_type: 'slight' or 'sharp'
            
        Returns:
            (left_weight, right_weight) tuple
        """
        # Calculate left vs right bias from location weights
        left_weight = location_weights['left'] / (location_weights['left'] + location_weights['right'])
        right_weight = 1 - left_weight
        
        # For sharp turns, exaggerate the bias
        if turn_type == 'sharp':
            # Exaggerate the bias for sharp turns
            left_weight = left_weight ** 0.7  # Reduce the exponent to make bias less extreme
            right_weight = right_weight ** 0.7
            
            # Renormalize
            total = left_weight + right_weight
            left_weight /= total
            right_weight /= total
        
        return left_weight, right_weight