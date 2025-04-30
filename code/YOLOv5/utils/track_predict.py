import cv2
import numpy as np
import torch
from collections import defaultdict, deque
import random
from scipy.ndimage import gaussian_filter1d
import math

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
        
        return trajectories
    
    def _sample_location_history(self, x, y, vx, vy):
        """Sample the location history in the direction of movement
        
        Args:
            x, y: Current position
            vx, vy: Velocity vector
            
        Returns:
            Dictionary with weights for different directions
        """
        # Convert to integers
        x, y = int(x), int(y)
        
        # Ensure coordinates are within bounds
        h, w = self.location_history.shape
        if not (0 <= x < w and 0 <= y < h):
            return {'straight': 1.0, 'left': 0.5, 'right': 0.5}
        
        # Calculate direction angle
        angle = math.atan2(vy, vx)
        
        # Sample points in different directions
        samples = {
            'straight': 0.0,
            'left': 0.0,
            'right': 0.0
        }
        
        # Sample points ahead (straight)
        straight_samples = []
        for dist in range(10, 100, 10):
            sample_x = int(x + dist * math.cos(angle))
            sample_y = int(y + dist * math.sin(angle))
            if 0 <= sample_x < w and 0 <= sample_y < h:
                straight_samples.append(self.location_history[sample_y, sample_x])
        
        if straight_samples:
            samples['straight'] = np.mean(straight_samples)
        
        # Sample points to the left
        left_samples = []
        for dist in range(10, 100, 10):
            sample_x = int(x + dist * math.cos(angle + math.pi/6))
            sample_y = int(y + dist * math.sin(angle + math.pi/6))
            if 0 <= sample_x < w and 0 <= sample_y < h:
                left_samples.append(self.location_history[sample_y, sample_x])
        
        if left_samples:
            samples['left'] = np.mean(left_samples)
        
        # Sample points to the right
        right_samples = []
        for dist in range(10, 100, 10):
            sample_x = int(x + dist * math.cos(angle - math.pi/6))
            sample_y = int(y + dist * math.sin(angle - math.pi/6))
            if 0 <= sample_x < w and 0 <= sample_y < h:
                right_samples.append(self.location_history[sample_y, sample_x])
        
        if right_samples:
            samples['right'] = np.mean(right_samples)
        
        # Normalize samples
        total = sum(samples.values())
        if total > 0:
            for key in samples:
                samples[key] /= total
        else:
            # Default if no samples
            samples = {'straight': 0.6, 'left': 0.2, 'right': 0.2}
        
        return samples
    
    def _adjust_pattern_weights(self, pattern, location_weights):
        """Adjust pattern weights based on location history
        
        Args:
            pattern: Original movement pattern
            location_weights: Weights from location history
            
        Returns:
            Adjusted pattern
        """
        # Create a copy of the pattern
        adjusted = pattern.copy()
        
        # Adjust straight probability based on location history
        straight_factor = 1.0 + location_weights['straight'] * 2.0
        adjusted['straight'] = min(0.9, pattern['straight'] * straight_factor)
        
        # Calculate remaining probability
        remaining = 1.0 - adjusted['straight']
        
        # Distribute remaining probability between slight and sharp turns
        turn_ratio = pattern['slight_turn'] / (pattern['slight_turn'] + pattern['sharp_turn'] + 1e-10)
        
        adjusted['slight_turn'] = remaining * turn_ratio
        adjusted['sharp_turn'] = remaining * (1.0 - turn_ratio)
        
        return adjusted
    
    def _get_turn_weights(self, location_weights, turn_type):
        """Get weights for left and right turns
        
        Args:
            location_weights: Weights from location history
            turn_type: 'slight' or 'sharp'
            
        Returns:
            (left_weight, right_weight) tuple
        """
        # Base weights
        left = location_weights['left']
        right = location_weights['right']
        
        # Ensure they sum to 1.0
        total = left + right
        if total > 0:
            left /= total
            right /= total
        else:
            left, right = 0.5, 0.5
        
        # For sharp turns, exaggerate the difference
        if turn_type == 'sharp':
            # Apply power function to increase contrast
            left = left ** 0.7
            right = right ** 0.7
            
            # Renormalize
            total = left + right
            left /= total
            right /= total
        
        return left, right
    
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
        
        # Convert angle to radians
        turn_angle_rad = math.radians(turn_angle)
        
        # Calculate initial direction angle
        initial_angle = math.atan2(vy, vx)
        
        # Calculate speed (magnitude of velocity)
        speed = math.sqrt(vx**2 + vy**2)
        
        for i in range(1, steps + 1):
            # Calculate progress through the turn (0 to 1)
            progress = min(1.0, i / (steps * smoothing))
            
            # Calculate current angle with gradual turn
            current_angle = initial_angle + turn_angle_rad * progress
            
            # Calculate velocity components at this angle
            current_vx = speed * math.cos(current_angle)
            current_vy = speed * math.sin(current_angle)
            
            # Add small random noise
            noise_x = random.gauss(0, 0.5) * i
            noise_y = random.gauss(0, 0.5) * i
            
            # Apply damping for more realistic prediction
            damping = 0.95 ** i
            
            # Calculate next position
            next_x = x + current_vx * i * damping + noise_x
            next_y = y + current_vy * i * damping + noise_y
            
            trajectory.append((next_x, next_y))
        
        return trajectory