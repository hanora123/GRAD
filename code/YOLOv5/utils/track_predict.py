import torch
import numpy as np
from .trajectory import TrajectoryPredictor
from .general import scale_boxes, xyxy2xywh
from .plots import Annotator, colors
import cv2

class TrackPredictor:
    def __init__(self, history_size=30, future_steps=10):
        self.trajectory_predictor = TrajectoryPredictor(history_size, future_steps)
        self.frame_count = 0
        self.next_id = 0
        self.iou_threshold = 0.3
        
    def _assign_ids(self, dets):
        """Assign tracking IDs to detections using IoU matching.
        
        Args:
            dets (torch.Tensor): Detections tensor [x1, y1, x2, y2, conf, cls]
            
        Returns:
            torch.Tensor: Detections with tracking IDs [x1, y1, x2, y2, conf, cls, obj_id]
        """
        if len(dets) == 0:
            return torch.empty((0, 7), device=dets.device)
            
        # Add tracking IDs column - ensure it's the same dtype as dets
        id_column = torch.zeros((len(dets), 1), device=dets.device, dtype=dets.dtype)
        dets_with_ids = torch.cat([dets, id_column], dim=1)
        
        # For first frame, assign new IDs to all detections
        if self.frame_count == 0:
            dets_with_ids[:, -1] = torch.arange(len(dets), device=dets.device, dtype=dets.dtype)
            self.next_id = len(dets)
            return dets_with_ids
            
        # Check if there are any tracks
        if len(self.trajectory_predictor.tracks) == 0:
            # No tracks yet, assign new IDs to all detections
            dets_with_ids[:, -1] = torch.arange(self.next_id, self.next_id + len(dets), 
                                               device=dets.device, dtype=dets.dtype)
            self.next_id += len(dets)
            return dets_with_ids
        
        # Calculate IoU between current and previous detections
        current_boxes = dets[:, :4]
        
        # Get previous boxes, ensuring they are in the correct format
        try:
            # Convert list to tensor and ensure it's on the same device as dets
            prev_boxes_list = [list(track['positions'][-1]) for track in self.trajectory_predictor.tracks.values()]
            prev_boxes = torch.tensor(prev_boxes_list, device=dets.device)
            
            # Check if prev_boxes has the right shape
            if prev_boxes.shape[1] == 2:  # If only center points [x, y]
                # Convert center points to boxes [x1, y1, x2, y2]
                # Using a small box size around each center point
                box_size = 20  # Adjust this value as needed
                x, y = prev_boxes[:, 0], prev_boxes[:, 1]
                prev_boxes = torch.stack([
                    x - box_size/2, y - box_size/2,  # x1, y1
                    x + box_size/2, y + box_size/2   # x2, y2
                ], dim=1)
            
            if len(prev_boxes) > 0:
                iou_matrix = box_iou(current_boxes, prev_boxes)
                matched_indices = torch.nonzero(iou_matrix > self.iou_threshold)
                
                # Assign existing IDs to matched detections
                for i, j in matched_indices:
                    track_id = list(self.trajectory_predictor.tracks.keys())[j]
                    # Convert track_id to the same dtype as dets
                    dets_with_ids[i, -1] = float(track_id)  # Convert to float if dets is float
        except Exception as e:
            # If there's any error in matching, just assign new IDs
            print(f"Error in tracking: {e}")
        
        # Assign new IDs to unmatched detections
        unmatched_mask = dets_with_ids[:, -1] == 0
        num_unmatched = int(unmatched_mask.sum().item())
        
        # Create new IDs with the same dtype as dets
        new_ids = torch.arange(self.next_id, self.next_id + num_unmatched, 
                              device=dets.device, dtype=dets.dtype)
        
        # Assign new IDs to unmatched detections
        if num_unmatched > 0:
            dets_with_ids[unmatched_mask, -1] = new_ids
            self.next_id += num_unmatched
        
        return dets_with_ids
        
    def process_frame(self, dets, img):
        """Process frame detections and predict trajectories.
        
        Args:
            dets (torch.Tensor): YOLOv5 detections tensor [x1, y1, x2, y2, conf, cls]
            img (numpy.ndarray): Original image
            
        Returns:
            numpy.ndarray: Annotated image with trajectories
        """
        # Assign tracking IDs
        dets_with_ids = self._assign_ids(dets)
        
        # Update trajectory predictor
        self.trajectory_predictor.update_tracks(dets_with_ids, self.frame_count)
        
        # Draw detections and trajectories
        annotator = Annotator(img.copy())
        
        for det in dets_with_ids:
            xyxy = det[:4].cpu().numpy()
            conf = det[4].cpu().numpy()
            cls_id = int(det[5])
            obj_id = int(det[6])
            
            # Draw bounding box
            label = f'{obj_id} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(cls_id, True))
            
            # Predict and draw trajectory
            trajectory = self.trajectory_predictor.predict_trajectory(obj_id, self.frame_count)
            if trajectory is not None:
                points = trajectory.astype(np.int32)
                
                # Draw historical trajectory (past positions) with a different color
                if 'positions' in self.trajectory_predictor.tracks.get(obj_id, {}):
                    past_positions = np.array(self.trajectory_predictor.tracks[obj_id]['positions'])
                    if len(past_positions) > 1:
                        past_points = past_positions.astype(np.int32)
                        for i in range(len(past_points) - 1):
                            # Draw historical trajectory with a different color (e.g., blue)
                            cv2.line(annotator.im, 
                                    (past_points[i][0], past_points[i][1]), 
                                    (past_points[i+1][0], past_points[i+1][1]), 
                                    (255, 0, 0),  # Blue for past trajectory
                                    thickness=2)
                
                # Draw predicted future trajectory
                for i in range(len(points) - 1):
                    # Draw future trajectory with the class color
                    cv2.line(annotator.im, 
                            (points[i][0], points[i][1]), 
                            (points[i+1][0], points[i+1][1]), 
                            colors(cls_id, True),  # Use class color for future trajectory
                            thickness=2)
                    
                # Draw dots at each predicted point for better visibility
                for point in points:
                    cv2.circle(annotator.im, (point[0], point[1]), 3, colors(cls_id, True), -1)
                    
        self.frame_count += 1
        self.trajectory_predictor.cleanup_old_tracks(self.frame_count)
        
        return annotator.result()
        
def box_iou(box1, box2):
    """Calculate IoU between two sets of boxes.
    
    Args:
        box1 (torch.Tensor): First set of boxes (N, 4)
        box2 (torch.Tensor): Second set of boxes (M, 4)
        
    Returns:
        torch.Tensor: IoU matrix (N, M)
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    return inter / (area1[:, None] + area2 - inter)