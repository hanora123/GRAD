import torch
import numpy as np
from .trajectory import TrajectoryPredictor
from .general import scale_coords, xyxy2xywh
from .plots import Annotator, colors

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
            
        # Add tracking IDs column
        dets_with_ids = torch.cat([dets, torch.zeros((len(dets), 1), device=dets.device)], dim=1)
        
        # For first frame, assign new IDs to all detections
        if self.frame_count == 0:
            dets_with_ids[:, -1] = torch.arange(len(dets), device=dets.device)
            self.next_id = len(dets)
            return dets_with_ids
            
        # Calculate IoU between current and previous detections
        current_boxes = dets[:, :4]
        prev_boxes = torch.tensor([list(track['positions'][-1]) for track in self.trajectory_predictor.tracks.values()])
        
        if len(prev_boxes) > 0:
            iou_matrix = box_iou(current_boxes, prev_boxes)
            matched_indices = torch.nonzero(iou_matrix > self.iou_threshold)
            
            # Assign existing IDs to matched detections
            for i, j in matched_indices:
                track_id = list(self.trajectory_predictor.tracks.keys())[j]
                dets_with_ids[i, -1] = track_id
                
        # Assign new IDs to unmatched detections
        unmatched_mask = dets_with_ids[:, -1] == 0
        num_unmatched = unmatched_mask.sum()
        new_ids = torch.arange(self.next_id, self.next_id + num_unmatched, device=dets.device)
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
                for i in range(len(points) - 1):
                    annotator.line([points[i], points[i+1]], color=colors(cls_id, True))
                    
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