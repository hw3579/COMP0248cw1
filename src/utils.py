
import torch
import numpy as np
import cv2
import torchvision.ops as ops
import math
from torch.nn import functional as F

# from dataloader import Comp0249Dataset

def draw_the_box(image, boxes, nms_threshold=0.45):
    """Draw bounding boxes on the image using NMS to reduce duplicate boxes."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Convert image to OpenCV-compatible format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Print box shape and an example grid element for debugging
    print(f"Box shape: {boxes.shape}")
    if boxes.size > 0 and len(boxes.shape) >= 3:
        print(f"Example grid element [0,0]: {boxes[0, 0, :]}")
    
    # Process grid boxes and collect predictions for NMS
    S_h, S_w = boxes.shape[:2]
    detection_count = 0
    confidence_threshold = 0.3
    
    all_boxes = []      # Format: [x1, y1, x2, y2]
    all_scores = []     # Confidence scores
    all_classes = []    # Class indices
    all_metadata = []   # Additional information such as class probability
    
    # Compute grid cell size
    cell_w = w / S_w
    cell_h = h / S_h

    # Step 1: Collect all potential boxes from each grid cell
    for i in range(S_h):
        for j in range(S_w):
            class_probs = boxes[i, j, :5]  # First 5 values are class probabilities
            x = boxes[i, j, 5]             # x coordinate (position 6)
            y = boxes[i, j, 6]             # y coordinate (position 7)
            w_box = boxes[i, j, 7]         # width (position 8)
            h_box = boxes[i, j, 8]         # height (position 9)
            confidence = boxes[i, j, 9]    # confidence (position 10)
            
            if confidence > confidence_threshold:
                # Coordinate transformation
                cx = (x + j) * cell_w
                cy = (y + i) * cell_h
                box_w = w_box * w
                box_h = h_box * h
                
                # Calculate top-left and bottom-right coordinates
                x1 = max(0, int(cx - box_w/2))
                y1 = max(0, int(cy - box_h/2))
                x2 = min(w - 1, int(cx + box_w/2))
                y2 = min(h - 1, int(cy + box_h/2))
                
                # Determine class (add 1 since classes are 1-indexed)
                class_idx = np.argmax(class_probs) + 1
                class_prob = class_probs[class_idx - 1]
                
                # Save box information and related metadata
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(confidence)
                all_classes.append(class_idx)
                all_metadata.append({
                    'class_prob': class_prob,
                    'class_idx': class_idx
                })
    
    print(f"Detected {len(all_boxes)} initial boxes")
    
    # Apply enhanced NMS
    if len(all_boxes) > 0:
        keep_indices, merged_boxes, merged_scores, merged_classes = enhanced_nms(
            all_boxes, all_scores, all_classes, 
            iou_threshold=0.2,  # Traditional NMS IoU threshold
            merge_threshold=0.01  # Merging threshold for boxes
        )
        
        print(f"After merging, {len(merged_boxes)} boxes remain")
        
        # Draw merged boxes on the image
        for i in range(len(merged_boxes)):
            detection_count += 1
            x1, y1, x2, y2 = [int(c) for c in merged_boxes[i]]
            class_idx = merged_classes[i]
            confidence = merged_scores[i]
            
            # Choose color based on class
            color = (0, 255, 0)  # Default green
            if class_idx == 1:
                color = (255, 0, 0)     # Blue for Car
            elif class_idx == 2:
                color = (0, 0, 255)     # Red for Pedestrian
            elif class_idx == 3:
                color = (255, 255, 0)   # Cyan for Bicyclist
            elif class_idx == 4:
                color = (255, 0, 255)   # Purple for Motorcycle/Scooter
            elif class_idx == 5:
                color = (0, 255, 255)   # Yellow for Truck/Bus
            
            # Draw rectangle on image
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Determine the maximum class probability if merged
            class_prob = 0.0
            if len(keep_indices) > 0:
                orig_indices = keep_indices[i] if isinstance(keep_indices[i], list) else [keep_indices[i]]
                probs = [all_metadata[idx]['class_prob'] for idx in orig_indices if idx < len(all_metadata)]
                class_prob = max(probs) if probs else 0.0
            
            # Set label text, indicating if the box was merged
            is_merged = "merged" if isinstance(keep_indices[i], list) and len(keep_indices[i]) > 1 else ""
            label = f"{is_merged} C{class_idx}: {class_prob:.2f} Conf:{confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    print(f"Drew {detection_count} boxes")
    return image
def enhanced_nms(boxes, scores, classes, iou_threshold=0.45, merge_threshold=0.25, distance_threshold=50):
    """
    Enhanced NMS: Apply standard NMS and then fuse overlapping boxes.
    
    Parameters:
        boxes: List of bounding box coordinates [x1, y1, x2, y2].
        scores: Confidence scores of each box.
        classes: Class indices of each box.
        iou_threshold: IoU threshold for standard NMS.
        merge_threshold: IoU threshold for merging boxes.
        distance_threshold: Center distance threshold for merging boxes.
        
    Returns:
        A tuple containing:
            - kept indices (merged indices list),
            - merged boxes as a numpy array,
            - merged scores as a numpy array,
            - merged classes as a numpy array.
    """
    if len(boxes) == 0:
        return [], [], [], []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # Perform standard NMS per class
    keep_indices = []
    unique_classes = np.unique(classes)
    
    for cls in unique_classes:
        cls_indices = np.where(classes == cls)[0]
        if len(cls_indices) == 0:
            continue
            
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]
        
        cls_boxes_tensor = torch.tensor(cls_boxes, dtype=torch.float32)
        cls_scores_tensor = torch.tensor(cls_scores, dtype=torch.float32)
        keep_tensor = ops.nms(cls_boxes_tensor, cls_scores_tensor, iou_threshold)
        
        keep_cls = [cls_indices[i] for i in keep_tensor.cpu().numpy()]
        keep_indices.extend(keep_cls)
    
    if len(keep_indices) <= 1:
        return keep_indices, boxes[keep_indices] if keep_indices else [], scores[keep_indices] if keep_indices else [], classes[keep_indices] if keep_indices else []
    
    kept_boxes = boxes[keep_indices]
    kept_scores = scores[keep_indices]
    kept_classes = classes[keep_indices]
    
    # Helper function to compute IoU between two boxes
    def calculate_iou(box1, box2):
        # Ensure box coordinates are in correct order
        x1_1, y1_1, x2_1, y2_1 = sorted(box1[:2]) + sorted(box1[2:])
        x1_2, y1_2, x2_2, y2_2 = sorted(box2[:2]) + sorted(box2[2:])
        
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
        
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    # Merge overlapping boxes
    is_merged = np.zeros(len(kept_boxes), dtype=bool)
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    merged_indices = []
    
    for i in range(len(kept_boxes)):
        if is_merged[i]:
            continue
        current_box = kept_boxes[i]
        current_score = kept_scores[i]
        current_class = kept_classes[i]
        cx1 = (current_box[0] + current_box[2]) / 2
        cy1 = (current_box[1] + current_box[3]) / 2
        
        overlaps = []
        for j in range(len(kept_boxes)):
            if i != j and not is_merged[j] and kept_classes[j] == current_class:
                iou = calculate_iou(current_box, kept_boxes[j])
                other_box = kept_boxes[j]
                cx2 = (other_box[0] + other_box[2]) / 2
                cy2 = (other_box[1] + other_box[3]) / 2
                distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if iou > merge_threshold or distance < distance_threshold:
                    overlaps.append(j)
                    
        if overlaps:
            # Mark all overlapping boxes as merged
            is_merged[i] = True
            for j in overlaps:
                is_merged[j] = True
            indices_to_merge = [keep_indices[i]] + [keep_indices[j] for j in overlaps]
            merged_indices.append(indices_to_merge)
            
            boxes_to_merge = [current_box] + [kept_boxes[j] for j in overlaps]
            scores_to_merge = [current_score] + [kept_scores[j] for j in overlaps]
            
            x1 = min(box[0] for box in boxes_to_merge)
            y1 = min(box[1] for box in boxes_to_merge)
            x2 = max(box[2] for box in boxes_to_merge)
            y2 = max(box[3] for box in boxes_to_merge)
            avg_score = sum(scores_to_merge) / len(scores_to_merge)
            
            merged_boxes.append([x1, y1, x2, y2])
            merged_scores.append(avg_score)
            merged_classes.append(current_class)
        elif not is_merged[i]:
            merged_boxes.append(current_box)
            merged_scores.append(current_score)
            merged_classes.append(current_class)
            merged_indices.append([keep_indices[i]])
            is_merged[i] = True
    
    return merged_indices, np.array(merged_boxes), np.array(merged_scores), np.array(merged_classes)


def segmentation_to_yolov3_1(label, Sx, Sy, num_classes=20, B=2, scale=1):
    """
    Convert a segmentation label (H, W) to a YOLO-format label (Sy, Sx, num_classes + B*5).

    Parameters:
        label: torch.Tensor, segmentation label with shape (H, W). Pixels with value 0 indicate background,
               while nonzero values correspond to class indices (preferably consecutive).
        Sx: Horizontal grid size for YOLO (number of grid cells along width).
        Sy: Vertical grid size for YOLO (number of grid cells along height).
        num_classes: Number of classes.
        B: Number of bounding boxes predicted per grid cell (typically only the first is used for annotation, others remain 0).

    Returns:
        yolo_label: torch.Tensor with shape (Sy, Sx, num_classes + B*5). Format:
                    - The first num_classes dimensions indicate the target class.
                    - The following B*5 dimensions represent each bounding box as [cx, cy, w, h, confidence].
    """
    import math
    Sx = math.ceil(Sx)
    Sy = math.ceil(Sy)
    
    # Convert label to numpy array
    label_np = label.cpu().numpy().astype(np.uint8)
    if len(label_np.shape) == 3:
        label_np = label_np[0]
    H, W = label_np.shape
    yolo_label = np.zeros((Sy, Sx, B * 5 + num_classes), dtype=np.float32)

    # Use OpenCV connected components analysis (assume background is 0)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(label_np, connectivity=8)
    stats = np.concatenate([stats, centroids], axis=1)  # Append centroid coordinates to stats
    
    # Filter out regions with area smaller than 1e3 (background not counted)
    filtered_stats = stats[stats[:, -3] >= (1e3 // scale)].astype(np.int32)
    
    # Iterate through each object (excluding background)
    for i in range(1, len(filtered_stats)):
        x, y, w, h, area, cx, cy = filtered_stats[i]
        # Normalize bounding box parameters relative to the full image
        norm_cx = cx / W
        norm_cy = cy / H
        norm_w = w / W
        norm_h = h / H

        # Determine the object class by taking the most frequently occurring class in the region (ignore background)
        region = label_np[y:y+h, x:x+w]
        region_flat = region[region > 0]
        if len(region_flat) == 0:
            continue
        class_id = np.bincount(region_flat).argmax()

        # Determine in which grid cell the object's center falls
        grid_x = int(norm_cx * Sx)
        grid_y = int(norm_cy * Sy)
        grid_x = min(grid_x, Sx - 1)
        grid_y = min(grid_y, Sy - 1)

        # Set the class and bounding box information for the grid cell (only fill in the first bounding box; others remain 0)
        yolo_label[grid_y, grid_x, class_id - 1] = 1
        # Compute the offsets relative to the grid cell
        rel_cx = norm_cx * Sx - grid_x
        rel_cy = norm_cy * Sy - grid_y
        
        # Store the relative coordinates and set confidence to 1.0
        yolo_label[grid_y, grid_x, num_classes:num_classes+5] = np.array([rel_cx, rel_cy, norm_w, norm_h, 1.0], dtype=np.float32)

    # Convert back to a torch.Tensor
    return torch.from_numpy(yolo_label)

def yolo_loss(predictions, targets, Sx, Sy, B=1, C=5, lambda_coord=5, lambda_noobj=0.5, gamma=2.0, alpha=0.25, imbalanced_mode=False):
    """Modified YOLO loss function with support for imbalanced data
    
    Parameters:
        predictions: Model predictions
        targets: Target labels
        Sx, Sy: YOLO grid dimensions
        B: Number of bounding boxes per grid cell
        C: Number of classes
        lambda_coord: Coordinate loss weight
        lambda_noobj: Confidence loss weight for cells without objects
        gamma: Focal Loss gamma parameter
        alpha: Focal Loss alpha parameter
        imbalanced_mode: Enable imbalanced data handling mode
    """
    batch = predictions.shape[0]
    
    # Get predictions and targets
    pred_class = predictions[..., :C]
    target_class = targets[..., :C]
    
    # Create object masks
    obj_mask = (targets[..., C+4] > 0.5)
    noobj_mask = ~obj_mask
    
    # Calculate class weights if imbalanced mode is enabled
    if imbalanced_mode:
        class_counts = []
        for c in range(C):
            class_mask = target_class[..., c] > 0.5
            class_count = (class_mask & obj_mask).sum().float() + 1e-6
            class_counts.append(class_count)
        
        class_counts = torch.tensor(class_counts, device=predictions.device)
        total_count = class_counts.sum()
        
        # Class weights are inversely proportional to class frequency
        class_weights = total_count / (C * class_counts)
        class_weights = class_weights / class_weights.sum() * C
    else:
        class_weights = torch.ones(C, device=predictions.device)
    
    # 1. Class loss calculation with Focal Loss
    mse_loss = F.mse_loss(pred_class, target_class, reduction='none')
    pt = torch.exp(-mse_loss)
    
    pos_mask = (target_class > 0.5)
    neg_mask = ~pos_mask
    
    # Apply different alpha weights for different classes in imbalanced mode
    if imbalanced_mode:
        alpha_weight = torch.zeros_like(target_class)
        for c in range(C):
            alpha_weight[..., c] = pos_mask[..., c] * (alpha * class_weights[c]) + neg_mask[..., c] * (1-alpha)
    else:
        alpha_weight = pos_mask * alpha + neg_mask * (1-alpha)
    
    # Apply Focal Loss weight
    focal_weight = alpha_weight * (1 - pt) ** gamma
    
    # Only calculate class loss for cells with objects
    obj_expanded = obj_mask.unsqueeze(-1).expand_as(pred_class)
    focal_loss = focal_weight * mse_loss * obj_expanded
    class_loss = focal_loss.sum()
    
    # 2. Coordinate loss calculation
    pred_xy = predictions[..., C:C+2]
    pred_wh = predictions[..., C+2:C+4]
    target_xy = targets[..., C:C+2]
    target_wh = targets[..., C+2:C+4]
    
    # Apply mask to object cells
    obj_expanded_xy = obj_mask.unsqueeze(-1).expand_as(pred_xy)
    obj_expanded_wh = obj_mask.unsqueeze(-1).expand_as(pred_wh)
    
    # Calculate coordinate losses
    xy_loss = F.mse_loss(
        pred_xy * obj_expanded_xy, 
        target_xy * obj_expanded_xy, 
        reduction='mean'
    )
    
    wh_loss = F.mse_loss(
        pred_wh * obj_expanded_wh, 
        target_wh * obj_expanded_wh, 
        reduction='mean'
    )

        # 
    # wh_loss = F.mse_loss(
    #     torch.sqrt(torch.abs(pred_wh) + 1e-6) * obj_expanded_wh, 
    #     torch.sqrt(torch.abs(target_wh) + 1e-6) * obj_expanded_wh, 
    #     reduction='mean'
    # )
    
    coord_loss = lambda_coord * (xy_loss + wh_loss)
    
    # 3. Confidence loss calculation
    if imbalanced_mode and obj_mask.sum() > 0:
        # Get class indices for each object
        target_class_idx = torch.argmax(target_class, dim=-1)
        
        # Create confidence weight matrix
        conf_weights = torch.ones_like(targets[..., C+4])
        
        # Set weights based on class frequency
        for i in range(batch):
            for y in range(Sy):
                for x in range(Sx):
                    if obj_mask[i, y, x]:
                        cls_idx = target_class_idx[i, y, x].item()
                        conf_weights[i, y, x] = class_weights[cls_idx]
        
        obj_conf_loss = F.mse_loss(
            predictions[..., C+4][obj_mask], 
            targets[..., C+4][obj_mask], 
            reduction='mean'
        ) * conf_weights[obj_mask].mean()
    else:
        obj_conf_loss = F.mse_loss(
            predictions[..., C+4][obj_mask], 
            targets[..., C+4][obj_mask], 
            reduction='mean'
        )
    
    # Adjust weight for no-object loss
    lambda_noobj_adjusted = lambda_noobj
    if imbalanced_mode:
        # Further reduce no-object loss weight if objects are rare
        obj_ratio = obj_mask.sum().float() / obj_mask.numel()
        if obj_ratio < 0.01:  # Less than 1%
            lambda_noobj_adjusted = lambda_noobj * 0.5
    
    noobj_conf_loss = lambda_noobj_adjusted * F.mse_loss(
        predictions[..., C+4][noobj_mask], 
        targets[..., C+4][noobj_mask], 
        reduction='mean'
    )
    
    # 4. Total loss calculation
    total_loss = class_loss + coord_loss + obj_conf_loss + noobj_conf_loss
    
    if imbalanced_mode:
        # No additional normalization in imbalanced mode
        return total_loss
    else:
        # Original normalization method
        num_obj = obj_mask.sum().float() + 1e-6
        return total_loss / (batch * num_obj)
    

def compute_iou_yolo(output_yolo, labels_yolo):
    """
    Compute IoU between YOLO outputs and ground truth labels.
    
    Parameters:
        output_yolo: torch.Tensor, shape (batch_size, Sx, Sy, num_classes + B*5)
        labels_yolo: torch.Tensor, shape (batch_size, Sx, Sy, num_classes + B*5)
    
    Returns:
        mean_iou: float, average IoU value.
        accuracy: float, classification accuracy.
    """
    batch_size = output_yolo.shape[0]
    num_classes = 5  # assume 5 classes based on other parts of the code
    
    # Extract class predictions and ground truth targets
    pred_class = output_yolo[..., :num_classes]
    target_class = labels_yolo[..., :num_classes]
    
    # Extract bounding box predictions (cx, cy, w, h, conf)
    pred_boxes = output_yolo[..., num_classes:num_classes+5]
    target_boxes = labels_yolo[..., num_classes:num_classes+5]
    
    # Compute classification accuracy for grids that contain objects
    target_obj_mask = (target_boxes[..., 4] > 0.5)
    if target_obj_mask.sum() > 0:
        pred_class_idx = torch.argmax(pred_class, dim=-1)
        target_class_idx = torch.argmax(target_class, dim=-1)
        correct = (pred_class_idx[target_obj_mask] == target_class_idx[target_obj_mask]).float()
        accuracy = correct.sum() / (target_obj_mask.sum() + 1e-6)
    else:
        accuracy = torch.tensor(0.0, device=output_yolo.device)
    
    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
    
    target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
    
    # Compute intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h
    
    # Compute union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union = pred_area + target_area - intersection + 1e-6
    
    # Compute IoU
    ious = intersection / union
    
    if target_obj_mask.sum() > 0:
        mean_iou = ious[target_obj_mask].mean()
    else:
        mean_iou = torch.tensor(0.0, device=output_yolo.device)
    
    return mean_iou.item(), accuracy.item()