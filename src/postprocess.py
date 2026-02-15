"""
Post-processing Module - NMS, Mask Processing, etc.
"""

import numpy as np
from typing import List, Tuple, Optional

import cv2


def process_masks(
    masks: np.ndarray,
    orig_shape: Tuple[int, int],
    target_shape: Tuple[int, int] = (640, 640)
) -> List[np.ndarray]:
    """
    Process and resize segmentation masks
    
    Args:
        masks: Array of masks (N, H, W)
        orig_shape: Original image shape (H, W)
        target_shape: Model input shape
        
    Returns:
        List of resized binary masks
    """
    if masks is None or len(masks) == 0:
        return []
    
    processed = []
    orig_h, orig_w = orig_shape
    target_h, target_w = target_shape
    
    # Calculate padding
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    for mask in masks:
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            mask = mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
        
        # Resize to original size
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Binarize
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        processed.append(mask_binary)
    
    return processed


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Apply Non-Maximum Suppression
    
    Args:
        boxes: Array of boxes (N, 4) in format [x1, y1, x2, y2]
        scores: Array of scores (N,)
        iou_threshold: IoU threshold
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        iou = compute_iou(boxes[i], boxes[order[1:]])
        
        # Keep boxes with IoU below threshold
        order = order[1:][iou < iou_threshold]
    
    return keep


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between one box and multiple boxes
    
    Args:
        box: Single box [x1, y1, x2, y2]
        boxes: Array of boxes (N, 4)
        
    Returns:
        Array of IoU values
    """
    # Compute intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute union
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    
    return intersection / np.maximum(union, 1e-6)


def filter_by_confidence(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    masks: Optional[np.ndarray],
    conf_threshold: float = 0.5
) -> Tuple:
    """
    Filter detections by confidence threshold
    
    Args:
        boxes: Array of boxes (N, 4)
        scores: Array of scores (N,)
        classes: Array of class indices (N,)
        masks: Array of masks (N, H, W) or None
        conf_threshold: Confidence threshold
        
    Returns:
        Filtered boxes, scores, classes, masks
    """
    mask = scores >= conf_threshold
    
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    
    if masks is not None:
        masks = masks[mask]
    
    return boxes, scores, classes, masks


def scale_boxes(
    boxes: np.ndarray,
    scale_factor: float,
    pad: Tuple[int, int] = (0, 0),
    orig_shape: Tuple[int, int] = None
) -> np.ndarray:
    """
    Scale boxes from model input size to original image size
    
    Args:
        boxes: Array of boxes (N, 4)
        scale_factor: Scale factor
        pad: Padding (pad_w, pad_h)
        orig_shape: Original shape (H, W) - if provided, use this instead
        
    Returns:
        Scaled boxes
    """
    if orig_shape is not None:
        # Use original shape directly
        h, w = orig_shape
        # Boxes are already in original coordinates if model handles this
        return boxes
    
    # Remove padding and scale
    boxes[:, [0, 2]] -= pad[0]  # x coordinates
    boxes[:, [1, 3]] -= pad[1]  # y coordinates
    boxes /= scale_factor
    
    return boxes


def decode_masks(
    masks: np.ndarray,
    orig_shape: Tuple[int, int],
    threshold: float = 0.5
) -> List[np.ndarray]:
    """
    Decode and resize masks
    
    Args:
        masks: Raw mask predictions
        orig_shape: Original image shape (H, W)
        threshold: Binarization threshold
        
    Returns:
        List of binary masks
    """
    h, w = orig_shape
    decoded = []
    
    for mask in masks:
        # Resize to original size
        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Binarize
        mask_binary = (mask_resized > threshold).astype(np.uint8)
        decoded.append(mask_binary)
    
    return decoded


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """
    Convert mask to polygon contour
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        List of contour points
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return []
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    # Convert to list
    polygon = approx.reshape(-1, 2).tolist()
    
    return polygon


def polygon_to_mask(polygon: List[List[int]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert polygon contour to mask
    
    Args:
        polygon: List of [x, y] points
        shape: Mask shape (H, W)
        
    Returns:
        Binary mask
    """
    mask = np.zeros(shape, dtype=np.uint8)
    
    if len(polygon) == 0:
        return mask
    
    polygon = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    
    return mask


def compute_mask_area(mask: np.ndarray) -> int:
    """
    Compute pixel area of mask
    
    Args:
        mask: Binary mask
        
    Returns:
        Number of pixels
    """
    return np.sum(mask > 0)


def compute_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Compute area of bounding box
    
    Args:
        bbox: (x1, y1, x2, y2)
        
    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two masks
    
    Args:
        mask1: Binary mask
        mask2: Binary mask
        
    Returns:
        IoU value
    """
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))
    
    return intersection / union if union > 0 else 0.0
