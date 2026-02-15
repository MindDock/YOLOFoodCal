"""
Utility Functions Module
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR) or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image {image_path}")
        return None
    
    return img


def load_image_batch(image_paths: List[str], resize_to: Optional[tuple] = None) -> List[np.ndarray]:
    """
    Load multiple images
    
    Args:
        image_paths: List of image paths
        resize_to: Optional (width, height) to resize to
        
    Returns:
        List of images
    """
    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        img = load_image(path)
        if img is not None:
            if resize_to:
                img = cv2.resize(img, resize_to)
            images.append(img)
    return images


def save_json(data: dict, output_path: str, indent: int = 2):
    """
    Save data as JSON
    
    Args:
        data: Dictionary to save
        output_path: Output file path
        indent: JSON indentation
    """
    import json
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(json_path: str) -> dict:
    """
    Load JSON data
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary
    """
    import json
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str):
    """
    Ensure directory exists
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp as string"""
    return time.strftime("%Y%m%d_%H%M%S")


def resize_keep_aspect(
    image: np.ndarray,
    target_size: int = 640,
    interpolation: int = cv2.INTER_LINEAR
) -> tuple:
    """
    Resize image while keeping aspect ratio
    
    Args:
        image: Input image
        target_size: Target size for longest edge
        interpolation: Interpolation method
        
    Returns:
        Tuple of (resized_image, scale_factor, padding)
    """
    h, w = image.shape[:2]
    
    # Calculate scale
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
    else:
        scale = 1.0
    
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Pad to square
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded, scale, (pad_w, pad_h)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def nms_boxes(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: List of [x1, y1, x2, y2]
        scores: List of confidence scores
        iou_threshold: IoU threshold
        
    Returns:
        List of indices to keep
    """
    if not boxes:
        return []
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = [calculate_iou(boxes[current], boxes[i]) for i in indices[1:]]
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][np.array(ious) < iou_threshold]
    
    return keep


def format_time(seconds: float) -> str:
    """
    Format seconds as human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def get_file_list(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get list of files in directory
    
    Args:
        directory: Directory path
        extensions: List of extensions to filter (e.g., ['.jpg', '.png'])
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(f"*{ext}"))
        files.extend(Path(directory).glob(f"*{ext.upper()}"))
    
    return [str(f) for f in sorted(files)]


def download_model(model_name: str, cache_dir: str = "~/.cache/yolo") -> str:
    """
    Download YOLO model from Ultralytics
    
    Args:
        model_name: Model name (e.g., 'yolo26n-seg.pt')
        cache_dir: Cache directory
        
    Returns:
        Path to downloaded model
    """
    from ultralytics import YOLO
    
    cache_path = os.path.expanduser(cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    
    model_path = os.path.join(cache_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        # This will download automatically when creating YOLO model
        model = YOLO(model_name)
        # The model is cached, find its path
        import shutil
        # Model will be downloaded to cache
        model_path = model.model_path if hasattr(model, 'model_path') else model_name
    
    return model_path


class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"{self.name}: {format_time(self.elapsed)}")
    
    def get_elapsed(self) -> float:
        if self.start_time is None:
            return 0
        if self.elapsed > 0:
            return self.elapsed
        return time.time() - self.start_time
