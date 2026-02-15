"""
Food Detector Module - YOLO26 Model Wrapper
"""

import os
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import cv2
from pathlib import Path


@dataclass
class DetectionResult:
    """Single detection result"""
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    mask: Optional[np.ndarray] = None  # Segmentation mask
    
    def __repr__(self):
        return f"DetectionResult({self.class_name}, conf={self.confidence:.2f})"


class FoodDetector:
    """
    YOLO26 Food Detector with Segmentation Support
    
    Supports both PyTorch (.pt) and ONNX (.onnx) formats.
    """
    
    def __init__(
        self,
        model_path: str = "yolo26n-seg.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        verbose: bool = False
    ):
        """
        Initialize Food Detector
        
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            imgsz: Input image size
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps')
            verbose: Enable verbose output
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._imgsz_override = imgsz != 640  # Track if user explicitly set imgsz
        self.verbose = verbose
        
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
            self.ultralytics_available = True
        except ImportError:
            self.ultralytics_available = False
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Class names (will be updated when model loads)
        self.class_names: List[str] = []

        # Load model
        self._load_model()

    
    def _load_model(self):
        """Load YOLO model"""
        model_path = self.model_path
        
        # Check if model exists
        if not os.path.exists(model_path):
            # Try to download pretrained model
            if self.verbose:
                print(f"Model not found at {model_path}, attempting to download...")
            # Use default YOLO26n-seg from ultralytics
            model_path = "yolo26n-seg.pt"
        
        if self.verbose:
            print(f"Loading model from {model_path}...")
        
        self.model = None
        self.is_segmentation = "-seg" in os.path.basename(model_path).lower()
        
        if self.ultralytics_available:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            
            # Get class names from model
            names_dict = {}
            if hasattr(self.model, 'names'):
                names_dict = self.model.names
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                names_dict = self.model.model.names
            
            if names_dict:
                # Convert dict to list ensuring order
                if isinstance(names_dict, dict):
                    self.class_names = [names_dict[i] for i in sorted(names_dict.keys())]
                else:
                    self.class_names = list(names_dict)
            else:
                # Default COCO classes (will be updated with custom classes)
                self.class_names = self._get_default_classes()
        
        # Auto-detect training imgsz from model metadata if user didn't override
        if not self._imgsz_override and hasattr(self.model, 'overrides'):
            train_imgsz = self.model.overrides.get('imgsz')
            if train_imgsz and isinstance(train_imgsz, int):
                self.imgsz = train_imgsz
                if self.verbose:
                    print(f"Auto-detected training imgsz: {train_imgsz}")

        if self.verbose:
            print(f"Model loaded. Segmentation: {self.is_segmentation}")
            print(f"Device: {self.device}")
            print(f"Inference imgsz: {self.imgsz}")
            print(f"Classes: {len(self.class_names)}")
    
    def _get_default_classes(self) -> List[str]:
        """Get default COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def set_classes(self, class_names: List[str]):
        """Set custom class names for the model"""
        self.class_names = class_names
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        return_masks: bool = True
    ) -> List[DetectionResult]:
        """
        Detect food items in image
        
        Args:
            image: Image path or numpy array (BGR format)
            return_masks: Return segmentation masks if available
            
        Returns:
            List of DetectionResult objects
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            save=False,
            augment=False,
            agnostic_nms=False,
            retina_masks=False
        )
        
        # Parse results
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Get boxes
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Get masks if available and requested
                masks = None
                if return_masks and self.is_segmentation and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                
                # Convert to DetectionResult objects
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(classes[i])
                    confidence = float(confs[i])
                    
                    # Get class name
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    # Get mask if available
                    mask = None
                    if masks is not None and i < len(masks):
                        mask = masks[i]
                    
                    detections.append(DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        mask=mask
                    ))
        
        if self.verbose:
            print(f"Detected {len(detections)} objects")
        
        return detections
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        return_masks: bool = True
    ) -> List[List[DetectionResult]]:
        """
        Detect food items in batch of images
        
        Args:
            images: List of image paths or numpy arrays
            return_masks: Return segmentation masks if available
            
        Returns:
            List of detection lists (one per image)
        """
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        
        all_detections = []
        
        for result in results:
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                masks = None
                if return_masks and self.is_segmentation and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(classes[i])
                    confidence = float(confs[i])
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    mask = None
                    if masks is not None and i < len(masks):
                        mask = masks[i]
                    
                    detections.append(DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        mask=mask
                    ))
            
            all_detections.append(detections)
        
        return all_detections
    
    def get_model_info(self) -> dict:
        """Get model information"""
        info = {
            "model_path": self.model_path,
            "is_segmentation": self.is_segmentation,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": len(self.class_names),
        }
        
        if hasattr(self.model, 'model'):
            import torch
            model = self.model.model
            info["parameters"] = sum(p.numel() for p in model.parameters())
            info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return info


def create_detector(
    model_path: str = "yolo26n-seg.pt",
    conf_threshold: float = 0.25,
    device: str = "auto",
    **kwargs
) -> FoodDetector:
    """
    Factory function to create a FoodDetector
    
    Args:
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold
        device: Device to run on
        **kwargs: Additional arguments for FoodDetector
        
    Returns:
        FoodDetector instance
    """
    return FoodDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        **kwargs
    )
