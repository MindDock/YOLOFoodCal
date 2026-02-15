"""
Portion Estimator Module - Estimate Food Portion from Segmentation Mask
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortionResult:
    """Portion estimation result"""
    area_px: int              # Pixel area of the food mask
    area_ratio: float         # Ratio to total image area
    estimated_grams: float    # Estimated weight in grams
    estimated_volume_ml: float # Estimated volume in ml
    confidence: float         # Confidence of the estimation
    
    def __repr__(self):
        return f"PortionResult({self.estimated_grams:.1f}g, {self.estimated_volume_ml:.1f}ml)"


class PortionEstimator:
    """
    Estimate food portion size from segmentation mask
    
    Uses pixel area to estimate weight and volume based on
    food-specific conversion factors.
    """
    
    def __init__(
        self,
        reference_area_px: Optional[int] = None,
        default_px_to_gram: float = 0.015,
        image_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize Portion Estimator
        
        Args:
            reference_area_px: Known reference object area in pixels (optional)
                              If provided, enables more accurate estimation
            default_px_to_gram: Default pixel-to-gram conversion factor
            image_size: Expected input image size (height, width)
        """
        self.reference_area_px = reference_area_px
        self.default_px_to_gram = default_px_to_gram
        self.image_size = image_size
        self.total_pixels = image_size[0] * image_size[1]
        
        # Reference object sizes (for scale estimation)
        # Common reference objects with approximate real-world sizes
        self.reference_objects = {
            'coin': {'diameter_mm': 25, 'area_mm2': 490},
            'credit_card': {'width_mm': 85.6, 'height_mm': 53.98, 'area_mm2': 4618},
            'smartphone': {'width_mm': 75, 'height_mm': 150, 'area_mm2': 11250},
            'hand_palm': {'area_mm2': 8000},  # Average adult palm
            'fist': {'area_mm2': 7000},       # Average adult fist
        }
    
    def estimate_from_mask(
        self,
        mask: np.ndarray,
        px_to_gram_factor: Optional[float] = None,
        food_density: float = 1.0,
        reference_area_px: Optional[int] = None
    ) -> PortionResult:
        """
        Estimate portion size from segmentation mask
        
        Args:
            mask: Binary segmentation mask (HxW or HxWx1)
            px_to_gram_factor: Food-specific pixel-to-gram conversion factor
            food_density: Food density (g/ml), default 1.0
            reference_area_px: Reference object area in pixels (overrides instance default)
            
        Returns:
            PortionResult with estimation
        """
        # Handle different mask formats
        if len(mask.shape) == 3:
            mask = mask.squeeze()
        
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        
        # Calculate pixel area
        area_px = np.sum(mask > 0)
        
        if area_px == 0:
            return PortionResult(
                area_px=0,
                area_ratio=0.0,
                estimated_grams=0.0,
                estimated_volume_ml=0.0,
                confidence=0.0
            )
        
        # Calculate area ratio
        area_ratio = area_px / self.total_pixels
        
        # Determine conversion factor
        factor = px_to_gram_factor if px_to_gram_factor is not None else self.default_px_to_gram
        
        # Use reference object if provided
        ref_area = reference_area_px if reference_area_px is not None else self.reference_area_px
        
        if ref_area is not None and ref_area > 0:
            # Calibrated estimation using reference object
            # Scale factor = known real area / reference pixel area
            # This gives us pixels per mm^2
            known_ref_area_mm2 = 490  # Default: 1 CNY coin
            px_per_mm2 = ref_area / known_ref_area_mm2
            
            # Convert food pixels to mm^2
            food_area_mm2 = area_px / px_per_mm2
            
            # Estimate grams based on food type (density * volume)
            # Assuming average food thickness of 10mm
            thickness_mm = 10
            volume_mm3 = food_area_mm2 * thickness_mm
            volume_ml = volume_mm3 / 1000
            
            estimated_grams = volume_ml * food_density
            
            # Higher confidence with reference
            confidence = 0.9
        else:
            # Simple estimation using pixel-to-gram factor
            # This is a rough approximation
            estimated_grams = area_px * factor
            volume_ml = estimated_grams / food_density
            
            # Lower confidence without reference
            confidence = 0.6
        
        return PortionResult(
            area_px=int(area_px),
            area_ratio=float(area_ratio),
            estimated_grams=float(estimated_grams),
            estimated_volume_ml=float(volume_ml),
            confidence=float(confidence)
        )
    
    def estimate_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        px_to_gram_factor: Optional[float] = None,
        food_density: float = 1.0
    ) -> PortionResult:
        """
        Estimate portion size from bounding box (simplified version)
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            px_to_gram_factor: Food-specific pixel-to-gram conversion factor
            food_density: Food density (g/ml)
            
        Returns:
            PortionResult with estimation
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Use bounding box area as approximation
        area_px = width * height
        area_ratio = area_px / self.total_pixels
        
        factor = px_to_gram_factor if px_to_gram_factor is not None else self.default_px_to_gram
        estimated_grams = area_px * factor * 0.5  # Reduce factor for bbox (less accurate)
        volume_ml = estimated_grams / food_density
        
        return PortionResult(
            area_px=int(area_px),
            area_ratio=float(area_ratio),
            estimated_grams=float(estimated_grams),
            estimated_volume_ml=float(volume_ml),
            confidence=0.4  # Lower confidence for bbox-based estimation
        )
    
    def set_reference_object(self, name: str, real_size_mm2: float):
        """
        Set a reference object for calibration
        
        Args:
            name: Reference object name
            real_size_mm2: Real-world size in mm^2
        """
        self.reference_objects[name] = {'area_mm2': real_size_mm2}
    
    def set_image_size(self, height: int, width: int):
        """Set expected image size"""
        self.image_size = (height, width)
        self.total_pixels = height * width
    
    def calibrate_with_reference(
        self,
        reference_mask: np.ndarray,
        reference_name: str = 'coin'
    ) -> float:
        """
        Calibrate the estimator using a reference object in the image
        
        Args:
            reference_mask: Binary mask of the reference object
            reference_name: Name of the reference object
            
        Returns:
            Calculated pixels per mm^2
        """
        if len(reference_mask.shape) == 3:
            reference_mask = reference_mask.squeeze()
        
        ref_area_px = np.sum(reference_mask > 0)
        
        if reference_name in self.reference_objects:
            real_area_mm2 = self.reference_objects[reference_name].get('area_mm2', 490)
        else:
            real_area_mm2 = 490  # Default coin
        
        px_per_mm2 = ref_area_px / real_area_mm2
        self.reference_area_px = ref_area_px
        
        return px_per_mm2
    
    def estimate_multiplier(
        self,
        area_px: int,
        mode: str = 'medium'
    ) -> float:
        """
        Estimate portion multiplier based on area
        
        Args:
            area_px: Detected food area in pixels
            mode: Portion size mode ('small', 'medium', 'large')
            
        Returns:
            Multiplier value
        """
        # Typical portion sizes in pixels (for 640x640 image)
        typical_medium = 20000  # ~100g for average food
        typical_small = 10000  # ~50g
        typical_large = 40000  # ~200g
        
        multipliers = {
            'small': 0.5,
            'medium': 1.0,
            'large': 1.5
        }
        
        if area_px < typical_small:
            return multipliers.get(mode, 1.0) * 0.5
        elif area_px > typical_large:
            return multipliers.get(mode, 1.0) * 1.5
        else:
            return multipliers.get(mode, 1.0)


def create_portion_estimator(
    reference_area_px: Optional[int] = None,
    default_px_to_gram: float = 0.015,
    image_size: Tuple[int, int] = (640, 640)
) -> PortionEstimator:
    """
    Factory function to create a PortionEstimator
    
    Args:
        reference_area_px: Reference object area in pixels
        default_px_to_gram: Default conversion factor
        image_size: Input image size
        
    Returns:
        PortionEstimator instance
    """
    return PortionEstimator(
        reference_area_px=reference_area_px,
        default_px_to_gram=default_px_to_gram,
        image_size=image_size
    )
