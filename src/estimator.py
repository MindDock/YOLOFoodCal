"""
Calorie Estimator Module - Main Estimation Logic
"""

from typing import List, Optional, Dict
from dataclasses import dataclass

import numpy as np

from .detector import DetectionResult, FoodDetector
from .nutrition_db import NutritionDatabase, NutritionInfo
from .portion_estimator import PortionEstimator, PortionResult


@dataclass
class FoodItem:
    """Estimated food item with nutrition information"""
    name: str
    name_en: str
    category: str
    confidence: float
    
    # Portion
    portion_grams: float
    portion_ml: float
    area_px: int
    area_ratio: float
    
    # Nutrition (per 100g)
    calories_per_100g: float
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float
    
    # Total nutrition
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    
    # Metadata
    food_key: str
    estimation_confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "name_en": self.name_en,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "portion": {
                "grams": round(self.portion_grams, 1),
                "ml": round(self.portion_ml, 1),
                "area_px": self.area_px,
                "area_ratio": round(self.area_ratio, 4)
            },
            "nutrition_per_100g": {
                "calories": self.calories_per_100g,
                "protein": self.protein_per_100g,
                "carbs": self.carbs_per_100g,
                "fat": self.fat_per_100g
            },
            "total_nutrition": {
                "calories": round(self.total_calories, 1),
                "protein": round(self.total_protein, 1),
                "carbs": round(self.total_carbs, 1),
                "fat": round(self.total_fat, 1)
            },
            "estimation_confidence": round(self.estimation_confidence, 3)
        }
    
    def __repr__(self):
        return f"FoodItem({self.name_en}, {self.total_calories:.1f}kcal, {self.portion_grams:.1f}g)"


class CalorieEstimator:
    """
    Main Calorie Estimator
    
    Combines detection, portion estimation, and nutrition lookup
    to estimate calorie and nutrition information for food images.
    """
    
    def __init__(
        self,
        detector: FoodDetector,
        nutrition_db: NutritionDatabase,
        portion_estimator: Optional[PortionEstimator] = None,
        default_portion_grams: float = 100.0,
        portion_multiplier: float = 1.0
    ):
        """
        Initialize Calorie Estimator
        
        Args:
            detector: FoodDetector instance
            nutrition_db: NutritionDatabase instance
            portion_estimator: PortionEstimator instance
            default_portion_grams: Default portion size when estimation fails
            portion_multiplier: Global multiplier for portion sizes
        """
        self.detector = detector
        self.nutrition_db = nutrition_db
        self.portion_estimator = portion_estimator or PortionEstimator()
        self.default_portion_grams = default_portion_grams
        self.portion_multiplier = portion_multiplier
        
        # Class name mapping for the detector
        # Maps detector class names to nutrition database keys
        self.class_mapping: Dict[str, str] = self._build_class_mapping()
    
    def _build_class_mapping(self) -> Dict[str, str]:
        """Build mapping from detector class names to nutrition keys"""
        mapping = {}
        
        # Get all available foods from nutrition database
        all_foods = self.nutrition_db.get_all_foods()
        
        # Try to match detector classes with nutrition database
        for class_name in self.detector.class_names:
            class_name_lower = class_name.lower()
            
            # Direct match
            if class_name_lower in all_foods:
                mapping[class_name] = class_name_lower
                continue
            
            # Try to find in database
            for food_key in all_foods:
                food_info = self.nutrition_db.get_food_info(food_key)
                if food_info:
                    # Match by English name
                    if class_name_lower == food_info.name_en.lower():
                        mapping[class_name] = food_key
                        break
                    # Partial match
                    if class_name_lower in food_info.name_en.lower() or                        food_info.name_en.lower() in class_name_lower:
                        mapping[class_name] = food_key
                        break
        
        return mapping
    
    def map_class_to_nutrition_key(self, class_name: str) -> Optional[str]:
        """Map detector class name to nutrition database key"""
        return self.class_mapping.get(class_name)
    
    def estimate(
        self,
        image,
        portion_multiplier: Optional[float] = None,
        use_mask: bool = True,
        reference_object: Optional[str] = None
    ) -> List[FoodItem]:
        """
        Estimate calories for food in image
        
        Args:
            image: Image path or numpy array
            portion_multiplier: Override default portion multiplier
            use_mask: Use segmentation mask for portion estimation
            reference_object: Reference object name for calibration
            
        Returns:
            List of FoodItem objects
        """
        # Detect food items
        detections = self.detector.detect(image, return_masks=use_mask)
        
        if not detections:
            return []
        
        # Get image size
        import cv2
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        if img is None:
            return []
        
        h, w = img.shape[:2]
        self.portion_estimator.set_image_size(h, w)
        
        # Estimate for each detection
        food_items = []
        multiplier = portion_multiplier or self.portion_multiplier
        
        for det in detections:
            # Get nutrition info
            nutrition_key = self.map_class_to_nutrition_key(det.class_name)
            
            if nutrition_key is None:
                # Try direct lookup
                nutrition_info = self.nutrition_db.lookup(det.class_name)
            else:
                nutrition_info = self.nutrition_db.get_food_info(nutrition_key)
            
            if nutrition_info is None:
                # Use default if not found in database
                nutrition_info = self._get_default_nutrition(det.class_name)
            
            # Estimate portion
            if use_mask and det.mask is not None:
                portion_result = self.portion_estimator.estimate_from_mask(
                    det.mask,
                    px_to_gram_factor=nutrition_info.px_to_gram_factor,
                    food_density=nutrition_info.density
                )
            else:
                portion_result = self.portion_estimator.estimate_from_bbox(
                    det.bbox,
                    px_to_gram_factor=nutrition_info.px_to_gram_factor,
                    food_density=nutrition_info.density
                )
            
            # Apply multiplier
            portion_grams = portion_result.estimated_grams * multiplier
            portion_grams = max(portion_grams, self.default_portion_grams * 0.1)  # Minimum 10g
            portion_ml = portion_result.estimated_volume_ml * multiplier
            
            # Calculate total nutrition
            portion_ratio = portion_grams / 100.0
            total_calories = nutrition_info.calories * portion_ratio
            total_protein = nutrition_info.protein * portion_ratio
            total_carbs = nutrition_info.carbs * portion_ratio
            total_fat = nutrition_info.fat * portion_ratio
            
            # Combine confidences
            estimation_confidence = det.confidence * portion_result.confidence
            
            food_item = FoodItem(
                name=nutrition_info.name,
                name_en=nutrition_info.name_en,
                category=nutrition_info.category,
                confidence=det.confidence,
                portion_grams=portion_grams,
                portion_ml=portion_ml,
                area_px=portion_result.area_px,
                area_ratio=portion_result.area_ratio,
                calories_per_100g=nutrition_info.calories,
                protein_per_100g=nutrition_info.protein,
                carbs_per_100g=nutrition_info.carbs,
                fat_per_100g=nutrition_info.fat,
                total_calories=total_calories,
                total_protein=total_protein,
                total_carbs=total_carbs,
                total_fat=total_fat,
                food_key=nutrition_key or det.class_name,
                estimation_confidence=estimation_confidence
            )
            
            food_items.append(food_item)
        
        return food_items
    
    def _get_default_nutrition(self, class_name: str) -> NutritionInfo:
        """Get default nutrition info for unknown foods"""
        # Return a generic "mixed food" estimate
        return NutritionInfo(
            name=class_name,
            name_en=class_name,
            category='unknown',
            calories=150,  # Default estimate
            protein=5.0,
            carbs=20.0,
            fat=5.0,
            px_to_gram_factor=0.015,
            cooked=True,
            density=1.0
        )
    
    def get_summary(self, food_items: List[FoodItem]) -> dict:
        """
        Get summary of all food items
        
        Args:
            food_items: List of FoodItem objects
            
        Returns:
            Summary dictionary
        """
        if not food_items:
            return {
                "total_items": 0,
                "total_calories": 0,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fat": 0,
                "items": []
            }
        
        return {
            "total_items": len(food_items),
            "total_calories": round(sum(f.total_calories for f in food_items), 1),
            "total_protein": round(sum(f.total_protein for f in food_items), 1),
            "total_carbs": round(sum(f.total_carbs for f in food_items), 1),
            "total_fat": round(sum(f.total_fat for f in food_items), 1),
            "items": [f.to_dict() for f in food_items]
        }


def create_estimator(
    model_path: str = "yolo26n-seg.pt",
    nutrition_path: str = "data/nutrition_table.json",
    conf_threshold: float = 0.25,
    device: str = "auto",
    **kwargs
) -> CalorieEstimator:
    """
    Factory function to create a CalorieEstimator
    
    Args:
        model_path: Path to YOLO model
        nutrition_path: Path to nutrition JSON
        conf_threshold: Confidence threshold
        device: Device to run on
        **kwargs: Additional arguments
        
    Returns:
        CalorieEstimator instance
    """
    from .detector import FoodDetector
    
    detector = FoodDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        verbose=False
    )
    
    nutrition_db = NutritionDatabase(json_path=nutrition_path)
    portion_estimator = PortionEstimator()
    
    return CalorieEstimator(
        detector=detector,
        nutrition_db=nutrition_db,
        portion_estimator=portion_estimator,
        **kwargs
    )
