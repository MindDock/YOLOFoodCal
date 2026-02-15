"""
Visualizer Module - Draw Detection Results and Nutrition Info
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

from .detector import DetectionResult
from .estimator import FoodItem


# Color palette for different food categories
CATEGORY_COLORS = {
    'staple': (255, 100, 0),      # Orange
    'meat': (255, 0, 0),          # Red
    'fruit': (0, 255, 0),         # Green
    'vegetable': (0, 255, 100),   # Light Green
    'snack': (255, 255, 0),       # Yellow
    'drink': (0, 100, 255),       # Blue
    'unknown': (128, 128, 128),   # Gray
}


def get_category_color(category: str) -> Tuple[int, int, int]:
    """Get color for food category"""
    return CATEGORY_COLORS.get(category.lower(), CATEGORY_COLORS['unknown'])


def draw_detection(
    image: np.ndarray,
    detection: DetectionResult,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
    show_confidence: bool = True,
    show_label: bool = True
) -> np.ndarray:
    """
    Draw a single detection on image
    
    Args:
        image: Input image (will be modified in-place)
        detection: DetectionResult object
        color: BGR color tuple (auto-determined if None)
        thickness: Line thickness
        show_confidence: Show confidence score
        show_label: Show class label
        
    Returns:
        Image with detection drawn
    """
    x1, y1, x2, y2 = detection.bbox
    
    # Get color from category if not provided
    if color is None:
        color = (0, 255, 0)  # Default green
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    if show_label:
        label = detection.class_name
        if show_confidence:
            label += f" {detection.confidence:.2f}"
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4
) -> np.ndarray:
    """
    Draw segmentation mask on image
    
    Args:
        image: Input image
        mask: Binary segmentation mask
        color: BGR color tuple
        alpha: Transparency factor
        
    Returns:
        Image with mask drawn
    """
    # Ensure mask is the right size
    if len(mask.shape) == 2:
        mask_h, mask_w = mask.shape
    else:
        mask_h, mask_w = mask.shape[:2]
    
    img_h, img_w = image.shape[:2]
    
    # Resize mask if needed
    if mask_h != img_h or mask_w != img_w:
        mask_resized = cv2.resize(mask, (img_w, img_h))
    else:
        mask_resized = mask
    
    # Ensure binary mask
    if mask_resized.dtype != np.uint8:
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
    else:
        mask_binary = mask_resized
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:] = color
    
    # Apply mask
    mask_3ch = cv2.merge([mask_binary, mask_binary, mask_binary])
    masked_color = cv2.bitwise_and(colored_mask, mask_3ch)
    
    # Blend with original image
    image = cv2.addWeighted(image, 1, masked_color, alpha, 0)
    
    return image


def draw_food_item(
    image: np.ndarray,
    food_item: FoodItem,
    position: Tuple[int, int] = (10, 30),
    show_details: bool = True
) -> np.ndarray:
    """
    Draw food item nutrition information on image
    
    Args:
        image: Input image
        food_item: FoodItem object
        position: Starting position (x, y)
        show_details: Show detailed nutrition info
        
    Returns:
        Image with info drawn
    """
    x, y = position
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    # Get category color
    color = get_category_color(food_item.category)
    
    # Draw name and calories
    text = f"{food_item.name_en}: {food_item.total_calories:.0f} kcal"
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    y += line_height
    
    # Draw portion
    text = f"  {food_item.portion_grams:.0f}g"
    cv2.putText(image, text, (x, y), font, font_scale * 0.8, (200, 200, 200), thickness)
    y += line_height
    
    if show_details:
        # Draw macros
        text = f"  P: {food_item.total_protein:.1f}g  C: {food_item.total_carbs:.1f}g  F: {food_item.total_fat:.1f}g"
        cv2.putText(image, text, (x, y), font, font_scale * 0.8, (180, 180, 180), thickness)
    
    return image


def draw_summary(
    image: np.ndarray,
    food_items: List[FoodItem],
    position: str = 'bottom'
) -> np.ndarray:
    """
    Draw summary panel with all food items
    
    Args:
        image: Input image
        food_items: List of FoodItem objects
        position: 'bottom' or 'right'
        
    Returns:
        Image with summary drawn
    """
    if not food_items:
        return image
    
    img_h, img_w = image.shape[:2]
    
    if position == 'bottom':
        # Create bottom panel
        panel_height = min(200, 50 + len(food_items) * 45)
        panel = np.ones((panel_height, img_w, 3), dtype=np.uint8) * 30
        
        y = 30
        line_height = 35
        
        # Title
        total_cal = sum(f.total_calories for f in food_items)
        cv2.putText(
            panel,
            f"Total: {total_cal:.0f} kcal",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        y += line_height
        
        # Each food item
        for item in food_items:
            color = get_category_color(item.category)
            text = f"{item.name_en}: {item.total_calories:.0f} kcal ({item.portion_grams:.0f}g)"
            cv2.putText(
                panel,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            y += 25
        
        # Concatenate panel
        image = np.vstack([image, panel])
    
    else:  # right
        # Create right panel
        panel_width = min(300, img_w // 3)
        panel = np.ones((img_h, panel_width, 3), dtype=np.uint8) * 30
        
        y = 30
        line_height = 35
        
        # Title
        total_cal = sum(f.total_calories for f in food_items)
        cv2.putText(
            panel,
            f"Total: {total_cal:.0f} kcal",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y += line_height
        
        # Each food item
        for item in food_items:
            color = get_category_color(item.category)
            text = f"{item.name_en}: {item.total_calories:.0f} kcal"
            cv2.putText(
                panel,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            y += 25
            
            # Portion
            text = f"  {item.portion_grams:.0f}g"
            cv2.putText(
                panel,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (150, 150, 150),
                1
            )
            y += 20
        
        # Concatenate panel
        image = np.hstack([image, panel])
    
    return image


def create_result_image(
    image: np.ndarray,
    detections: List[DetectionResult],
    food_items: List[FoodItem],
    show_masks: bool = True,
    show_summary: bool = True
) -> np.ndarray:
    """
    Create a complete result visualization
    
    Args:
        image: Input image
        detections: List of DetectionResult objects
        food_items: List of FoodItem objects
        show_masks: Show segmentation masks
        show_summary: Show nutrition summary
        
    Returns:
        Visualization image
    """
    # Make a copy
    result = image.copy()
    
    # Draw each detection
    for det, food in zip(detections, food_items):
        # Get color from category
        color = get_category_color(food.category)
        
        # Draw mask if available
        if show_masks and det.mask is not None:
            result = draw_mask(result, det.mask, color, alpha=0.4)
        
        # Draw bounding box
        result = draw_detection(result, det, color=color)
    
    # Draw summary panel
    if show_summary and food_items:
        result = draw_summary(result, food_items, position='bottom')
    
    return result


def save_result(
    image: np.ndarray,
    output_path: str,
    detections: List[DetectionResult] = None,
    food_items: List[FoodItem] = None,
    show_masks: bool = True,
    show_summary: bool = True
) -> str:
    """
    Save result image to file
    
    Args:
        image: Input image
        output_path: Output file path
        detections: Detection results
        food_items: Food items with nutrition
        show_masks: Show masks
        show_summary: Show summary
        
    Returns:
        Path to saved file
    """
    if detections and food_items:
        result = create_result_image(
            image, detections, food_items,
            show_masks=show_masks,
            show_summary=show_summary
        )
    else:
        result = image
    
    cv2.imwrite(output_path, result)
    return output_path
