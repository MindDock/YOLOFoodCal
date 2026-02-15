
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.detector import FoodDetector
from src.estimator import CalorieEstimator
from src.nutrition_db import NutritionDatabase
from src.portion_estimator import PortionEstimator

def test_model(model_path, model_name):
    print(f"\n{'='*20}\nTesting Model: {model_name} ({model_path})\n{'='*20}")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    detector = FoodDetector(
        model_path=model_path,
        conf_threshold=0.10, # Very low threshold to see if ANYTHING is detected
        device="auto",
        verbose=True,
    )
    
    print(f"Classes ({len(detector.class_names)}): {detector.class_names[:10]}...")

    # Use extended nutrition database
    nutrition_path = "data/nutrition_table_extended.json"
    if not os.path.exists(nutrition_path):
        nutrition_path = "data/nutrition_table.json"

    nutrition_db = NutritionDatabase(nutrition_path)
    portion_estimator = PortionEstimator()

    estimator = CalorieEstimator(
        detector=detector,
        nutrition_db=nutrition_db,
        portion_estimator=portion_estimator,
    )

    # Test images
    test_images = [
        "data/sample_images/test_burger.jpg",
        "data/sample_images/test_pizza.jpg"
    ]

    for img_path in test_images:
        print(f"\nProcessing {img_path}...")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Pass path directly to see if it makes a difference
        food_items = estimator.estimate(img_path, use_mask=True)
        
        if not food_items:
            print(f"❌ No food detected in {img_path}")
        else:
            print(f"✅ Detected {len(food_items)} items in {img_path}:")
            for item in food_items:
                print(f"  - {item.name_en} (conf={item.confidence:.2f}, cat={item.category})")

if __name__ == "__main__":
    # Test 1: The model being used by default
    test_model("models/yolo11n-food.pt", "Default Food Model")
    
    # Test 2: The fallback model (COCO)
    test_model("yolo26n-seg.pt", "Fallback COCO Model")
