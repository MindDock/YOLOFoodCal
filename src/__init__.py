"""
YOLOFoodCal - Lightweight AI Food Detection & Calorie Estimation
"""

__version__ = "0.1.0"
__author__ = "YOLOFoodCal Team"

from .detector import FoodDetector
from .estimator import CalorieEstimator
from .nutrition_db import NutritionDatabase
from .portion_estimator import PortionEstimator

__all__ = [
    "FoodDetector",
    "CalorieEstimator", 
    "NutritionDatabase",
    "PortionEstimator",
]
