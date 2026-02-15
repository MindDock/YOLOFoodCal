"""
Nutrition Database Module - JSON-based Food Nutrition Data
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NutritionInfo:
    """Nutrition information for 100g of food"""
    name: str
    name_en: str
    category: str
    calories: float
    protein: float
    carbs: float
    fat: float
    px_to_gram_factor: float  # Pixel area to grams conversion factor
    cooked: bool
    density: float  # g/ml
    
    def __repr__(self):
        return f"NutritionInfo({self.name_en}, {self.calories}kcal/100g)"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "name_en": self.name_en,
            "category": self.category,
            "calories": self.calories,
            "protein": self.protein,
            "carbs": self.carbs,
            "fat": self.fat,
            "px_to_gram_factor": self.px_to_gram_factor,
            "cooked": self.cooked,
            "density": self.density
        }


class NutritionDatabase:
    """
    JSON-based Nutrition Database for Food Calorie Estimation
    
    Loads nutrition data from JSON file and provides lookup functions.
    """
    
    def __init__(self, json_path: str = "data/nutrition_table.json"):
        """
        Initialize Nutrition Database
        
        Args:
            json_path: Path to nutrition JSON file
        """
        self.json_path = json_path
        self.data: Dict[str, dict] = {}
        self.version: str = ""
        self.description: str = ""
        
        self._load_json()
    
    def _load_json(self):
        """Load nutrition data from JSON file"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Nutrition data not found at {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.version = raw_data.get('version', '1.0')
        self.description = raw_data.get('description', '')
        self.data = raw_data.get('foods', {})
        
        # Build name mappings for fuzzy matching
        self._build_mappings()
    
    def _build_mappings(self):
        """Build mappings for fuzzy matching"""
        # Map English names to keys
        self.en_to_key: Dict[str, str] = {}
        # Map Chinese names to keys
        self.cn_to_key: Dict[str, str] = {}
        # Map lowercase English names to keys
        self.en_lower_to_key: Dict[str, str] = {}
        
        for key, info in self.data.items():
            if 'name_en' in info:
                self.en_to_key[info['name_en'].lower()] = key
                self.en_lower_to_key[info['name_en'].lower()] = key
            if 'name' in info:
                self.cn_to_key[info['name']] = key
    
    def lookup(self, food_name: str) -> Optional[NutritionInfo]:
        """
        Lookup nutrition info by food name
        
        Args:
            food_name: Food name (English, Chinese, or key)
            
        Returns:
            NutritionInfo object or None if not found
        """
        food_name = food_name.strip()
        if not food_name:
            return None
        
        # Direct key match
        if food_name in self.data:
            return self._create_nutrition_info(food_name)
        
        # Case-insensitive English name match
        food_name_lower = food_name.lower()
        if food_name_lower in self.en_lower_to_key:
            key = self.en_lower_to_key[food_name_lower]
            return self._create_nutrition_info(key)
        
        # Fuzzy match - contains
        for key, info in self.data.items():
            if food_name_lower in key.lower():
                return self._create_nutrition_info(key)
            if 'name_en' in info and food_name_lower in info['name_en'].lower():
                return self._create_nutrition_info(key)
            if 'name' in info and food_name in info['name']:
                return self._create_nutrition_info(key)
        
        return None
    
    def _create_nutrition_info(self, key: str) -> NutritionInfo:
        """Create NutritionInfo object from data"""
        info = self.data[key]
        return NutritionInfo(
            name=info['name'],
            name_en=info['name_en'],
            category=info['category'],
            calories=info['calories'],
            protein=info['protein'],
            carbs=info['carbs'],
            fat=info['fat'],
            px_to_gram_factor=info.get('px_to_gram_factor', 0.015),
            cooked=info.get('cooked', True),
            density=info.get('density', 1.0)
        )
    
    def get_categories(self) -> List[str]:
        """Get all available food categories"""
        categories = set()
        for info in self.data.values():
            if 'category' in info:
                categories.add(info['category'])
        return sorted(list(categories))
    
    def get_foods_by_category(self, category: str) -> List[str]:
        """Get all food keys in a category"""
        foods = []
        for key, info in self.data.items():
            if info.get('category') == category:
                foods.append(key)
        return foods
    
    def get_all_foods(self) -> List[str]:
        """Get all available food keys"""
        return list(self.data.keys())
    
    def get_food_info(self, food_key: str) -> Optional[NutritionInfo]:
        """Get nutrition info by key"""
        if food_key in self.data:
            return self._create_nutrition_info(food_key)
        return None
    
    def get_all_nutrition(self) -> Dict[str, NutritionInfo]:
        """Get all nutrition info as a dictionary"""
        return {key: self._create_nutrition_info(key) for key in self.data.keys()}
    
    def add_food(self, key: str, nutrition_info: NutritionInfo):
        """Add a new food to the database"""
        self.data[key] = nutrition_info.to_dict()
        self._build_mappings()
    
    def save(self, output_path: Optional[str] = None):
        """Save database to JSON file"""
        path = output_path or self.json_path
        
        data = {
            'version': self.version,
            'description': self.description,
            'foods': self.data
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def __len__(self):
        """Return number of foods in database"""
        return len(self.data)
    
    def __repr__(self):
        return f"NutritionDatabase({len(self.data)} foods)"


def create_nutrition_db(json_path: str = "data/nutrition_table.json") -> NutritionDatabase:
    """
    Factory function to create a NutritionDatabase
    
    Args:
        json_path: Path to nutrition JSON file
        
    Returns:
        NutritionDatabase instance
    """
    return NutritionDatabase(json_path=json_path)
