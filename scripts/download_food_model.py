#!/usr/bin/env python3
"""
Download Pre-trained Food Model
Downloads a pre-trained food detection model from Roboflow and integrates it with YOLOFoodCal.

Usage:
    python scripts/download_food_model.py --model roboflow+model_id
    python scripts/download_food_model.py --model ahmad-nabil/food-detection-for-yolo-training
"""

import argparse
import os
import sys
import zipfile
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download Pre-trained Food Model")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="ahmad-nabil/food-detection-for-yolo-training",
        help="Roboflow model ID or dataset name",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="models", help="Output directory"
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m"],
        help="Model size",
    )
    return parser.parse_args()


def download_from_roboflow(model_name: str, output_dir: str, model_size: str):
    """Download model from Roboflow"""
    try:
        from roboflow import Roboflow

        print(f"Downloading model: {model_name}")

        # Initialize Roboflow
        rf = Roboflow(api_key="")  # Will prompt for API key if needed

        # Get project
        project = rf.workspace().project(model_name)

        # Get latest version
        version = project.version()

        # Download model (YOLO format)
        model = version.model
        model_path = model.download(model_size)

        print(f"Model downloaded to: {model_path}")
        return model_path

    except ImportError:
        print("Roboflow SDK not installed. Installing...")
        os.system("pip install roboflow")
        return download_from_roboflow(model_name, output_dir, model_size)
    except Exception as e:
        print(f"Error: {e}")
        print("\nAlternative: Use Roboflow web interface to download model")
        print(f"1. Go to: https://universe.roboflow.com/{model_name}")
        print("2. Click 'Download'")
        print("3. Select YOLO format")
        return None


def download_public_dataset(dataset_name: str, output_dir: str):
    """Download a public dataset and use for inference"""
    print(f"\nDownloading dataset: {dataset_name}")

    # Common public food datasets
    datasets = {
        "ahmad-nabil/food-detection-for-yolo-training": {
            "url": "https://universe.roboflow.com/ds/XXXXXXXXX",  # Need API key
            "classes": [
                "cake",
                "chicken curry",
                "croissant",
                "french fries",
                "fried chicken",
                "fried rice",
                "hamburger",
                "noodles",
                "pasta",
                "pizza",
                "roast chicken",
                "waffle",
            ],
        },
    }

    print(f"\nAvailable public food datasets:")
    print("1. ahmad-nabil/food-detection-for-yolo-training (12 classes)")
    print("2. food-image-classification/food-imgae-yolo (85 classes)")
    print("3. foods-project/foods-project-2 (various)")

    print("\nTo download, you need a Roboflow API key:")
    print("1. Go to https://app.roboflow.com/settings/api")
    print("2. Copy your API key")
    print("3. Use: roboflow api_key <your_key>")

    return None


def use_ultralytics_model():
    """Use Ultralytics pre-trained models with food classes"""
    from ultralytics import YOLO

    print("\n" + "=" * 50)
    print("Using Ultralytics YOLO Models")
    print("=" * 50)

    # Try to load YOLO26n-seg (general purpose, can detect some foods)
    print("\nOption 1: Use YOLO26n-seg (COCO - detects pizza, banana, apple, etc.)")
    print("  Model: yolo26n-seg.pt")
    print("  Classes: 80 (including pizza, burger, hot dog, etc.)")

    print("\nOption 2: Use YOLO11n-seg (also COCO)")
    print("  Model: yolo11n-seg.pt")

    # Test what COCO food classes are available
    coco_foods = {
        52: "pizza",
        53: "pizza",
        54: "banana",
        55: "apple",
        56: "sandwich",
        57: "orange",
        58: "broccoli",
        59: "carrot",
        60: "hot dog",
        61: "pizza",
        62: "donut",
        63: "cake",
    }

    print("\nCOCO food classes available:")
    for cls_id, name in coco_foods.items():
        print(f"  Class {cls_id}: {name}")

    return ["yolo26n-seg.pt"]


def main():
    args = parse_args()

    print("=" * 50)
    print("YOLOFoodCal - Model Downloader")
    print("=" * 50)

    # Method 1: Use Ultralytics built-in models
    use_ultralytics_model()

    print("\n" + "=" * 50)
    print("Option 2: Download from Roboflow")
    print("=" * 50)

    # Method 2: Try to download from Roboflow
    if args.model:
        result = download_from_roboflow(args.model, args.output, args.format)
        if result:
            print(f"\nSuccess! Model saved to: {result}")

    print("\n" + "=" * 50)
    print("Next Steps")
    print("=" * 50)
    print("""
1. For quick testing - use COCO model (already works):
   python apps/cli_demo.py --image data/sample_images/test_pizza.jpg

2. For custom food detection:
   - Option A: Use Roboflow to create dataset and train
   - Option B: Use existing Roboflow food dataset
   - Option C: Manually label your images

3. Update nutrition database:
   Edit data/nutrition_table.json to add your food classes
""")


if __name__ == "__main__":
    main()
