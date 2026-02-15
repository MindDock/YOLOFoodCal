#!/usr/bin/env python3
"""
CLI Demo - Command Line Interface for Food Detection & Calorie Estimation

Usage:
    python apps/cli_demo.py --image path/to/image.jpg
    python apps/cli_demo.py --image path/to/image.jpg --output results/
    python apps/cli_demo.py --webcam
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from src.detector import FoodDetector
from src.estimator import CalorieEstimator
from src.nutrition_db import NutritionDatabase
from src.portion_estimator import PortionEstimator
from src.visualizer import create_result_image, draw_food_item
from src.utils import load_image, save_json, get_timestamp, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOFoodCal - Food Detection & Calorie Estimation"
    )

    parser.add_argument("--image", "-i", type=str, help="Path to input image")
    parser.add_argument("--webcam", "-w", action="store_true", help="Use webcam")
    parser.add_argument("--video", "-v", type=str, help="Path to input video")
    parser.add_argument("--model", "-m", type=str, default="models/yolo11n-food.pt")
    parser.add_argument("--conf", "-c", type=float, default=0.25)
    parser.add_argument("--output", "-o", type=str, default="outputs")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )

    return parser.parse_args()


def process_image(
    estimator, image_path, output_dir, save_image=False, save_json=False, use_mask=True
):
    print(f"Processing: {image_path}")

    from src.utils import load_image

    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"Image shape: {image.shape}")

    food_items = estimator.estimate(image, use_mask=use_mask)

    if not food_items:
        print("No food detected!")
        return

    detections = estimator.detector.detect(image, return_masks=use_mask)

    print("\n" + "=" * 50)
    print("DETECTION RESULTS")
    print("=" * 50)

    for item in food_items:
        print(f"\n{item.name_en} ({item.name})")
        print(f"  Confidence: {item.confidence:.2f}")
        print(f"  Portion: {item.portion_grams:.1f}g")
        print(f"  Calories: {item.total_calories:.1f} kcal")

    summary = estimator.get_summary(food_items)
    print("\n" + "=" * 50)
    print("TOTAL")
    print("=" * 50)
    print(f"Items: {summary['total_items']}")
    print(f"Calories: {summary['total_calories']:.1f} kcal")
    print(f"Protein: {summary['total_protein']:.1f}g")
    print(f"Carbs: {summary['total_carbs']:.1f}g")
    print(f"Fat: {summary['total_fat']:.1f}g")

    if save_image or save_json:
        from src.utils import ensure_dir, get_timestamp

        ensure_dir(output_dir)

        filename = Path(image_path).stem
        timestamp = get_timestamp()

    if save_image:
        from src.visualizer import create_result_image

        result_img = create_result_image(
            image, detections, food_items, show_masks=use_mask, show_summary=True
        )
        output_path = os.path.join(output_dir, f"{filename}_result_{timestamp}.jpg")
        cv2.imwrite(output_path, result_img)
        print(f"\nSaved result image: {output_path}")

    if save_json:
        result_data = {"image": image_path, "timestamp": timestamp, "summary": summary}
        output_path = os.path.join(output_dir, f"{filename}_result_{timestamp}.json")
        from src.utils import save_json as sj

        sj(result_data, output_path)
        print(f"Saved result JSON: {output_path}")


def main():
    args = parse_args()

    if not args.image and not args.webcam and not args.video:
        print("Error: Please specify --image, --webcam, or --video")
        sys.exit(1)

    print("Initializing YOLOFoodCal...")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")

    try:
        detector = FoodDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device,
            verbose=True,
        )
        nutrition_db = NutritionDatabase("data/nutrition_table.json")
        portion_estimator = PortionEstimator()

        estimator = CalorieEstimator(
            detector=detector,
            nutrition_db=nutrition_db,
            portion_estimator=portion_estimator,
        )
        print("Initialization complete!")

    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    use_mask = not args.no_mask

    if args.image:
        process_image(
            estimator,
            args.image,
            args.output,
            save_image=args.save_image,
            save_json=args.save_json,
            use_mask=use_mask,
        )


if __name__ == "__main__":
    main()
