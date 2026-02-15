#!/usr/bin/env python3
"""
Food Model Training Script
Usage:
    python scripts/train_food_model.py --dataset <path> --classes "rice,noodles,bread" --epochs 50
"""

import argparse
import os
import sys
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Food Detection Model")
    parser.add_argument(
        "--dataset", "-d", type=str, required=True, help="Path to dataset (YOLO format)"
    )
    parser.add_argument(
        "--classes", "-c", type=str, required=True, help="Comma-separated class names"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="yolo26n-seg.pt", help="Base model"
    )
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", "-s", type=int, default=640, help="Image size")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output", "-o", type=str, default="models/custom", help="Output directory"
    )
    return parser.parse_args()


def create_dataset_yaml(classes: list, output_path: str = "data/food.yaml"):
    """Create dataset YAML configuration"""
    content = f"""# Food Detection Dataset
path: .
train: train/images
val: valid/images
test: test/images

# Classes
names:
"""
    for i, cls in enumerate(classes):
        content += f"  {i}: {cls}\n"

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Created dataset config: {output_path}")
    return output_path


def train_model(args):
    """Train YOLO model"""
    from ultralytics import YOLO

    print(f"Starting training with {args.model}")
    print(f"Classes: {args.classes}")
    print(f"Epochs: {args.epochs}")

    # Create dataset YAML
    classes = [c.strip() for c in args.classes.split(",")]
    yaml_path = create_dataset_yaml(classes)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.output,
        name="train",
        exist_ok=True,
        verbose=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {args.output}/train/weights/best.pt")

    return results


def main():
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
