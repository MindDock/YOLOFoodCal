#!/usr/bin/env python3
"""
Improved Food Model Retraining Script

Fixes issues from initial training:
- imgsz: 320 -> 640 (match inference resolution)
- epochs: 10 -> 100 (allow proper convergence)
- batch: 4 -> 16 (better gradient estimation)
- augment: enabled (improve generalization)
- patience: 20 (early stopping to avoid overfitting)

Usage:
    python scripts/retrain_food_model.py
    python scripts/retrain_food_model.py --epochs 200 --batch 8
    python scripts/retrain_food_model.py --resume  # resume from last checkpoint
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain Food Detection Model (improved)")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt",
                        help="Base pretrained model to fine-tune from")
    parser.add_argument("--dataset", type=str, default="datasets/food-detection/data.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640, must match inference)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (reduce if OOM, e.g. 8 or 4)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--output", type=str, default="models/food-detect",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="retrain",
                        help="Run name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        print("Please ensure you have the food-detection dataset in datasets/food-detection/")
        sys.exit(1)

    # Auto-detect device
    device = args.device
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = 0  # First CUDA GPU
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            print("WARNING: Training on CPU will be very slow. Consider using GPU.")

    print("=" * 60)
    print("YOLOFoodCal - Model Retraining (Improved)")
    print("=" * 60)
    print(f"  Base model:  {args.base_model}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Image size:  {args.imgsz}")
    print(f"  Batch size:  {args.batch}")
    print(f"  Device:      {device}")
    print(f"  Patience:    {args.patience}")
    print(f"  Output:      {args.output}/{args.name}")
    print("=" * 60)

    from ultralytics import YOLO

    # Load base model
    if args.resume:
        # Resume from last checkpoint
        last_pt = Path(args.output) / args.name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}, starting fresh")
            model = YOLO(args.base_model)
    else:
        model = YOLO(args.base_model)

    # Train with improved parameters
    results = model.train(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.output,
        name=args.name,
        exist_ok=True,
        verbose=True,

        # Early stopping
        patience=args.patience,

        # Augmentation (critical for small datasets)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # Optimizer
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=0.0005,

        # Other
        cos_lr=True,
        pretrained=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,
        plots=True,
    )

    # Copy best model to models/ directory
    best_pt = Path(args.output) / args.name / "weights" / "best.pt"
    output_model = Path("models") / "yolo11n-food.pt"

    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, output_model)
        print(f"\nBest model copied to: {output_model}")

    print("\nTraining complete!")
    print(f"Best model: {best_pt}")
    print(f"\nTo use the retrained model:")
    print(f"  python apps/cli_demo.py --image <image> --model {output_model}")


if __name__ == "__main__":
    main()
