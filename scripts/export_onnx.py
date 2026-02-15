#!/usr/bin/env python3
"""
Export YOLO Model to ONNX Format

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model yolo26n-seg.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument(
        "--model", type=str, default="yolo26n-seg.pt", help="Input model path"
    )
    parser.add_argument(
        "--output", type=str, default="models/onnx", help="Output directory"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shape")
    return parser.parse_args()


def export_onnx(
    model_path, output_dir, imgsz=640, opset=12, simplify=True, dynamic=False
):
    """Export YOLO model to ONNX format"""
    from ultralytics import YOLO

    print(f"Loading model: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Using default YOLO26n-seg model...")
        model_path = "yolo26n-seg.pt"

    # Load model
    model = YOLO(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export
    print(f"Exporting to ONNX (opset={opset})...")
    export_path = model.export(
        format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=dynamic
    )

    print(f"Exported to: {export_path}")

    # Move to output directory
    if export_path and os.path.exists(export_path):
        output_path = os.path.join(output_dir, os.path.basename(export_path))

        if export_path != output_path:
            import shutil

            shutil.move(export_path, output_path)
            print(f"Moved to: {output_path}")

        return output_path

    return None


def main():
    args = parse_args()

    output_path = export_onnx(
        model_path=args.model,
        output_dir=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )

    if output_path:
        print("\n" + "=" * 50)
        print("Export successful!")
        print(f"Output: {output_path}")
        print("=" * 50)
    else:
        print("Export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
